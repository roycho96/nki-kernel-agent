"""
End-to-end benchmark runner and score calculator for end-to-end model evaluation.
Handles main.py output parsing, accuracy verification, and score computation.
"""
import re
import json
import logging
from dataclasses import dataclass, field
from typing import Optional

from infra.ssh_runner import SSHRunner, RemoteResult

logger = logging.getLogger(__name__)


@dataclass
class BenchmarkMetrics:
    """Parsed benchmark results from main.py --mode benchmark"""
    ttft_ms: float = 0.0          # Time to first token
    tokens_per_sec: float = 0.0   # Throughput
    nki_flops: float = 0.0        # NKI FLOPS
    total_flops: float = 0.0      # Total FLOPS
    raw_output: str = ""
    parse_success: bool = False


@dataclass
class CompetitionScore:
    """Competition score breakdown"""
    accuracy: float = 0.0          # binary: 0 or 1
    latency_ratio: float = 0.0    # ref_TTFT / my_TTFT
    throughput_ratio: float = 0.0  # my_tps / ref_tps
    nki_bonus: float = 1.0        # 1 + nki_flops / total_flops
    total_score: float = 0.0


class E2EBenchmark:
    """
    End-to-end benchmark and scoring for NKI kernel optimization.

    Competition score formula:
        score = accuracy(binary)
              × (ref_TTFT / my_TTFT)
              × (my_tokens_per_sec / ref_tokens_per_sec)
              × (1 + nki_flops / total_flops)
    """

    def __init__(self, ssh: SSHRunner, model_path: str, compiled_path: str):
        self.ssh = ssh
        self.model_path = model_path
        self.compiled_path = compiled_path
        self.ref_ttft: Optional[float] = None
        self.ref_tps: Optional[float] = None

    def set_reference(self, ttft_ms: float, tps: float):
        """Set reference (baseline without NKI) metrics"""
        self.ref_ttft = ttft_ms
        self.ref_tps = tps
        logger.info(f"Reference set: TTFT={ttft_ms:.2f}ms, TPS={tps:.2f}")

    def clear_and_recompile(self):
        """Clear caches to force fresh compilation"""
        self.ssh.clear_compiled_model(self.compiled_path)
        self.ssh.clear_compile_cache()

    def check_accuracy(self, prompts: list = None) -> bool:
        """
        Verify model accuracy with one or more prompts.
        Returns True if ALL prompts produce reasonable output.

        TODO: Adapt parsing logic once actual main.py output format is known.
        """
        if prompts is None:
            prompts = ["What is the capital of France?"]

        for prompt in prompts:
            result = self.ssh.run_generate(
                self.model_path, self.compiled_path,
                prompt=prompt, enable_nki=True
            )
            if not result.success:
                logger.error(f"Generation failed: {result.stderr[:200]}")
                return False

            # TODO: Replace with actual accuracy check logic
            # For now, check that output is non-empty and doesn't contain error indicators
            output = result.stdout.strip()
            if not output or "error" in output.lower() or "nan" in output.lower():
                logger.error(f"Accuracy check failed for prompt: {prompt[:50]}...")
                logger.error(f"Output: {output[:200]}")
                return False

        return True

    def run_benchmark(self) -> BenchmarkMetrics:
        """
        Run main.py --mode benchmark and parse results.

        TODO: Adapt parsing logic once actual main.py output format is known.
        The parser below handles common patterns. Update after seeing real output.
        """
        result = self.ssh.run_benchmark(
            self.model_path, self.compiled_path, enable_nki=True
        )

        metrics = BenchmarkMetrics(raw_output=result.stdout)

        if not result.success:
            logger.error(f"Benchmark failed: {result.stderr[:200]}")
            return metrics

        output = result.stdout

        # --- Parse TTFT ---
        # Try common patterns: "TTFT: 123.45 ms", "ttft_ms: 123.45", etc.
        ttft_match = re.search(r'[Tt][Tt][Ff][Tt][:\s_]*(\d+\.?\d*)\s*(?:ms)?', output)
        if ttft_match:
            metrics.ttft_ms = float(ttft_match.group(1))

        # --- Parse throughput ---
        # Try: "tokens/sec: 123.45", "throughput: 123.45", "tps: 123.45"
        tps_match = re.search(r'(?:tokens?[/_]?(?:per[/_])?sec|throughput|tps)[:\s]*(\d+\.?\d*)', output, re.I)
        if tps_match:
            metrics.tokens_per_sec = float(tps_match.group(1))

        # --- Parse NKI FLOPS ratio ---
        # Try: "nki_flops: 123", "nki_flops_ratio: 0.45"
        nki_match = re.search(r'nki[_\s]?flops[:\s]*(\d+\.?\d*)', output, re.I)
        if nki_match:
            metrics.nki_flops = float(nki_match.group(1))
        total_match = re.search(r'total[_\s]?flops[:\s]*(\d+\.?\d*)', output, re.I)
        if total_match:
            metrics.total_flops = float(total_match.group(1))

        metrics.parse_success = metrics.ttft_ms > 0 and metrics.tokens_per_sec > 0
        if not metrics.parse_success:
            logger.warning(f"Could not parse benchmark output. Raw:\n{output[:500]}")

        return metrics

    def compute_score(self, metrics: BenchmarkMetrics, accuracy: bool) -> CompetitionScore:
        """Compute optimization score"""
        score = CompetitionScore()
        score.accuracy = 1.0 if accuracy else 0.0

        if not accuracy or not metrics.parse_success:
            return score

        if self.ref_ttft and self.ref_ttft > 0 and metrics.ttft_ms > 0:
            score.latency_ratio = self.ref_ttft / metrics.ttft_ms

        if self.ref_tps and self.ref_tps > 0 and metrics.tokens_per_sec > 0:
            score.throughput_ratio = metrics.tokens_per_sec / self.ref_tps

        if metrics.total_flops > 0:
            score.nki_bonus = 1.0 + (metrics.nki_flops / metrics.total_flops)
        else:
            score.nki_bonus = 1.0

        score.total_score = (
            score.accuracy
            * score.latency_ratio
            * score.throughput_ratio
            * score.nki_bonus
        )
        return score

    def full_evaluation(self, clear_cache: bool = True) -> tuple:
        """
        Full evaluation pipeline:
        1. Clear caches (optional)
        2. Accuracy check
        3. Benchmark
        4. Score computation

        Returns: (score: CompetitionScore, metrics: BenchmarkMetrics, accuracy: bool)
        """
        if clear_cache:
            self.clear_and_recompile()

        # Step 1: Accuracy
        accuracy = self.check_accuracy()
        if not accuracy:
            logger.error("Accuracy check FAILED — score is 0")
            return CompetitionScore(), BenchmarkMetrics(), False

        # Step 2: Benchmark
        metrics = self.run_benchmark()
        if not metrics.parse_success:
            logger.warning("Benchmark parsing failed, score may be inaccurate")

        # Step 3: Score
        score = self.compute_score(metrics, accuracy)
        logger.info(
            f"Score: {score.total_score:.4f} "
            f"(latency_ratio={score.latency_ratio:.3f}, "
            f"throughput_ratio={score.throughput_ratio:.3f}, "
            f"nki_bonus={score.nki_bonus:.3f})"
        )
        return score, metrics, accuracy
