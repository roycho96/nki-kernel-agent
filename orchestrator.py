"""
NKI MoE Kernel Optimization Orchestrator.

AccelOpt's Planner → Executor → Summarizer loop
+ KernelAgent's Reflexion, Divergence Guard, Error Feedback patterns.

Usage:
    python orchestrator.py --host ubuntu@<trn2-ip> --rounds 50
"""
import json
import re
import logging
import time
import shutil
import argparse
import subprocess
from pathlib import Path
from datetime import datetime

from config import (
    RemoteConfig, CompileConfig, BenchmarkConfig, AgentConfig, PathConfig,
    PROFILE_FIELDS
)
from infra.ssh_runner import SSHRunner
from infra.e2e_benchmark import E2EBenchmark
from ka_extensions.stability import (
    ReflexionManager, DivergenceGuard, AttemptHistory, Attempt
)
from prompts.planner_prompts.construct_base_prompt import inject_memory

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("agent.log"),
    ]
)
logger = logging.getLogger("orchestrator")


def load_prompt(path: str) -> str:
    return Path(path).read_text()


def extract_code(text: str) -> str | None:
    """Extract first Python code block from LLM response"""
    match = re.search(r"```(?:python)?\s*\n(.*?)```", text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return None


def call_llm(system: str, user: str, model: str = "claude-sonnet-4-20250514") -> str:
    """
    Call Claude API via the `claude` CLI tool (Claude Code).
    Falls back to direct API call if CLI not available.

    Adapt this function based on your actual LLM setup:
    - Option A: `claude` CLI (Claude Code)
    - Option B: Anthropic Python SDK
    - Option C: OpenAI-compatible API
    """
    # Option A: Claude CLI
    try:
        full_prompt = f"{system}\n\n---\n\n{user}"
        result = subprocess.run(
            ["claude", "--print", "-p", full_prompt,
             "--model", model, "--max-turns", "1"],
            capture_output=True, text=True, timeout=300
        )
        if result.returncode == 0 and result.stdout.strip():
            return result.stdout.strip()
    except FileNotFoundError:
        pass  # CLI not available, try Option B

    # Option B: Anthropic Python SDK
    try:
        import anthropic
        client = anthropic.Anthropic()
        response = client.messages.create(
            model=model,
            max_tokens=8192,
            system=system,
            messages=[{"role": "user", "content": user}]
        )
        return response.content[0].text
    except ImportError:
        pass
    except Exception as e:
        logger.error(f"Anthropic API error: {e}")

    # Option C: Placeholder — replace with your actual LLM call
    logger.error("No LLM backend available. Install `claude` CLI or `anthropic` package.")
    return ""


class NKIMoEOptimizer:
    """
    Main optimization loop.

    Architecture:
    - AccelOpt: Planner → Executor → Summarizer (3-agent workflow)
    - AccelOpt: Optimization memory (slow→fast experience storage)
    - KernelAgent: Reflexion (structured self-reflection)
    - KernelAgent: Divergence guard (auto-revert on regression)
    - KernelAgent: Error feedback (inject errors into next prompt)
    - KernelAgent: Attempt history (structured experiment tracking)
    """

    def __init__(self, ssh: SSHRunner, e2e: E2EBenchmark, config: AgentConfig,
                 paths: PathConfig, compile_config: CompileConfig):
        self.ssh = ssh
        self.e2e = e2e
        self.config = config
        self.paths = paths
        self.compile_config = compile_config

        # AccelOpt components
        self.optimization_memory: list[dict] = []
        self._load_memory()

        # KernelAgent components
        self.reflexion = ReflexionManager(window=config.reflexion_window)
        self.divergence = DivergenceGuard(threshold_percent=config.divergence_threshold)
        self.history = AttemptHistory(
            log_path=str(paths.resolve(paths.experiments_log)),
            window=config.history_window
        )
        self.history.load_from_log()

        # Prompts
        self.planner_base = load_prompt(str(paths.resolve(paths.planner_base)))
        self.planner_template = load_prompt(str(paths.resolve(paths.planner_template)))
        self.executor_base = load_prompt(str(paths.resolve(paths.executor_base)))
        self.executor_template = load_prompt(str(paths.resolve(paths.executor_template)))
        self.summarizer_base = load_prompt(str(paths.resolve(paths.summarizer_base)))
        self.summarizer_template = load_prompt(str(paths.resolve(paths.summarizer_template)))

        # State
        self.no_improve_count = 0
        self.round_num = len(self.history.attempts)

    # ------------------------------------------------------------------
    # Memory management
    # ------------------------------------------------------------------

    def _load_memory(self):
        mem_path = self.paths.resolve(self.paths.memory_file)
        if mem_path.exists():
            try:
                self.optimization_memory = json.loads(mem_path.read_text())
            except Exception:
                self.optimization_memory = []

    def _save_memory(self):
        mem_path = self.paths.resolve(self.paths.memory_file)
        mem_path.parent.mkdir(parents=True, exist_ok=True)
        mem_path.write_text(json.dumps(self.optimization_memory, indent=2))

    # ------------------------------------------------------------------
    # Checkpoint management
    # ------------------------------------------------------------------

    def save_checkpoint(self, kernel_code: str, round_num: int):
        ckpt_dir = self.paths.resolve(self.paths.checkpoint_dir)
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = ckpt_dir / f"best_r{round_num}_{ts}.py"
        path.write_text(kernel_code)
        logger.info(f"Checkpoint saved: {path}")

    # ------------------------------------------------------------------
    # Core optimization round
    # ------------------------------------------------------------------

    def optimize_round(self, kernel_code: str, problem_code: str) -> str:
        """
        Single optimization round:
        1. Upload + Profile
        2. Plan (AccelOpt planner)
        3. Execute (AccelOpt executor + KA error feedback + reflexion)
        4. Verify + Benchmark
        5. Decide (KA divergence guard)
        6. Summarize (AccelOpt memory + KA reflexion)
        """
        self.round_num += 1
        rn = self.round_num
        logger.info(f"\n{'='*60}\nRound {rn}\n{'='*60}")

        # === Step 1: Upload kernel + run E2E evaluation ===
        kernel_path = self.paths.resolve(self.paths.kernel_file)
        kernel_path.write_text(kernel_code)

        if not self.ssh.upload_file(str(kernel_path)):
            logger.error("Failed to upload kernel")
            return kernel_code

        self.e2e.clear_and_recompile()

        # Accuracy check first (failure = 0 score)
        accuracy = self.e2e.check_accuracy()
        if not accuracy:
            logger.error(f"R{rn}: Accuracy FAILED on current kernel")
            # This shouldn't happen for the current best, but handle gracefully

        # Benchmark
        metrics = self.e2e.run_benchmark()
        current_latency = metrics.ttft_ms if metrics.ttft_ms > 0 else float('inf')

        # Build profile string for planner (from benchmark raw output for now)
        # TODO: Add neuron-profile summary-json parsing when running kernel-level profiling
        profile_str = json.dumps({
            "ttft_ms": metrics.ttft_ms,
            "tokens_per_sec": metrics.tokens_per_sec,
        }, indent=2)

        # === Step 2: PLAN (AccelOpt planner) ===
        logger.info(f"R{rn}: Planning optimization...")
        planner_base = inject_memory(self.planner_base, self.optimization_memory)

        plan = call_llm(
            system=planner_base,
            user=self.planner_template.format(
                problem_code=problem_code,
                kernel_code=kernel_code,
                profile=profile_str
            )
        )

        if not plan:
            logger.error(f"R{rn}: Planner returned empty response")
            return kernel_code

        plan_summary = plan[:100].replace("\n", " ")
        logger.info(f"R{rn}: Plan: {plan_summary}")

        # === Step 3: EXECUTE (AccelOpt executor + KA extensions) ===
        logger.info(f"R{rn}: Executing optimization...")

        # Build error feedback (KA pattern)
        error_feedback = ""
        last_attempt = self.history.attempts[-1] if self.history.attempts else None
        if last_attempt and last_attempt.status in ("correctness_fail", "compile_fail"):
            error_feedback = self.history.format_error_feedback()

        # Build reflexion context (KA pattern)
        reflexion_context = self.reflexion.format_for_prompt()

        new_kernel_response = call_llm(
            system=self.executor_base,
            user=self.executor_template.format(
                problem_code=problem_code,
                kernel_code=kernel_code,
                optimization_plan=plan,
                error_feedback=error_feedback,
                reflexion_context=reflexion_context
            )
        )

        new_kernel_code = extract_code(new_kernel_response)
        if not new_kernel_code:
            logger.error(f"R{rn}: Failed to extract code from executor response")
            self.history.record(Attempt(
                round_num=rn, status="compile_fail",
                error="No code block in executor response",
                plan_summary=plan_summary
            ))
            return kernel_code

        # === Step 4: VERIFY + BENCHMARK new kernel ===
        logger.info(f"R{rn}: Verifying new kernel...")
        kernel_path.write_text(new_kernel_code)

        if not self.ssh.upload_file(str(kernel_path)):
            logger.error(f"R{rn}: Failed to upload new kernel")
            kernel_path.write_text(kernel_code)  # restore
            return kernel_code

        self.e2e.clear_and_recompile()

        # Accuracy check on new kernel
        new_accuracy = self.e2e.check_accuracy()
        if not new_accuracy:
            logger.error(f"R{rn}: New kernel FAILED accuracy check")
            self.history.record(Attempt(
                round_num=rn, status="correctness_fail",
                error="Accuracy check failed after optimization",
                plan_summary=plan_summary
            ))
            self.reflexion.add(
                rn, f"Accuracy failed. Plan: {plan_summary}",
                category="error", plan_summary=plan_summary
            )
            # Restore old kernel
            kernel_path.write_text(kernel_code)
            self.ssh.upload_file(str(kernel_path))
            return kernel_code

        # Benchmark new kernel
        new_metrics = self.e2e.run_benchmark()
        new_latency = new_metrics.ttft_ms if new_metrics.ttft_ms > 0 else float('inf')
        new_score_obj = self.e2e.compute_score(new_metrics, True)
        new_score = new_score_obj.total_score

        logger.info(f"R{rn}: New latency={new_latency:.3f}ms, score={new_score:.4f}")

        # === Step 5: DECIDE (KA divergence guard) ===
        should_revert, divergence = self.divergence.check(new_latency)

        if should_revert:
            self.history.record(Attempt(
                round_num=rn, status="divergence_revert",
                latency_ms=new_latency, score=new_score,
                plan_summary=plan_summary
            ))
            self.reflexion.add(
                rn,
                f"Caused {divergence:.0f}% regression ({new_latency:.3f}ms vs best "
                f"{self.divergence.best_latency:.3f}ms). Plan: {plan_summary}",
                category="regression", plan_summary=plan_summary
            )
            best_code = self.divergence.get_best_kernel()
            if best_code:
                kernel_path.write_text(best_code)
                self.ssh.upload_file(str(kernel_path))
                return best_code
            return kernel_code

        # === Step 6: Compare and decide ===
        if new_latency < self.divergence.best_latency:
            improvement = 0
            if self.divergence.best_latency < float('inf'):
                improvement = (self.divergence.best_latency - new_latency) / self.divergence.best_latency * 100

            logger.info(f"R{rn}: ✅ IMPROVED by {improvement:.1f}%")

            # Update best
            self.divergence.update_best(new_latency, new_kernel_code, rn)
            self.save_checkpoint(new_kernel_code, rn)
            self.no_improve_count = 0

            # === Step 7: SUMMARIZE (AccelOpt memory) ===
            if improvement > 0:
                logger.info(f"R{rn}: Generating optimization summary...")
                summary = call_llm(
                    system=self.summarizer_base,
                    user=self.summarizer_template.format(
                        slow_kernel=kernel_code,
                        fast_kernel=new_kernel_code,
                        speedup=f"{self.divergence.best_latency / new_latency:.2f}x"
                        if self.divergence.best_latency > 0 and self.divergence.best_latency < float('inf')
                        else "N/A (first improvement)"
                    )
                )
                if summary and "No optimization found" not in summary:
                    self.optimization_memory.append({
                        "title": summary[:50],
                        "summary": summary
                    })
                    self._save_memory()

            # KA reflexion
            self.reflexion.add(
                rn, f"Improved {improvement:.1f}%. Plan: {plan_summary}",
                category="improved", plan_summary=plan_summary
            )

            # Record
            self.history.record(Attempt(
                round_num=rn, status="improved",
                latency_ms=new_latency, score=new_score,
                plan_summary=plan_summary
            ))

            return new_kernel_code
        else:
            logger.info(f"R{rn}: ➡️ No improvement ({new_latency:.3f}ms vs best {self.divergence.best_latency:.3f}ms)")
            self.no_improve_count += 1

            self.reflexion.add(
                rn,
                f"No improvement ({new_latency:.3f}ms vs {self.divergence.best_latency:.3f}ms). "
                f"Plan: {plan_summary}",
                category="no_improvement", plan_summary=plan_summary
            )
            self.history.record(Attempt(
                round_num=rn, status="no_improvement",
                latency_ms=new_latency, score=new_score,
                plan_summary=plan_summary
            ))

            # Keep old kernel
            kernel_path.write_text(kernel_code)
            self.ssh.upload_file(str(kernel_path))
            return kernel_code

    def should_stop(self) -> bool:
        if self.no_improve_count >= self.config.no_improve_limit:
            logger.info(f"Stopping: {self.no_improve_count} consecutive rounds without improvement")
            return True
        return False

    # ------------------------------------------------------------------
    # Main run loop
    # ------------------------------------------------------------------

    def run(self, max_rounds: int = None, problem_code: str = ""):
        """Run the full optimization loop"""
        max_rounds = max_rounds or self.config.max_rounds

        # Load current kernel
        kernel_path = self.paths.resolve(self.paths.kernel_file)
        if not kernel_path.exists():
            logger.error(f"Kernel file not found: {kernel_path}")
            return

        kernel_code = kernel_path.read_text()
        logger.info(f"Starting optimization with {len(kernel_code)} char kernel")

        # Initialize divergence guard with current performance
        logger.info("Establishing baseline...")
        kernel_path.write_text(kernel_code)
        self.ssh.upload_file(str(kernel_path))
        self.e2e.clear_and_recompile()

        if self.e2e.check_accuracy():
            baseline_metrics = self.e2e.run_benchmark()
            if baseline_metrics.parse_success:
                self.divergence.update_best(
                    baseline_metrics.ttft_ms, kernel_code, 0
                )
                logger.info(f"Baseline: TTFT={baseline_metrics.ttft_ms:.3f}ms, "
                          f"TPS={baseline_metrics.tokens_per_sec:.2f}")
            else:
                logger.warning("Could not parse baseline benchmark. Continuing anyway.")
        else:
            logger.error("Baseline accuracy check failed! Fix kernel before optimizing.")
            return

        # Main loop
        for i in range(max_rounds):
            try:
                kernel_code = self.optimize_round(kernel_code, problem_code)
            except KeyboardInterrupt:
                logger.info("Interrupted by user")
                break
            except Exception as e:
                logger.error(f"Round error: {e}", exc_info=True)
                continue

            if self.should_stop():
                break

            # Brief pause between rounds
            time.sleep(2)

        # Final summary
        logger.info(f"\n{'='*60}")
        logger.info("OPTIMIZATION COMPLETE")
        logger.info(f"Total rounds: {self.round_num}")
        logger.info(f"Best latency: {self.divergence.best_latency:.3f}ms (round {self.divergence.best_round})")
        logger.info(f"Optimization memories: {len(self.optimization_memory)}")
        logger.info(f"{'='*60}")


def main():
    parser = argparse.ArgumentParser(description="NKI MoE Kernel Optimization Agent")
    parser.add_argument("--host", required=True, help="Trn2 SSH host (e.g., ubuntu@1.2.3.4)")
    parser.add_argument("--key", default="~/.ssh/id_rsa", help="SSH key path")
    parser.add_argument("--rounds", type=int, default=50, help="Max optimization rounds")
    parser.add_argument("--remote-dir", default="~/nki-moe", help="Remote project directory")
    parser.add_argument("--model-path", default="~/qwen-30b-a3b/hf_model")
    parser.add_argument("--compiled-path", default="~/qwen-30b-a3b/traced_model")
    parser.add_argument("--kernel", default="qwen_with_nki.py", help="Kernel file name")
    parser.add_argument("--problem", default="", help="Path to problem description / reference code")
    parser.add_argument("--divergence", type=float, default=50.0, help="Divergence threshold %%")
    parser.add_argument("--no-dge", action="store_true", help="Disable DGE (AccelOpt default)")
    args = parser.parse_args()

    # Setup configs
    remote = RemoteConfig(host=args.host, key_path=args.key,
                          remote_dir=args.remote_dir,
                          model_path=args.model_path,
                          compiled_path=args.compiled_path)
    compile_cfg = CompileConfig(disable_dge=args.no_dge)
    agent_cfg = AgentConfig(max_rounds=args.rounds,
                           divergence_threshold=args.divergence)
    paths = PathConfig(kernel_file=args.kernel)

    # Setup components
    ssh = SSHRunner(host=remote.host, key_path=remote.key_path,
                    remote_dir=remote.remote_dir)

    # Check connection
    if not ssh.check_connection():
        logger.error(f"Cannot connect to {remote.host}. Check SSH config.")
        return

    e2e = E2EBenchmark(ssh, remote.model_path, remote.compiled_path)

    # Load problem code if provided
    problem_code = ""
    if args.problem and Path(args.problem).exists():
        problem_code = Path(args.problem).read_text()

    # Create and run optimizer
    optimizer = NKIMoEOptimizer(ssh, e2e, agent_cfg, paths, compile_cfg)
    optimizer.run(max_rounds=args.rounds, problem_code=problem_code)


if __name__ == "__main__":
    main()
