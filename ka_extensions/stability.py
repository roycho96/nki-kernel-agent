"""
KernelAgent stability patterns adapted for NKI optimization.

Three patterns from KernelAgent that AccelOpt doesn't have:
1. Reflexion — structured self-reflection after each round
2. Divergence Guard — auto-revert when performance regresses too much
3. Attempt History — structured tracking with error feedback
"""
import json
import logging
from dataclasses import dataclass, field
from typing import Optional
from pathlib import Path

logger = logging.getLogger(__name__)


# ============================================================
# 1. Reflexion
# ============================================================

@dataclass
class Reflexion:
    """A single self-reflection entry"""
    round_num: int
    lesson: str
    category: str = "neutral"  # "improved", "regression", "error", "no_improvement"
    plan_summary: str = ""

    def format_for_prompt(self) -> str:
        icon = {
            "improved": "✅",
            "regression": "🔴",
            "error": "❌",
            "no_improvement": "➡️"
        }.get(self.category, "•")
        return f"{icon} Round {self.round_num}: {self.lesson}"


class ReflexionManager:
    """Manages reflexion history and injects into prompts"""

    def __init__(self, window: int = 3):
        self.reflexions: list[Reflexion] = []
        self.window = window

    def add(self, round_num: int, lesson: str, category: str = "neutral",
            plan_summary: str = ""):
        r = Reflexion(round_num, lesson, category, plan_summary)
        self.reflexions.append(r)
        logger.info(f"Reflexion [{category}] R{round_num}: {lesson[:80]}")

    def get_recent(self) -> list[Reflexion]:
        return self.reflexions[-self.window:]

    def format_for_prompt(self) -> str:
        recent = self.get_recent()
        if not recent:
            return ""
        lines = ["# LESSONS FROM PREVIOUS ROUNDS"]
        for r in recent:
            lines.append(r.format_for_prompt())

        # Extract avoid/try patterns
        avoid = [r.plan_summary for r in recent
                 if r.category in ("regression", "error") and r.plan_summary]
        do_try = [r.plan_summary for r in recent
                  if r.category == "improved" and r.plan_summary]

        if avoid:
            lines.append(f"\nAVOID these approaches: {'; '.join(avoid[:3])}")
        if do_try:
            lines.append(f"\nBUILD ON these successes: {'; '.join(do_try[:3])}")
        return "\n".join(lines)

    def to_dict(self) -> list:
        return [
            {"round": r.round_num, "lesson": r.lesson,
             "category": r.category, "plan": r.plan_summary}
            for r in self.reflexions
        ]


# ============================================================
# 2. Divergence Guard
# ============================================================

class DivergenceGuard:
    """
    Auto-revert when kernel performance degrades beyond threshold.
    KernelAgent pattern: if new kernel is >X% slower than best, revert to best.
    """

    def __init__(self, threshold_percent: float = 50.0):
        self.threshold = threshold_percent
        self.best_latency: float = float('inf')
        self.best_kernel: Optional[str] = None
        self.best_round: int = -1

    def update_best(self, latency: float, kernel_code: str, round_num: int):
        if latency < self.best_latency:
            self.best_latency = latency
            self.best_kernel = kernel_code
            self.best_round = round_num
            logger.info(f"New best: {latency:.3f}ms at round {round_num}")

    def check(self, new_latency: float) -> tuple[bool, float]:
        """
        Check if new latency diverges too much from best.
        Returns: (should_revert, divergence_percent)
        """
        if self.best_latency <= 0 or self.best_latency == float('inf'):
            return False, 0.0

        divergence = (new_latency - self.best_latency) / self.best_latency * 100
        should_revert = divergence > self.threshold

        if should_revert:
            logger.warning(
                f"Divergence: {divergence:.1f}% > {self.threshold}% threshold. "
                f"New={new_latency:.3f}ms vs Best={self.best_latency:.3f}ms. REVERTING."
            )

        return should_revert, divergence

    def get_best_kernel(self) -> Optional[str]:
        return self.best_kernel


# ============================================================
# 3. Attempt History
# ============================================================

@dataclass
class Attempt:
    """A single optimization attempt record"""
    round_num: int
    status: str  # "improved", "no_improvement", "correctness_fail", "compile_fail", "divergence_revert"
    latency_ms: float = 0.0
    score: float = 0.0
    plan_summary: str = ""
    error: str = ""
    profile_snapshot: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "round": self.round_num,
            "status": self.status,
            "latency_ms": self.latency_ms,
            "score": self.score,
            "plan": self.plan_summary,
            "error": self.error[:500] if self.error else "",
            "profile": self.profile_snapshot,
        }


class AttemptHistory:
    """
    Tracks all optimization attempts and provides error feedback.
    KernelAgent pattern: feed compilation/correctness errors back into next prompt.
    """

    def __init__(self, log_path: str = "experiments.jsonl", window: int = 10):
        self.attempts: list[Attempt] = []
        self.log_path = log_path
        self.window = window
        self._error_patterns: dict[str, int] = {}  # error_hash -> count

    def record(self, attempt: Attempt):
        self.attempts.append(attempt)
        self._append_to_log(attempt)

        # Track error patterns
        if attempt.error:
            key = attempt.error[:100]
            self._error_patterns[key] = self._error_patterns.get(key, 0) + 1

        logger.info(
            f"Attempt R{attempt.round_num}: {attempt.status} "
            f"(latency={attempt.latency_ms:.3f}ms)"
        )

    def get_recent(self) -> list[Attempt]:
        return self.attempts[-self.window:]

    def get_last_error(self) -> Optional[str]:
        """Get most recent error for feedback injection"""
        for a in reversed(self.attempts):
            if a.error:
                return a.error
        return None

    def get_error_count(self, error_prefix: str) -> int:
        """Check how many times a similar error has occurred"""
        return self._error_patterns.get(error_prefix[:100], 0)

    def format_error_feedback(self) -> str:
        """Format recent errors for injection into executor prompt"""
        last_err = self.get_last_error()
        if not last_err:
            return ""

        lines = [
            "# PREVIOUS ATTEMPT FAILED",
            f"Error: {last_err[:500]}",
            "",
            "Fix this error while applying the optimization.",
            "Do NOT repeat the same mistake."
        ]
        return "\n".join(lines)

    def format_summary(self) -> str:
        """Format recent history for context"""
        recent = self.get_recent()
        if not recent:
            return ""

        lines = ["# RECENT EXPERIMENT HISTORY"]
        for a in recent:
            icon = {"improved": "✅", "no_improvement": "➡️",
                    "correctness_fail": "❌", "compile_fail": "💥",
                    "divergence_revert": "🔴"}.get(a.status, "•")
            line = f"{icon} R{a.round_num}: {a.status}"
            if a.latency_ms > 0:
                line += f" ({a.latency_ms:.3f}ms)"
            if a.plan_summary:
                line += f" — {a.plan_summary[:60]}"
            lines.append(line)

        # Stats
        total = len(self.attempts)
        improved = sum(1 for a in self.attempts if a.status == "improved")
        failed = sum(1 for a in self.attempts if a.status in ("correctness_fail", "compile_fail"))
        lines.append(f"\nTotal: {total} attempts, {improved} improvements, {failed} failures")

        return "\n".join(lines)

    def _append_to_log(self, attempt: Attempt):
        """Append to experiments.jsonl"""
        try:
            with open(self.log_path, "a") as f:
                f.write(json.dumps(attempt.to_dict()) + "\n")
        except Exception as e:
            logger.error(f"Failed to write log: {e}")

    def load_from_log(self):
        """Load history from existing experiments.jsonl"""
        path = Path(self.log_path)
        if not path.exists():
            return
        try:
            for line in path.read_text().strip().split("\n"):
                if not line:
                    continue
                d = json.loads(line)
                self.attempts.append(Attempt(
                    round_num=d.get("round", 0),
                    status=d.get("status", "unknown"),
                    latency_ms=d.get("latency_ms", 0),
                    score=d.get("score", 0),
                    plan_summary=d.get("plan", ""),
                    error=d.get("error", ""),
                    profile_snapshot=d.get("profile", {}),
                ))
            logger.info(f"Loaded {len(self.attempts)} attempts from {self.log_path}")
        except Exception as e:
            logger.error(f"Failed to load history: {e}")
