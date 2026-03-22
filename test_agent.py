#!/usr/bin/env python3
"""
NKI MoE Agent Test Suite
Validates all components that can be tested WITHOUT a Trn2 instance.

Usage:
    python3 test_agent.py           # Run all tests
    python3 test_agent.py -v        # Verbose
    python3 test_agent.py -k test_  # Run specific test pattern
"""
import json
import sys
import os
import tempfile
import textwrap
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

# ================================================================
# Test 1: All imports work
# ================================================================
def test_imports():
    """모든 모듈이 import 에러 없이 로드되는가"""
    print("  [1] Testing imports...")

    from config import (RemoteConfig, CompileConfig, BenchmarkConfig,
                        AgentConfig, PathConfig, PROFILE_FIELDS)
    assert len(PROFILE_FIELDS) == 18, f"Expected 18 profile fields, got {len(PROFILE_FIELDS)}"
    assert "tensor_engine_active_time_percent" in PROFILE_FIELDS, "Trn2 metric missing"
    assert "dma_active_time" in PROFILE_FIELDS, "Trn2 metric missing"

    from infra.ssh_runner import SSHRunner, RemoteResult
    from infra.e2e_benchmark import E2EBenchmark, BenchmarkMetrics, CompetitionScore
    from ka_extensions.stability import (
        ReflexionManager, DivergenceGuard, AttemptHistory, Attempt, Reflexion
    )
    from prompts.planner_prompts.construct_base_prompt import inject_memory

    print("    ✅ All imports successful")


# ================================================================
# Test 2: Config defaults are sane
# ================================================================
def test_config():
    """설정 기본값이 합리적인가"""
    print("  [2] Testing config defaults...")

    from config import CompileConfig, BenchmarkConfig, AgentConfig

    cc = CompileConfig()
    assert "--logical-nc-config=1" in cc.additional_compile_opt
    assert cc.env_vars["NEURON_CC_FLAGS"] == "--auto-cast=none"
    assert cc.env_vars["NEURON_RT_NUM_CORES"] == "1"

    bc = BenchmarkConfig()
    assert len(bc.correctness_seeds) == 5
    assert bc.rel_tol == 2e-5

    ac = AgentConfig()
    assert ac.divergence_threshold == 50.0
    assert ac.no_improve_limit == 5

    # DGE toggle test
    cc_dge = CompileConfig(disable_dge=False)
    assert "--disable-dge" not in cc_dge.additional_compile_opt

    print("    ✅ Config defaults valid")


# ================================================================
# Test 3: KA Stability patterns work correctly
# ================================================================
def test_reflexion():
    """Reflexion 매니저가 올바르게 동작하는가"""
    print("  [3] Testing ReflexionManager...")

    from ka_extensions.stability import ReflexionManager

    rm = ReflexionManager(window=2)
    assert rm.format_for_prompt() == "", "Empty reflexion should return empty string"

    rm.add(1, "Improved 15%", category="improved", plan_summary="fuse rmsnorm")
    rm.add(2, "Caused 60% regression", category="regression", plan_summary="bad tiling")
    rm.add(3, "No improvement", category="no_improvement", plan_summary="loop reorder")

    # Window=2 should only show last 2
    recent = rm.get_recent()
    assert len(recent) == 2
    assert recent[0].round_num == 2
    assert recent[1].round_num == 3

    prompt = rm.format_for_prompt()
    assert "LESSONS FROM PREVIOUS" in prompt
    assert "AVOID" in prompt, "Should have AVOID section for regression"
    assert "bad tiling" in prompt

    print("    ✅ ReflexionManager works")


def test_divergence_guard():
    """Divergence Guard가 올바르게 revert 판단하는가"""
    print("  [4] Testing DivergenceGuard...")

    from ka_extensions.stability import DivergenceGuard

    dg = DivergenceGuard(threshold_percent=50.0)

    # No best yet — should never revert
    should_revert, div = dg.check(100.0)
    assert not should_revert, "Should not revert when no best is set"

    # Set best
    dg.update_best(10.0, "best_kernel_code", 1)
    assert dg.best_latency == 10.0

    # Slight regression — should NOT revert (20% < 50%)
    should_revert, div = dg.check(12.0)
    assert not should_revert, f"20% regression should not trigger revert (div={div})"

    # Big regression — SHOULD revert (100% > 50%)
    should_revert, div = dg.check(20.0)
    assert should_revert, f"100% regression should trigger revert (div={div})"
    assert div == 100.0

    # Improvement — should NOT revert
    should_revert, div = dg.check(8.0)
    assert not should_revert

    # Get best kernel
    assert dg.get_best_kernel() == "best_kernel_code"

    print("    ✅ DivergenceGuard works")


def test_attempt_history():
    """AttemptHistory가 올바르게 기록/로드하는가"""
    print("  [5] Testing AttemptHistory...")

    from ka_extensions.stability import AttemptHistory, Attempt

    with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
        log_path = f.name

    try:
        ah = AttemptHistory(log_path=log_path)

        ah.record(Attempt(round_num=1, status="improved", latency_ms=10.5, plan_summary="fuse ops"))
        ah.record(Attempt(round_num=2, status="correctness_fail", error="Shape mismatch (128,) vs (256,)"))
        ah.record(Attempt(round_num=3, status="no_improvement", latency_ms=10.8))

        assert len(ah.attempts) == 3

        # Error feedback
        feedback = ah.format_error_feedback()
        assert "Shape mismatch" in feedback
        assert "PREVIOUS ATTEMPT FAILED" in feedback

        # Summary
        summary = ah.format_summary()
        assert "3 attempts" in summary
        assert "1 improvements" in summary
        assert "1 failures" in summary

        # Reload from file
        ah2 = AttemptHistory(log_path=log_path)
        ah2.load_from_log()
        assert len(ah2.attempts) == 3
        assert ah2.attempts[0].status == "improved"

        print("    ✅ AttemptHistory works (write + reload)")

    finally:
        os.unlink(log_path)


# ================================================================
# Test 4: Prompt construction
# ================================================================
def test_prompt_files_exist():
    """모든 프롬프트 파일이 존재하고 비어있지 않은가"""
    print("  [6] Testing prompt files...")

    required_files = [
        "prompts/planner_prompts/base_prompt.txt",
        "prompts/planner_prompts/planner_prompt_template.txt",
        "prompts/executor_prompts/base_prompt.txt",
        "prompts/executor_prompts/user_prompt_template.txt",
        "prompts/summarizer_prompts/base_prompt.txt",
        "prompts/summarizer_prompts/user_prompt_template.txt",
        "prompts/profile_list.json",
        "prompts/displayed_profiles.json",
    ]

    for f in required_files:
        path = Path(f)
        assert path.exists(), f"Missing: {f}"
        content = path.read_text()
        assert len(content) > 10, f"Empty or too short: {f}"

    print("    ✅ All prompt files exist and non-empty")


def test_executor_prompt_has_trn2_constraints():
    """Executor prompt에 Trn2 필수 제약이 포함되어 있는가"""
    print("  [7] Testing executor prompt Trn2 constraints...")

    content = Path("prompts/executor_prompts/base_prompt.txt").read_text()

    # 🔴 필수 제약
    assert "shared_hbm" in content or "HBM" in content, "Missing: top-level HBM constraint"
    assert "auto" in content and "direct" in content, "Missing: auto/direct allocation warning"
    assert "import nki" in content, "Missing: SDK 2.28 namespace"
    assert "neuronxcc" not in content.split("import nki")[0][-50:], "Old namespace should not appear as instruction"

    # 🟡 Trn2 신기능
    assert "VectorE" in content and "GPSIMD" in content and "SBUF" in content, \
        "Missing: Trn2 engine parallelism"
    assert "DMA transpose" in content, "Missing: DMA transpose"
    assert "307" in content, "Missing: GPSIMD DMA bandwidth"
    assert "gather_flattened" in content, "Missing: gather_flattened API"
    assert "no_reorder" in content, "Missing: no_reorder API"

    print("    ✅ Executor prompt has all Trn2 constraints")


def test_planner_prompt_has_underutilized():
    """Planner prompt에 underutilized 분류가 있는가"""
    print("  [8] Testing planner prompt bottleneck classification...")

    content = Path("prompts/planner_prompts/base_prompt.txt").read_text()

    assert "COMPUTE-BOUND" in content or "compute-bound" in content.lower()
    assert "MEMORY-BOUND" in content or "memory-bound" in content.lower()
    assert "UNDERUTILIZED" in content, "Missing: underutilized category"
    assert "tensor_engine_active_time_percent" in content, "Missing: Trn2 metric"
    assert "dma_active_time" in content, "Missing: Trn2 metric"

    print("    ✅ Planner prompt has 3-way bottleneck classification")


def test_optimization_memory_injection():
    """Optimization memory가 planner prompt에 올바르게 주입되는가"""
    print("  [9] Testing optimization memory injection...")

    from prompts.planner_prompts.construct_base_prompt import inject_memory

    base = "You are an expert."
    empty_result = inject_memory(base, [])
    assert empty_result == base, "Empty memory should not change prompt"

    memory = [
        {"title": "**Tile size increase**", "summary": "Increasing TILE_N from 256 to 512 improved..."},
        {"title": "**No optimization found**", "summary": "No optimization found"},
        {"title": "**Loop reorder**", "summary": "Reordering K loop inside N loop..."},
    ]
    result = inject_memory(base, memory)
    assert "Optimization Experiences" in result
    assert "Increasing TILE_N" in result  # from summary, not title
    assert "Reordering K loop" in result  # from summary, not title
    # "No optimization found" should be filtered
    assert result.count("No optimization found") == 0

    print("    ✅ Optimization memory injection works")


def test_prompt_template_formatting():
    """프롬프트 템플릿이 올바르게 포맷되는가"""
    print("  [10] Testing prompt template formatting...")

    planner_tpl = Path("prompts/planner_prompts/planner_prompt_template.txt").read_text()
    result = planner_tpl.format(
        problem_code="def forward(x): return x * 2",
        kernel_code="@nki.jit\ndef kernel(x): ...",
        profile='{"latency": 1.23}'
    )
    assert "forward(x)" in result
    assert "@nki.jit" in result
    assert "1.23" in result

    executor_tpl = Path("prompts/executor_prompts/user_prompt_template.txt").read_text()
    result = executor_tpl.format(
        problem_code="def forward(x): return x * 2",
        kernel_code="@nki.jit\ndef kernel(x): ...",
        optimization_plan="Increase tile size",
        error_feedback="",
        reflexion_context=""
    )
    assert "Increase tile size" in result

    # With error feedback
    result2 = executor_tpl.format(
        problem_code="...",
        kernel_code="...",
        optimization_plan="...",
        error_feedback="# PREVIOUS ATTEMPT FAILED\nError: Shape mismatch",
        reflexion_context="# LESSONS\n- R1: improved 15%"
    )
    assert "PREVIOUS ATTEMPT FAILED" in result2
    assert "LESSONS" in result2

    print("    ✅ Prompt templates format correctly")


# ================================================================
# Test 5: Score calculation
# ================================================================
def test_score_calculation():
    """대회 점수 공식이 올바른가"""
    print("  [11] Testing score calculation...")

    from infra.e2e_benchmark import E2EBenchmark, BenchmarkMetrics, CompetitionScore
    from infra.ssh_runner import SSHRunner

    # Create dummy objects (won't actually connect)
    ssh = SSHRunner.__new__(SSHRunner)
    e2e = E2EBenchmark.__new__(E2EBenchmark)
    e2e.ref_ttft = 100.0
    e2e.ref_tps = 50.0

    # Case 1: Perfect improvement
    metrics = BenchmarkMetrics(ttft_ms=50.0, tokens_per_sec=100.0,
                               nki_flops=500, total_flops=1000, parse_success=True)
    score = e2e.compute_score(metrics, accuracy=True)
    # score = 1.0 * (100/50) * (100/50) * (1 + 500/1000)
    # = 1.0 * 2.0 * 2.0 * 1.5 = 6.0
    assert abs(score.total_score - 6.0) < 0.01, f"Expected 6.0, got {score.total_score}"

    # Case 2: Accuracy failure = 0
    score_fail = e2e.compute_score(metrics, accuracy=False)
    assert score_fail.total_score == 0.0

    # Case 3: No NKI FLOPS
    metrics_no_nki = BenchmarkMetrics(ttft_ms=80.0, tokens_per_sec=60.0,
                                      nki_flops=0, total_flops=1000, parse_success=True)
    score_no_nki = e2e.compute_score(metrics_no_nki, accuracy=True)
    # nki_bonus = 1 + 0/1000 = 1.0
    expected = 1.0 * (100/80) * (60/50) * 1.0
    assert abs(score_no_nki.total_score - expected) < 0.01

    print("    ✅ Score calculation correct")


# ================================================================
# Test 6: Code extraction
# ================================================================
def test_code_extraction():
    """LLM 응답에서 코드 블록 추출이 올바른가"""
    print("  [12] Testing code extraction...")

    # Import from orchestrator
    sys.path.insert(0, ".")
    from orchestrator import extract_code

    # Normal case
    response = """
Here's the optimized kernel:

```python
@nki.jit
def kernel(x, y):
    out = nl.ndarray(x.shape, dtype=x.dtype, buffer=nl.shared_hbm)
    tile = nl.load(x)
    nl.store(out, tile)
    return out
```

This should improve performance by 20%.
"""
    code = extract_code(response)
    assert code is not None
    assert "@nki.jit" in code
    assert "nl.shared_hbm" in code

    # No code block
    assert extract_code("Just some text without code") is None

    # Code block without python marker
    response2 = "```\ndef kernel(): pass\n```"
    code2 = extract_code(response2)
    assert code2 is not None
    assert "def kernel" in code2

    print("    ✅ Code extraction works")


# ================================================================
# Test 7: SSH Runner (structure only, no actual connection)
# ================================================================
def test_ssh_runner_structure():
    """SSHRunner가 올바른 명령 구조를 만드는가"""
    print("  [13] Testing SSHRunner command construction...")

    from infra.ssh_runner import SSHRunner

    ssh = SSHRunner(host="ubuntu@10.0.1.100", key_path="/tmp/fake_key", remote_dir="~/nki-moe")

    # Check internal command construction
    assert ssh.host == "ubuntu@10.0.1.100"
    assert "StrictHostKeyChecking=no" in ssh._ssh_base
    assert ssh.remote_dir == "~/nki-moe"

    print("    ✅ SSHRunner structure valid")


# ================================================================
# Test 8: Profile list matches displayed profiles
# ================================================================
def test_profile_lists_consistent():
    """profile_list.json과 displayed_profiles.json이 일관성 있는가"""
    print("  [14] Testing profile list consistency...")

    profile_list = json.loads(Path("prompts/profile_list.json").read_text())
    displayed = json.loads(Path("prompts/displayed_profiles.json").read_text())

    # displayed should be a superset of profile_list (it adds "latency")
    for field in profile_list:
        assert field in displayed, f"{field} in profile_list but not in displayed_profiles"

    assert "latency" in displayed, "displayed_profiles should include latency"

    print("    ✅ Profile lists consistent")


# ================================================================
# Test 9: NKIBench seeds have correct namespace
# ================================================================
def test_nkibench_namespace():
    """NKIBench seed 커널이 SDK 2.28 namespace를 사용하는가"""
    print("  [15] Testing NKIBench seed namespaces...")

    seeds_dir = Path("nkibench_seeds")
    for f in seeds_dir.glob("*.py"):
        if f.name.startswith("ref_"):
            continue  # reference files don't use nki
        content = f.read_text()
        assert "neuronxcc" not in content, f"{f.name} uses old namespace!"
        assert "import nki" in content, f"{f.name} missing 'import nki'"

    print("    ✅ All NKIBench seeds use SDK 2.28 namespace")


# ================================================================
# Main
# ================================================================
def main():
    os.chdir(Path(__file__).parent)

    print("=" * 60)
    print("NKI MoE Agent Test Suite")
    print("=" * 60)
    print()

    tests = [
        test_imports,
        test_config,
        test_reflexion,
        test_divergence_guard,
        test_attempt_history,
        test_prompt_files_exist,
        test_executor_prompt_has_trn2_constraints,
        test_planner_prompt_has_underutilized,
        test_optimization_memory_injection,
        test_prompt_template_formatting,
        test_score_calculation,
        test_code_extraction,
        test_ssh_runner_structure,
        test_profile_lists_consistent,
        test_nkibench_namespace,
    ]

    passed = 0
    failed = 0
    errors = []

    for test in tests:
        try:
            test()
            passed += 1
        except AssertionError as e:
            failed += 1
            errors.append((test.__name__, str(e)))
            print(f"    ❌ FAILED: {e}")
        except Exception as e:
            failed += 1
            errors.append((test.__name__, f"{type(e).__name__}: {e}"))
            print(f"    💥 ERROR: {type(e).__name__}: {e}")

    print()
    print("=" * 60)
    print(f"Results: {passed} passed, {failed} failed, {passed + failed} total")
    print("=" * 60)

    if errors:
        print("\nFailures:")
        for name, err in errors:
            print(f"  - {name}: {err}")
        return 1

    print("\n✅ All tests passed! Agent is ready for deployment.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
