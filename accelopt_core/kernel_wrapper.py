"""
NKI Kernel Profiler & Benchmarker
Ported from AccelOpt kernel_wrapper.py with SDK 2.28 namespace updates.

Changes from AccelOpt:
- neuronxcc.nki -> nki (SDK 2.28)
- Compile flags configurable (DGE toggle for Trn2 testing)
- Profile fields extended for Trn2 metrics
"""
import json
import uuid
import time
import tempfile
import os
import traceback
import numpy as np
from dataclasses import dataclass, field
from typing import Optional
from pathlib import Path


@dataclass
class KernelProperties:
    """Single kernel execution result (from AccelOpt eval_numpy.py)"""
    compiled: bool = False
    correct: bool = False
    runnable: bool = False
    metadata: dict = field(default_factory=dict)


def load_module_from_path(file_path: str):
    """Load a Python module from file path (from AccelOpt eval_numpy.py)"""
    import sys
    import importlib.util
    parent_dir = str(Path(file_path).parent)
    if parent_dir not in sys.path:
        sys.path.append(parent_dir)
    spec = importlib.util.spec_from_file_location("module", file_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load module from {file_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def l2norm_allclose(v_k, v_r, rel_tol=1e-5):
    """AccelOpt's L2-norm based allclose check"""
    return np.linalg.norm((v_k - v_r).astype(np.float64)) < rel_tol * np.linalg.norm(v_r.astype(np.float64))


def check_correctness(output_nki, output_task, res: KernelProperties, rel_tol: float = 2e-5):
    """Check kernel output correctness (from AccelOpt eval_numpy.py)"""
    import re

    if not isinstance(output_task, tuple):
        output_task = (output_task,)

    is_correct = True
    if len(output_nki) != len(output_task):
        res.metadata.setdefault("correctness_error", []).append(
            f"Num outputs mismatch: nki={len(output_nki)} vs ref={len(output_task)}"
        )
        res.correct = False
        return

    for i, (v_k, v_r) in enumerate(zip(output_nki, output_task)):
        if hasattr(v_r, "shape") and hasattr(v_k, "shape"):
            if v_k.shape != v_r.shape:
                res.metadata.setdefault("correctness_error", []).append(
                    f"Output {i} shape mismatch, expected {v_r.shape}, got {v_k.shape}"
                )
                is_correct = False
            if not l2norm_allclose(v_k, v_r, rel_tol=rel_tol):
                max_diff = np.amax(np.abs(v_k - v_r))
                avg_diff = np.mean(np.abs(v_k - v_r))
                l2_diff = np.linalg.norm((v_k - v_r).astype(np.float64))
                l2_ref = np.linalg.norm(v_r.astype(np.float64))
                res.metadata.setdefault("correctness_error", []).append(
                    f"Output {i} value mismatch: max_diff={max_diff:.6f}, "
                    f"avg_diff={avg_diff:.6f}, l2_rel_diff={l2_diff/l2_ref:.6f}"
                )
                is_correct = False
        else:
            if np.issubdtype(type(v_r), np.floating):
                if not l2norm_allclose(v_k, v_r, rel_tol=rel_tol):
                    res.metadata.setdefault("correctness_error", []).append(
                        f"Output {i} scalar mismatch: expected {v_r}, got {v_k}"
                    )
                    is_correct = False
            else:
                if v_k != v_r:
                    res.metadata.setdefault("correctness_error", []).append(
                        f"Output {i} value mismatch: expected {v_r}, got {v_k}"
                    )
                    is_correct = False
    res.correct = is_correct


def check_precision_and_correctness(program_path, output_nki, output_task, res, rel_tol):
    """Check for float16 usage and correctness (AccelOpt pattern)"""
    import re
    with open(program_path, 'r') as f:
        program_code = f.read()
    program_code = re.sub(r'#.*', '', program_code)
    if "float16" in program_code:
        res.metadata["correctness_error"] = "Float16 is used in the program."
        res.correct = False
        return
    check_correctness(output_nki, output_task, res, rel_tol=rel_tol)


def get_latency(nki_kernel_fn, nki_inputs, artifact_dir, compile_opt="--disable-dge --logical-nc-config=1"):
    """Measure kernel latency via neuron-profile summary-json (AccelOpt pattern)"""
    import nki as _nki

    kernel_id = uuid.uuid4()
    neff_path = os.path.join(artifact_dir, f"neff_{kernel_id}.neff")
    ntff_path = os.path.join(artifact_dir, f"ntff_{kernel_id}.ntff")

    _nki.baremetal(
        nki_kernel_fn,
        save_neff_name=neff_path,
        save_trace_name=ntff_path,
        additional_compile_opt=compile_opt
    )(*nki_inputs)

    summary_path = os.path.join(artifact_dir, f"profile_{kernel_id}.json")
    cmd = f"neuron-profile view --output-format summary-json -n {neff_path} -s {ntff_path} > {summary_path}"
    os.system(cmd)

    summary = json.load(open(summary_path, 'r'))
    latency_ms = summary[next(iter(summary))]["total_time"] * 1e3
    return latency_ms


def benchmark_latency(warmup_iters, bench_iters, nki_kernel_fn, nki_inputs, artifact_dir,
                      compile_opt="--disable-dge --logical-nc-config=1"):
    """Run warmup + benchmark iterations, return stats (AccelOpt pattern)"""
    import nki as _nki

    for _ in range(warmup_iters):
        _nki.baremetal(
            nki_kernel_fn,
            additional_compile_opt=compile_opt
        )(*nki_inputs)

    latencies = []
    for _ in range(bench_iters):
        lat = get_latency(nki_kernel_fn, nki_inputs, artifact_dir, compile_opt)
        latencies.append(lat)

    return {
        "mean_ms": np.mean(latencies),
        "min_ms": np.min(latencies),
        "max_ms": np.max(latencies),
        "rel_diffs": (np.max(latencies) - np.min(latencies)) / np.min(latencies)
    }


class NKIKernel:
    """
    NKI Kernel profiler/benchmarker.
    Ported from AccelOpt kernel_wrapper.py with SDK 2.28 adaptations.

    Usage:
        kernel = NKIKernel("path/to/kernel.py", "path/to/reference.py")
        result = kernel.profile(save_fields=["hbm_read_bytes", ...])
        print(result.metadata["latency"])
    """

    def __init__(self, program_path: str, base_numpy_path: str,
                 compile_opt: str = "--disable-dge --logical-nc-config=1",
                 rel_tol: float = 2e-5, perf_tol: float = 0.01,
                 warmup: int = 2, bench_iters: int = 10, max_retries: int = 2,
                 seeds: list = None):
        self.program_path = program_path
        self.base_numpy_path = base_numpy_path
        self.compile_opt = compile_opt
        self.rel_tol = rel_tol
        self.perf_tol = perf_tol
        self.warmup = warmup
        self.bench_iters = bench_iters
        self.max_retries = max_retries
        self.seeds = seeds or [0, 21, 42, 63, 84]

    def profile(self, save_fields: list = None) -> KernelProperties:
        """
        Full profile pipeline:
        1. Compile + run (NEFF/NTFF generation)
        2. Multi-seed correctness check
        3. Benchmark latency with variance retry
        4. Extract neuron-profile summary-json metrics
        """
        import nki as _nki

        save_fields = save_fields or []
        os.environ["NEURON_CC_FLAGS"] = "--auto-cast=none"
        os.environ["NEURON_RT_NUM_CORES"] = "1"
        np.random.seed(42)

        task_module = load_module_from_path(self.base_numpy_path)
        task_fn = task_module.forward
        task_np_input_fn = task_module.get_inputs
        task_np_inputs = task_np_input_fn()
        task_nki_output_fn = task_module.transform_nki_outputs

        res = KernelProperties()
        profile_name = f"nki_{uuid.uuid4()}"

        with tempfile.TemporaryDirectory(dir="/tmp", prefix=f"{profile_name}_") as artifact_dir:
            neff_path = os.path.join(artifact_dir, "kernel_file.neff")
            ntff_path = os.path.join(artifact_dir, "kernel_profile.ntff")

            # === Phase 1: Compile + initial run ===
            try:
                kernel_module = load_module_from_path(self.program_path)
                if hasattr(kernel_module, "kernel"):
                    kernel_fn = kernel_module.kernel
                elif hasattr(kernel_module, "optimized_kernel"):
                    kernel_fn = kernel_module.optimized_kernel
                else:
                    raise ValueError(f"No kernel function found in {self.program_path}")

                if hasattr(task_module, "transform_to_nki_inputs"):
                    nki_input_fn = task_module.transform_to_nki_inputs
                else:
                    raise ValueError(f"No transform_to_nki_inputs in {self.base_numpy_path}")

                nki_inputs = nki_input_fn(task_np_inputs)
                _nki.baremetal(
                    kernel_fn,
                    save_neff_name=neff_path,
                    save_trace_name=ntff_path,
                    additional_compile_opt=self.compile_opt
                )(*nki_inputs)
                res.compiled = True
                res.runnable = True

            except Exception as e:
                res.metadata["compilation_error"] = str(e)
                res.metadata["compilation_traceback"] = traceback.format_exc()
                return res

            # === Phase 2: Multi-seed correctness ===
            try:
                for seed in self.seeds:
                    np.random.seed(seed)
                    task_np_inputs = task_np_input_fn()
                    nki_inputs = nki_input_fn(task_np_inputs)
                    output_task = task_fn(*task_np_inputs)
                    output_nki_raw = _nki.baremetal(
                        kernel_fn,
                        additional_compile_opt=self.compile_opt
                    )(*nki_inputs)
                    output_nki = task_nki_output_fn(output_nki_raw, output_task)
                    check_precision_and_correctness(
                        self.program_path, output_nki, output_task, res, self.rel_tol
                    )
                    if not res.correct:
                        break
            except Exception as e:
                res.metadata["correctness_error"] = str(e)
                return res

            if not res.correct:
                return res

            # === Phase 3: Benchmark with variance retry ===
            try:
                stats = benchmark_latency(
                    self.warmup, self.bench_iters, kernel_fn, nki_inputs,
                    artifact_dir, self.compile_opt
                )
                all_stats = [stats]
                all_rel_diffs = [stats["rel_diffs"]]

                retry = 0
                while stats["rel_diffs"] > self.perf_tol and retry < self.max_retries:
                    time.sleep(1)
                    stats = benchmark_latency(
                        self.warmup, self.bench_iters, kernel_fn, nki_inputs,
                        artifact_dir, self.compile_opt
                    )
                    all_stats.append(stats)
                    all_rel_diffs.append(stats["rel_diffs"])
                    retry += 1

                # Pick most stable run
                best_idx = int(np.argmin(all_rel_diffs))
                best_stats = all_stats[best_idx]

                res.metadata["latency"] = best_stats["mean_ms"]
                res.metadata["min_ms"] = best_stats["min_ms"]
                res.metadata["max_ms"] = best_stats["max_ms"]
                res.metadata["rel_diffs"] = best_stats["rel_diffs"]

                # === Phase 4: Extract profile metrics ===
                summary_path = os.path.join(artifact_dir, f"{profile_name}_summary.json")
                cmd = (f"neuron-profile view --output-format summary-json "
                       f"-n {neff_path} -s {ntff_path} > {summary_path}")
                os.system(cmd)
                summary = json.load(open(summary_path, 'r'))
                profile_data = summary[next(iter(summary))]
                for field_name in save_fields:
                    if field_name in profile_data:
                        res.metadata[field_name] = profile_data[field_name]

            except Exception as e:
                res.metadata["benchmarking_error"] = traceback.format_exc()
                return res

            return res
