"""NKI MoE Kernel Agent Configuration"""
from dataclasses import dataclass, field
from pathlib import Path
import os


@dataclass
class RemoteConfig:
    """Trn2 instance SSH configuration"""
    host: str = os.environ.get("TRN2_HOST", "ubuntu@<trn2-ip>")
    key_path: str = os.environ.get("TRN2_KEY", "~/.ssh/id_rsa")
    remote_dir: str = "~/nki-moe"
    model_path: str = "~/qwen-30b-a3b/hf_model"
    compiled_path: str = "~/qwen-30b-a3b/traced_model"
    ssh_timeout: int = 1800  # 30 min for compilation
    scp_timeout: int = 60


@dataclass
class CompileConfig:
    """Neuron compiler flags"""
    # AccelOpt default: "--disable-dge --logical-nc-config=1"
    # Trn2: DGE might be beneficial, test both
    disable_dge: bool = True  # Set False to test DGE on Trn2
    logical_nc_config: int = 1  # 1 = single physical core per logical
    auto_cast: str = "none"  # No automatic precision conversion
    num_cores: int = 1

    @property
    def additional_compile_opt(self) -> str:
        opts = []
        if self.disable_dge:
            opts.append("--disable-dge")
        opts.append(f"--logical-nc-config={self.logical_nc_config}")
        return " ".join(opts)

    @property
    def env_vars(self) -> dict:
        return {
            "NEURON_CC_FLAGS": f"--auto-cast={self.auto_cast}",
            "NEURON_RT_NUM_CORES": str(self.num_cores),
        }


@dataclass
class BenchmarkConfig:
    """Benchmark parameters (from AccelOpt kernel_wrapper.py)"""
    warmup_iterations: int = 2
    benchmark_iterations: int = 10
    correctness_seeds: list = field(default_factory=lambda: [0, 21, 42, 63, 84])
    rel_tol: float = 2e-5  # L2-norm relative tolerance
    perf_tol: float = 0.01  # 1% variance threshold
    max_perf_retries: int = 2


@dataclass
class AgentConfig:
    """Agent loop parameters"""
    max_rounds: int = 100
    divergence_threshold: float = 50.0  # % regression to trigger revert
    no_improve_limit: int = 5  # consecutive no-improvement rounds before stop
    max_error_retries: int = 3  # max retries for same error pattern
    reflexion_window: int = 3  # number of recent reflexions to include in prompt
    history_window: int = 10  # number of recent experiments to include


@dataclass
class PathConfig:
    """File paths"""
    project_root: Path = Path(__file__).parent
    kernel_file: str = "qwen_with_nki.py"
    experiments_log: str = "experiments.jsonl"
    checkpoint_dir: str = "checkpoints"
    memory_file: str = "optimization_memory/rewrites.json"

    # Prompt files
    planner_base: str = "prompts/planner_prompts/base_prompt.txt"
    planner_template: str = "prompts/planner_prompts/planner_prompt_template.txt"
    executor_base: str = "prompts/executor_prompts/base_prompt.txt"
    executor_template: str = "prompts/executor_prompts/user_prompt_template.txt"
    summarizer_base: str = "prompts/summarizer_prompts/base_prompt.txt"
    summarizer_template: str = "prompts/summarizer_prompts/user_prompt_template.txt"
    profile_list: str = "prompts/profile_list.json"

    def resolve(self, relative: str) -> Path:
        return self.project_root / relative


# Profile fields to request from neuron-profile summary-json
# AccelOpt 16 original + 2 Trn2 additions
PROFILE_FIELDS = [
    "hbm_read_bytes",
    "hbm_write_bytes",
    "psum_read_bytes",
    "psum_write_bytes",
    "sbuf_read_bytes",
    "sbuf_write_bytes",
    "spill_reload_bytes",
    "spill_save_bytes",
    "hardware_flops",
    "transpose_flops",
    "peak_flops_bandwidth_ratio",
    "mm_arithmetic_intensity",
    "hfu_estimated_percent",
    "scalar_engine_active_time_percent",
    "vector_engine_active_time_percent",
    "gpsimd_engine_active_time_percent",
    # Trn2 additions (verify actual key names on Trn2 instance)
    "tensor_engine_active_time_percent",
    "dma_active_time",
]
