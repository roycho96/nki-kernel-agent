# NKI Kernel Optimization Agent

An autonomous, profile-guided optimization agent for [NKI (Neuron Kernel Interface)](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/general/nki/index.html) kernels on AWS Trainium2. It iteratively improves NKI kernel performance by combining LLM-based code generation with hardware profiling feedback, structured self-reflection, and automatic regression detection.

The agent is a hybrid of two systems:

- **[AccelOpt](https://arxiv.org/abs/2502.15253)** — an NKI-native kernel optimization framework with a Planner → Executor → Summarizer workflow and optimization memory (slow→fast experience accumulation). Provides the NKI-specific profiling infrastructure, prompt engineering, and domain knowledge.
- **[KernelAgent](https://github.com/meta-pytorch/KernelAgent)** — Meta's multi-agent GPU kernel synthesis system. Provides the stability patterns: reflexion (structured self-reflection), divergence-based revert, error feedback loops, and experiment history tracking.

AccelOpt handles the *what* (NKI-aware optimization planning and code generation), while KernelAgent patterns handle the *how* (keeping the agent loop stable over many rounds of autonomous execution).

## How It Works

```
Local Machine                          Trn2 Instance
    │                                      │
    │  orchestrator.py                     │
    │  ├── Plan (LLM: planner prompt)      │
    │  ├── Execute (LLM: executor prompt)  │
    │  ├── Upload ─── scp ──────────────>  │ target kernel file
    │  │                                   │ ├── compile (neuronxcc)
    │  │                                   │ ├── correctness check
    │  │  <─── ssh result ────────────────── ├── benchmark
    │  ├── Decide (divergence guard)       │
    │  ├── Summarize (optimization memory) │
    │  └── Reflexion (self-reflection)     │
    │                                      │
    └── repeat ────────────────────────────┘
```

Each round:

1. **Profile** the current kernel on Trn2 via `neuron-profile summary-json`
2. **Plan** an optimization using profile metrics + accumulated experience (AccelOpt planner)
3. **Execute** the plan as code, injecting error feedback and reflexion context (AccelOpt executor + KernelAgent patterns)
4. **Verify** correctness with multi-seed L2-norm checks — incorrect kernels are immediately discarded
5. **Benchmark** end-to-end latency and throughput
6. **Decide** whether to keep or revert — if the new kernel regresses beyond a threshold, the agent automatically reverts to the best known version (KernelAgent divergence guard)
7. **Summarize** improvements into optimization memory for future rounds (AccelOpt summarizer)
8. **Reflect** on what worked or failed, feeding lessons into subsequent prompts (KernelAgent reflexion)

## Quick Start

```bash
# 1. Set up the Trn2 instance (run once)
ssh ubuntu@<trn2-ip> 'bash -s' < setup_trn2.sh

# 2. Deploy your target code on Trn2
ssh ubuntu@<trn2-ip> "cd ~/nki-moe && git clone <YOUR_REPO> ."

# 3. Run the agent locally
chmod +x run.sh
./run.sh ubuntu@<trn2-ip>

# Or with more control:
python3 orchestrator.py \
    --host ubuntu@<trn2-ip> \
    --rounds 50 \
    --kernel qwen_with_nki.py \
    --problem reference_implementation.py
```

## Project Structure

```
nki-kernel-agent/
├── orchestrator.py                    # Main optimization loop
├── config.py                          # All configuration (remote, compile, agent params)
├── CLAUDE.md                          # Instructions for Claude Code integration
│
├── accelopt_core/
│   └── kernel_wrapper.py              # NKI profiling, benchmarking, correctness checking
│                                      # (ported from AccelOpt, SDK 2.28 namespace)
│
├── ka_extensions/
│   └── stability.py                   # ReflexionManager, DivergenceGuard, AttemptHistory
│                                      # (patterns adapted from KernelAgent)
│
├── infra/
│   ├── ssh_runner.py                  # SSH remote execution, file transfer, cache management
│   └── e2e_benchmark.py              # End-to-end benchmark parsing and score calculation
│
├── prompts/
│   ├── planner_prompts/
│   │   ├── base_prompt.txt            # NKI API reference + Trn2 features + bottleneck taxonomy
│   │   ├── planner_prompt_template.txt
│   │   └── construct_base_prompt.py   # Injects optimization memory into planner prompt
│   ├── executor_prompts/
│   │   ├── base_prompt.txt            # NKI constraints + Trn2-specific opportunities
│   │   └── user_prompt_template.txt   # Slots for error feedback + reflexion context
│   ├── summarizer_prompts/
│   │   ├── base_prompt.txt
│   │   └── user_prompt_template.txt
│   ├── profile_list.json              # 18 neuron-profile metrics (16 original + 2 Trn2)
│   └── displayed_profiles.json
│
├── nkibench_seeds/                    # Reference NKI kernels (SDK 2.28 namespace)
│   ├── add_rmsnorm_matmul_*.py        # Fused residual + norm + matmul
│   ├── matmul_add_rmsnorm_*.py        # Fused matmul + residual + norm
│   ├── rope_single_freq_apply.py      # Rotary position embedding
│   └── ref_add_rmsnorm_matmul.py      # NumPy reference for correctness
│
├── optimization_memory/
│   └── rewrites.json                  # Accumulated slow→fast transformation experiences
│
├── checkpoints/                       # Saved kernel versions at each improvement
├── experiments.jsonl                  # Structured experiment log (auto-generated)
│
├── test_agent.py                      # 15-test validation suite (runs without Trn2)
├── setup_trn2.sh                      # One-time Trn2 instance setup
└── run.sh                             # Convenience launcher
```

## Trn2 Adaptations

The original AccelOpt was built for Trainium1 (NKI Beta 1, SDK ≤2.27). This agent applies the following adaptations for Trainium2:

**Must-fix (compile errors without these):**
- `neuronxcc.nki.*` → `nki.*` namespace (SDK 2.28)
- Top-level kernel I/O must be HBM (`buffer=nl.shared_hbm`)
- Auto/direct SBUF/PSUM allocation cannot be mixed in the same kernel

**Performance-relevant (Trn2 features not in original AccelOpt):**
- Engine parallelism: VectorE+GPSIMD can access SBUF simultaneously; VectorE+ScalarE can access PSUM simultaneously
- DMA transpose during HBM→SBUF transfer (replaces identity matrix trick)
- GPSIMD integrated DMA at 307 GB/s (useful for irregular data movement)
- New APIs: `gather_flattened`, `no_reorder`, `range_select`
- "Underutilized" bottleneck category added (neither compute nor memory bound)
- `tensor_engine_active_time_percent` and `dma_active_time` metrics added to profile list

## Testing

```bash
python3 test_agent.py
```

Validates all components that work without a Trn2 instance: imports, config sanity, reflexion/divergence/history logic, prompt construction, score calculation, code extraction, namespace compliance of seed kernels.

## References

- **AccelOpt**: Jia et al., "AccelOpt: An NKI Kernel Optimization Agent on Trainium" (2025). [arXiv:2502.15253](https://arxiv.org/abs/2502.15253) — NKI-native Planner/Executor/Summarizer workflow, optimization memory, NKIBench profiling metrics.
- **KernelAgent**: Meta PyTorch, "KernelAgent — Multi-Agent GPU Kernel Synthesis and Optimization." [GitHub](https://github.com/meta-pytorch/KernelAgent) / [Blog](https://pytorch.org/blog/kernelfalcon-autonomous-gpu-kernel-generation-via-deep-agents/) — Parallel worker verification, reflexion, divergence-based revert, roofline-guided bottleneck analysis.
- **NKI Documentation**: [AWS Neuron NKI Guide](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/general/nki/index.html)
