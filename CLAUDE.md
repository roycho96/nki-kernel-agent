# NKI MoE Kernel Optimization Agent

## 목표
qwen_with_nki.py를 수정하여 Qwen3-30B-A3B inference 성능을 최적화한다.

## 환경
- AWS Trn2, Neuron SDK 2.28, NKI Beta 2
- 단일 Trn2 칩, batch_size=1
- import는 `import nki` / `import nki.language as nl` (neuronxcc.nki 아님)

## 메트릭
- TTFT (낮을수록 좋음)
- tokens/sec (높을수록 좋음)
- NKI FLOPS / total FLOPS (높을수록 좋음)
- accuracy (reference와 일치해야 함 — 불일치 시 점수 0)

## 점수 공식
score = accuracy(binary) × (ref_TTFT/my_TTFT) × (my_tps/ref_tps) × (1 + nki_flops/total_flops)

## 제약
- 반드시 단일 파일 (qwen_with_nki.py)
- accuracy가 실패하면 모든 점수가 0 — correctness 우선
- NKI 커버리지 넓을수록 점수 보너스

## NKI 핵심 제약 (위반 시 컴파일 에러)
- SBUF/PSUM P dimension ≤ 128
- PSUM F dimension ≤ 512
- GEMM: stationary F ≤ 128, moving F ≤ 512
- top-level tensors는 HBM에 있어야 함 (buffer=nl.shared_hbm)
- auto/direct allocation 섞지 말 것
- import nki (NOT neuronxcc.nki)

## Trainium2 특유의 기회
- VectorE + GPSIMD가 SBUF 동시 접근 가능
- VectorE + ScalarE가 PSUM 동시 접근 가능
- DMA transpose: HBM→SBUF 이동 중 transpose 가능 (identity matrix 트릭 대체)
- GPSIMD integrated DMA: 307 GB/s
- gather_flattened, no_reorder, range_select API

## 우선순위
1. Expert MLP fused kernel (gate/up/down projection — SwiGLU)
2. Router 후처리 / dispatch / gather
3. RMSNorm / residual fusion
4. Decode 전용 최적화

## 벤치마크 실행
ssh trn2 "cd ~/nki-moe && python3 main.py --mode benchmark --enable-nki ..."

## 규칙
- 수정 전 반드시 현재 baseline 점수를 먼저 측정
- 수정 후 accuracy 먼저 확인, 통과해야만 성능 측정
- 성능이 떨어지면 revert
- 매 실험마다 experiments.jsonl에 기록
