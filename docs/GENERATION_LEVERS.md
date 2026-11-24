# Generation levers (vLLM) and speculative decoding

Regime unless stated: vLLM 0.33.0, Qwen3.5-9B (hybrid: 24 Gated-DeltaNet + 8 full-attention blocks), bf16/fp16
weights, one NVIDIA L40S per job, on a university Slurm cluster. Token gates use greedy decoding of 50 fixed prompts.
All numbers MEASURED, single run, single-agent checks, unless labelled otherwise. The raw result JSON files were not
re-opened for this write-up; the numbers come from the run notes written at harvest time.

## A. Lever table

| lever | result | status |
|---|---|---|
| `max_num_seqs` 512 + `gpu_memory_utilization` 0.95 | 0.985x throughput; mean running batch only 259 because the KV cache binds first | REJECTED |
| prefix caching off | 1.039x and 1.042x (two runs); token gate PASS | CANDIDATE (used in one adapter-serving path; optional elsewhere) |
| fp8 KV cache | KV capacity 1.90x, throughput 1.18x; token gate 25/50 diverged vs floor 18; teacher-forced max abs dlogprob 0.496 vs limit 0.2 (deciding run on an L40). Not supported on A100 (SM80) in this version | REJECTED (lossy at this version) |
| runtime multi-LoRA (one engine serves 2-8 adapters, no merge per adapter) | 1.023x vs a merged model; adapter-effect, token and faithfulness checks PASS | ADOPTED for new adapter families |
| MTP speculative decoding, k=3, always on | 1.60-2.23x at 1-48 active sequences; 0.81x at 256 active; token gate 14/50 = floor (k=1: 13 PASS; k=2: 17 FAIL) | default OFF; used only for low-concurrency work (<= 16 active) |
| dynamic MTP schedule (k=3 at 1-16 active, k=1 at 17-48, k=0 at 49-256) | main phase s_m = 0.604 at 256 active; tail s_t = 1.977 (min over 1/4/16 active); whole-job ratio 1.285; see section C | REJECTED |
| chunked prefill tuning, delta outputs | prompts short, driver CPU ~5 % | CLOSED (low value) |
| warm compile/JIT caches on node-local scratch | saves 1-2 min per job | ADOPTED as practice |
| length dispatch / second engine for the tail | covered mostly by the tail-exit rule (TAIL_EXIT.md) | NOT BUILT |

Production occupancy of an in-process vLLM reader (195-328 tok/s per L40S, tail-bound shards, 0.73-0.74 prefix-cache
hit) is documented in the sibling repository `judge-lab-public`, `docs/PERFORMANCE.md`, and not repeated here.

## B. Speculative decoding on long think-on generation

Workload: think-on generation with 65,536-token context, 32,768-token think budget, 2 attempts per problem,
streaming with aborts at the think budget and on loops, then a forced answer phase.

### B1. Synthetic screen (MEASURED; timed windows, batch 1 / 8 / 32, context 262,144)

MTP uses the checkpoint's own single MTP layer through vLLM's built-in `mtp` method (k > 1 re-runs the one layer).
A separate Qwen3.5-0.8B draft model was refused at engine init (hidden size 1024 vs 4096): dropped.

| arm | greedy gate (diverged / rounding floor) | length-matched sampling gate | per-sequence speed b1 / b8 / b32 |
|---|---|---|---|
| n-gram, 4 tokens (`prompt_lookup` 2-4) | batch 4: 3/4, floor 3: pass by floor | PASS | 2.36 / 1.82 / 1.28x |
| MTP k=3 | batch 4: 2/4, floor 3: pass; batch 8: 8/8, floor 6: FAIL | PASS | 2.08 / 2.07 / 1.64x |
| MTP k=2 | batch 8: 7/8, floor 6: FAIL | PASS | 1.80 / 1.87 / 1.53x |
| MTP k=1 | batch 8: 8/8, floor 6: FAIL | PASS | 1.41 / 1.57 / 1.32x |
| MTP k=5 | batch 4: 4/4, floor 3: FAIL | not run | 2.30 / 2.25 / 1.74x |

DERIVED, not a gate result: most greedy divergences of the speculative arms sit at the same token positions as the
no-spec rounding control's (`token_gate.shared_positions`), which is the pattern of near-tie argmax flips under a
different forward shape, not of state corruption.

Open risk (MEASURED, small n, not a measured difference): beyond ~16k generated tokens, where the base arm never got
within the timed windows, speculative arms produced loop-like text. n-gram drafting is fastest exactly on loops
(perfectly predictable text), so its batch-1 ratio is inflated by them.

### B2. Integration bench on the real workload (MEASURED; 1x L40S, 4 sequences, interleaved N1 MTP NGRAM N2 MTP NGRAM)

| arm | per-attempt tok/s (block 1 / block 2) | ratio | acceptance |
|---|---|---|---|
| no spec | 35.95 / 35.92 | 1.00 | - |
| MTP k=3 | 64.83 / 64.83 | 1.80x | 64 % (~2.9 tokens per step) |
| n-gram 4 | 48.99 / 49.00 | 1.36x | 16.5 % |

- Noise band between the two no-spec blocks: 0.1 %. Both arms pass the >= 15 % interleaved speed rule.
- Boundary check PASS in 6/6 blocks: the think-budget cut and the loop detector see exactly the same token index
  with and without speculation (speculation emits up to k+1 tokens per step, so this has to be checked).
- Caveats: batch 4 only; the speculative arms ended 4/4 at the think budget vs 1 natural + 3 forced for no-spec
  (different trajectories, n = 4).
- At production concurrency (24 active) the same MTP k=3 setting measured 1.848x (546.3 vs 312.7 aggregate tok/s,
  L40S), and 1.604x at 1 active (L40); see `judge-lab-public/docs/PERFORMANCE.md`. Not tested with runtime LoRA.

### B3. Reproducibility finding (MEASURED; saved think prefixes, cold vs warm prefix cache)

Greedy 2,048-token continuations from saved think prefixes, each request run twice (cold, then a warm prefix-cache hit).

| comparison | prompts diverged (of 9) |
|---|---|
| no-spec vs no-spec rounding control, cold | 6 (first divergences at tokens 94-369) |
| no-spec vs no-spec rounding control, warm | 5 |
| no-spec control, cold vs warm | **0** (token-identical) |
| MTP k=3, cold vs warm | **6** |
| n-gram 4, cold vs warm | **6** |

- Pre-registered greedy gate: FAIL for both speculative arms. Long-sample gate: PASS for both (loop rate 0/54 in
  every arm; min Mann-Whitney p 0.22 / 0.13 vs control 0.25).
- Reading (DERIVED): with speculation on, the output depends on prefix-cache and scheduling state, while the
  no-spec path is run-to-run reproducible from a cached prefix. This is a reproducibility property, not a quality
  one. It matters for paired experiments (two arms must see identical decoding), not for production generation,
  where the sampling gate is what counts.
- Decision: speculative decoding stayed bench-only for every running paired experiment (changing the decode path in
  the middle of an experiment breaks the pairing).

## C. Why a dynamic speculation schedule cannot rescue the main phase

Whole-job model (`inference_levers.speedup`): with f = share of generation GPU time spent at low concurrency,

    T(lever) / T(base) = (1 - f) / s_m + f / s_t

- f = 0.322: share of production generation GPU time with < 32 active sequences (MEASURED from 44 generation logs,
  317,511 GPU-seconds).
- Dynamic schedule: s_m 0.604, s_t 1.977 -> ratio 1.285 (slower). Break-even would need f >= 0.61. FAIL on both the
  s_m >= 0.97 rule and the ratio <= 0.95 rule. Token gate 15/50 passed against the pre-registered floor of 18; the
  repeat floor measured on that GPU was 11.
- Always-on k=3 in the same job (info only): s_m 0.803, s_t 1.985, ratio 1.006.
- Reading (DERIVED): with a speculative config loaded, the engine pays a cost at 256 active even when the schedule
  says k=0, and that cost is larger than always-on k=3's. A separate tail-only engine stays a design (NOT BUILT).

## Reproduce the gate logic

`tests/test_gates.py` reproduces the recorded whole-job numbers (1.285, break-even 0.61) and exercises the floor gate,
the arm filter and the ABAB rule on synthetic token ids. No engine is needed.
