# Training levers (LoRA on Qwen3.5-9B)

Regime: single-GPU LoRA training of Qwen3.5-9B (PyTorch + peft-style LoRA, fla chunked Gated-DeltaNet kernels),
one row per optimizer step. GPUs: NVIDIA L40S / L40 and A100 on a university Slurm cluster, A40 on a PBS cluster.
All numbers MEASURED, single run, single-agent checks, from the run notes.

Gate: a lever is compared with rounding-only controls run in the same job (an exact re-run of the baseline, and
controls that only change rounding, e.g. an fp32 convolution). Their spread sets the floor for relative weight
change, loss, dev NLL and first-step LoRA gradients. Speed is timed interleaved (see METHOD.md).

| lever | result | status |
|---|---|---|
| bf16 LoRA branch (keep the LoRA matmuls in bf16, drop the fp32 casts) | 1.29x interleaved (1,518 -> 1,962 tok/s, L40S); ~1.25x on the same node class; dev-NLL bias ~+0.0027 over 3 runs (~2 sd of re-runs) | ADOPTED only in new families where every arm uses it |
| bf16 branch + thresholded checkpointing | 1.33x; same bias | CANDIDATE, not shipped (the checkpointing part failed alone) |
| selective / thresholded gradient checkpointing | no wall gain; thresholded variant 1.04x and FAIL on dev NLL | REJECTED |
| bf16 x@A with fp32 B side | no real speed-up (node variation dominates) | REJECTED |
| exact-math fused LoRA ("F3": split each fp32 operand into 3 bf16 pieces so products are exact; chunked variant "F3C") | 1.14-1.18x interleaved; unit test: F3 LoRA gradients 5-10x the fp32 reference error (F3C with 256-chunks inside it); bench gate: F3 FAIL on dev NLL (abs 0.00359 > floor 0.00162), F3C FAIL on loss spread (0.0282 > 0.0251); gradients 18-25x further from the baseline than the tightest controls | REJECTED at this version |
| gradient bank | gradient parity FAIL | REJECTED |
| Triton causal_conv1d instead of the torch fallback | conv ~1 % of kernel time; no wall gain | CLOSED |
| sequence packing / larger micro-batch | changes the one-step-per-row recipe | CLOSED (recipe change, not a speed lever) |
| fused linear cross-entropy, torch.compile / CUDA graphs, FP8 frozen GEMMs, multi-arm shared-base process | 0-20 % PROJECTED, some lossy | NOT BUILT |
| resume guard + CUDA RNG restore on preemption | smoke PASS | ADOPTED (hardening, not speed) |

Notes
- First attempt on an L40 showed no lever gaining wall time, partly because arms drifted 6 % between an exact re-run
  pair (power 305 -> 265 W). That is why every later speed claim is interleaved inside one job.
- A unit test that a lever fails stays failed: in the fused-LoRA case the test was corrected for two genuine test
  design flaws (subnormal range, per-element tolerance), but the gradient criterion was left unchanged and the lever
  was rejected on it.
- Training was 20 % of an experiment family's GPU time (TAIL_EXIT.md), so even the adopted 1.29x moves the whole job
  by only ~5 % (DERIVED).
