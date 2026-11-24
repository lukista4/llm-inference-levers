# Method: how a lever was adopted or rejected

Every lever in this study went through the same steps. Lever means a change that should make generation or training
faster without changing what the model computes (or that changes it only by a bounded, stated amount).

## Labels used for numbers

| label | meaning |
|---|---|
| MEASURED | read from a run log or a result file of a real run |
| DERIVED | arithmetic on measured numbers, or a reading of source code |
| PROJECTED | an estimate or guess made before (or instead of) a measurement |

All runs were single runs unless stated. All checks were done by one agent; none was repeated by an independent
party. Each number carries its regime (hardware, engine version, concurrency, context).

## Statuses

| status | meaning |
|---|---|
| ADOPTED | passed its pre-registered gate and was switched on, with the stated scope |
| CANDIDATE | passed a gate, not switched on (scope or a further gate missing) |
| REJECTED | failed its pre-registered gate; a retry needs a new version and a new pre-registration |
| CLOSED | no gain, or not worth pursuing at this version |
| NOT BUILT | idea only |

## Gates

1. **Pre-registration.** The gate, its thresholds and the comparison were written and hashed before the run.
   Post-hoc variants are reported as post-hoc and never count as a pass.
2. **Token-equivalence gate against a rounding floor** (`inference_levers.token_gate`). Greedy decoding of a fixed
   prompt set (typically 50 prompts). The same engine already diverges from itself when the forward shape changes
   (bf16/fp16 near-tie argmax flips), so the bar is the baseline's divergence against a rounding-only control run
   (same engine, different batch). A lever passes if it diverges on no more prompts than that floor and covers at
   least half as many tokens.
3. **Teacher-forced log-prob check.** For levers that change numerics (KV-cache dtype), the max absolute change in
   per-token log-prob on teacher-forced text, threshold 0.2.
4. **Interleaved timing** (`inference_levers.speedup.abab_gate`). The node-to-node spread on the same kit was about
   14 % (MEASURED), so single-run speed claims below 15 % are not trusted. Arms run interleaved in one job on one GPU
   (base, lever, base, lever); the noise band is the spread between same-arm blocks; adopt only if the gain is
   >= 15 % and the slower lever block beats the faster base block.
5. **Training levers** are gated on rounding-only control floors: an exact re-run of the baseline and controls that
   change only rounding define the allowed spread of relative weight change, loss, dev NLL and first-step gradients.
6. **No mixing.** An adopted lever applies to new experiment families only, where every arm and the reference use it.
   Numbers produced with and without the lever are never compared with each other.

The sibling repository `judge-lab-public` (`docs/METHODOLOGY.md`, `docs/READER_EQUIVALENCE.md`) applies the same
discipline to an LLM-judge reader.
