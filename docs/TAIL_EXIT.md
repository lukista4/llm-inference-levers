# Where the GPU time goes: generation tail, tail-exit rule, tail replay

Regime: one experiment family = 10 jobs of LoRA training + batch generation + grading on NVIDIA L40S, Qwen3.5-9B,
vLLM 0.33.0, 920 generation attempts per config, 90-minute (5,400 s) generation deadline, a university Slurm
cluster. Sources: scheduler accounting and the run logs (both read-only). All MEASURED, single-agent, unless labelled.

## 1. Stage breakdown (one family, 10 jobs, 62,124 GPU-seconds = 17.3 GPU-h)

| stage | GPU-seconds | share |
|---|---|---|
| generation | 46,328 | 75 % |
| training | 12,383 | 20 % |
| everything else (model load, JIT, export, merge, engine start) | 3,413 | 5.5 % (~5.7 min per job) |

Engine init alone: 45-58 s warm, 64-153 s cold.

## 2. The tail

- 64 % of the generation stage (48 % of ALL GPU time of the family) ran after the point where at most 4 sequences
  were still running.
- 13 of 21 configs ran into the deadline with 1-3 sequences alive; those had 131k-147k tokens at the deadline. The
  longest attempt that ever finished with `stop` had 137k tokens; in configs without such a late finisher the
  longest finished attempt had 14k-32k tokens.

Offline replay of the rule "after dispatch is exhausted, once active <= 4, wait 300 s, then exit"
(`inference_levers.tail_exit.replay`):

| family | configs | generation s | saved by the rule | share | finished attempts the rule would censor |
|---|---|---|---|---|---|
| R3 | 3 | 6,610 | 4,055 | 61 % | 1 of 2,760 |
| R4 | 3 | 4,942 | 2,320 | 47 % | 2 of 2,760 |
| R5 | 10 | 46,328 | 29,634 | 64 % | 7 of 9,200 |
| R6 | 5 | 17,134 | 9,038 | 53 % | 3 of 4,598 |

The 13 newly censored attempts are late finishers of 33k-137k tokens on a handful of recurring problems
(2 problems cover 7 of the 13). That is the comparability cost of the rule; the replay lists it per config.

Other generation folders (MEASURED, same replay):
- 80-attempt proof-generation jobs (11 configs, 47,946 GPU-s): the rule saves **84 %**; 80 % of their wall time runs
  with 1-3 sequences in flight; 7 of 11 hit the deadline on 165k-221k-token runaways; 4 attempts newly censored.
- Large sharded generation jobs (12 configs, 95,045 GPU-s): the rule saves only 3 % (they stay busy).

## 3. Real run (TB-1, MEASURED)

A tail-exit run versus a full-deadline reference of the same model on the same prompts, gated by
`inference_levers.tail_exit.compare_runs` (same attempt ids, newly censored <= 3, finish-reason mix, time share):

- generation time 1,822 s vs 5,400 s (0.337x; the reference itself was deadline-bound);
- 1 newly censored attempt, counted as a failure;
- all six checks PASS.

Decision: ADOPTED for new generation families only, where every arm and the reference use the rule; never mixed with
numbers produced under the old rule. A per-attempt token cap was explicitly not used: the rule is a time rule on the
whole job, so short attempts are never truncated.

## 4. Deciding early: bounds instead of waiting

`inference_levers.decide.bound` checks whether a pre-registered paired McNemar decision is already fixed whatever the
uncertain pairs turn out to be. Replay on 40 recorded pairwise decisions: 26 of 40 were already fixed at tail-exit
with u = 4 uncertain pairs, 14 of 40 with u = 8 (MEASURED). Adopted as a read-out label ("INVARIANT by bounds"); it
never changes a verdict.

## 5. Other time sinks (MEASURED)

- **Preemption:** 42.2 of 237 GPU-hours (18 %) sat in preempted job segments (55 segments). Policy adopted: jobs
  under 30 min go to a non-preemptible partition; trainers on preemptible partitions checkpoint every 10 min.
- **Queue wait:** p50 1.4 min, p75 19 min, p90 45 min; 19 % of jobs waited > 30 min (mostly one batch pinned to one
  partition).
- **Training throughput** is nearly flat in row length (1,400-2,050 tok/s from 512 to 16k tokens) and rows above
  4,096 tokens take 71-79 % of training time, so packing short rows can gain at most about 10 % (DERIVED).

## 6. Rejected decision-speed rules (MEASURED replays on the same 40 comparisons)

- **Sequential looks over the attempt index** (`decide.rule_s`, alpha split j/10 over 4 looks): agrees with the full
  test in 38 of 40 (29 of 29 clear cases), but only 2 of 40 stop early; token saving ~1 %. No gain at 4 attempts.
- **Informative sub-panel** (`decide.rule_i`, keep problems whose history pass rate is in [0.05, 0.95]): keeps 23.5 %
  of problems and 48.7 % of tokens, retains 89.0 % of discordant pairs; the pre-registered gate (>= 90 % retained,
  <= 60 % kept per comparison) passes in 17 of 40: FAIL. A post-hoc band ("history not unanimous") looks better
  (40/40, 97.5 % retained, 56.8 % of tokens) but was chosen after seeing the data, so it is not a result.

## Try it

    python scripts/demo_tail_replay.py

runs the replay on a synthetic drain trace (200 attempts, two runaways, one late finisher).
