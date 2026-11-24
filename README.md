# llm-inference-levers

A measured study of throughput levers for generating with and fine-tuning a 9B model (Qwen3.5-9B) on data-centre
GPUs (NVIDIA L40S, L40, A100-40GB, A40) in a university Slurm cluster and a PBS cluster. Each lever went through a
pre-registered gate and was then ADOPTED, kept as a CANDIDATE, REJECTED or CLOSED. Rejected levers are reported with the
same detail as adopted ones.

This repository holds:
- the write-ups with regime-tagged result tables (`docs/`);
- the data-free gate and accounting logic as a small Python package (`src/inference_levers/`), with CPU tests on
  synthetic data. No model, GPU, vLLM or torch is needed to run it.

Run data, logs, prompts and grades are not included.

Author: Lukáš Staník.

## Main results

All MEASURED unless labelled. They come from single runs with single-agent checks. Full regimes are in `docs/`.

| area | finding |
|---|---|
| generation tail | In one experiment family, 48 % of ALL GPU time ran while at most 4 sequences were still generating. A tail-exit rule (once active <= 4, hold 300 s, exit) cut generation time to 0.337x of a deadline-bound reference, with 1 newly censored attempt. Replayed on 80-attempt jobs it saves 84 % of GPU-seconds. ADOPTED for new families ([docs/TAIL_EXIT.md](docs/TAIL_EXIT.md)). |
| speculative decoding (MTP k=3) | 1.80x per sequence on the real think-on workload at batch 4 (acceptance 64 %); 1.60-2.23x at 1-48 active sequences; **0.81x at 256 active**. A dynamic schedule (k=0 at high concurrency) still makes the whole job 1.285x slower: loading a speculative config costs time even at k=0. |
| speculative decoding: reproducibility | Without speculation, greedy continuations from a cached prefix are token-identical cold vs warm (0/9 differ). With MTP or n-gram speculation, 6/9 differ. The outputs depend on prefix-cache and scheduling state. This breaks paired experiments but does not affect sampling quality (that gate passed). |
| fp8 KV cache | 1.18x throughput and 1.90x KV capacity, but lossy: the token gate failed (25/50 diverged vs a floor of 18) and the max teacher-forced log-prob change was 0.496 (limit 0.2). REJECTED. |
| runtime multi-LoRA | One engine serves 2-8 adapters at 1.023x the speed of a merged model, with all equivalence checks passing. ADOPTED. |
| `max_num_seqs` 512 | 0.985x, because the KV cache binds first (mean batch 259). REJECTED. |
| training | A bf16 LoRA branch gives 1.29x (interleaved) with a small, stated NLL bias, so it is ADOPTED only where every arm uses it. Exact-math fused LoRA, a gradient bank and selective checkpointing all failed their gates ([docs/TRAINING_LEVERS.md](docs/TRAINING_LEVERS.md)). |
| LLM-as-judge marking cost | 28 % of grader-panel call-hours produced no grade. One run graded the same proof up to several times (48 submissions, 18 distinct). A mid-run circuit breaker would have saved 18.3 call-hours with 0 grades lost. Batching 4 proofs per request was VETOED: it lowered the catch rate ([docs/MARKING_COST.md](docs/MARKING_COST.md)). |

Production occupancy of an in-process vLLM judge reader is documented in the sibling repository `judge-lab-public`, in
`docs/PERFORMANCE.md`: 195-328 tok/s per L40S, tail-bound shards, and the 1.848x MTP result at 24 active sequences.
It is linked here, not repeated.

## Negative results, stated plainly

1. fp8 KV cache is lossy for this model and engine version.
2. A dynamic speculation schedule cannot rescue high-concurrency throughput (whole-job ratio 1.285).
3. Speculative decoding breaks cold/warm greedy reproducibility.
4. `max_num_seqs` 512 gives no gain: the KV cache binds first.
5. Exact-math fused LoRA: the gradient error is 5-10x the fp32 reference error, so it was REJECTED despite 1.14-1.18x.
6. A gradient bank and selective checkpointing failed their parity or NLL gates.
7. Sequential looks over the attempt index save ~1 % at 4 attempts. The pre-registered informative-panel rule failed its gate (17/40).
8. Batching proofs into one grader request lowered that grader's catch rate. VETO.
9. Spending GPU on training levers moves the whole job little: training is 20 % of GPU time.

## Method

See [docs/METHOD.md](docs/METHOD.md). In short:
- every gate is pre-registered;
- greedy token-equivalence is measured against a rounding-only repeat floor, not against zero;
- speed is timed interleaved in one job (node-to-node spread was ~14 %);
- training levers are judged against rounding-only control floors;
- an adopted lever applies only to new experiment families. Numbers with and without it are never compared.

## Package

| module | what it does |
|---|---|
| `token_gate` | greedy divergence vs a rounding floor; shared-position diagnostic; result-file arm filter |
| `speedup` | interleaved ABAB speed gate; whole-job model `T = (1-f)/s_m + f/s_t` and its break-even |
| `tail_exit` | tail-exit replay, low-concurrency share, tail-exit run vs reference gate |
| `decide` | exact McNemar decide-on-bounds; the replayed (rejected) sequential and informative-panel rules |
| `panel` | grader-panel accounting: dedupe and fan-out, lane health, breaker replay, re-ask cache potential |
| `synth` | synthetic drain traces for demos and tests |

```
python -m pip install -e .[test]
python -m pytest -q
python scripts/demo_tail_replay.py
```

Python >= 3.10, no runtime dependencies.

## Status and limits

- These are engineering measurements for one model family and one engine version (vLLM 0.33.0). They are not
  general claims about vLLM or speculative decoding.
- Single runs and single-agent checks. Nothing here was replicated by an independent party.
- The numbers in `docs/` were taken from the run notes written at harvest time. The raw result files were not
  re-opened for this write-up.

## License

This repository: public domain (the Unlicense), see [LICENSE](LICENSE).

## AI usage disclosure

This work is run as a human-directed, AI-agent-executed workflow.

- **My role:** I set the goals, priorities, constraints and budgets, chose what to pursue or drop, ran it on my own hardware and accounts, and made the key decisions.
- **The agents' role:** most of the code, experiment runs, analysis and documentation were produced by AI agents:
  - Anthropic Claude (Claude Code) wrote most code and docs; commits carry a `Co-Authored-By: Claude` trailer;
  - OpenAI GPT models via Codex did code and statistics reviews;
  - Google Gemini did design and code reviews.
- **Models as instruments:** several model families (including Mistral-hosted models) were also used as graders. Those grades are model judgements, not human ground truth, and are labelled as such.
- **Checks:** results were checked by automated tests and by reviews from a different model family than the author's. An AI review is not an expert human review; where only one agent checked something, it says so.
- **Measurements:** all numbers come from real runs on real hardware. Each is labelled MEASURED, DERIVED or PROJECTED.
