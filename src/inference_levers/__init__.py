"""inference_levers: data-free gate and accounting logic behind an LLM inference/training lever study.

Modules
- token_gate: greedy token-equivalence gate against a rounding-only repeat floor
- speedup: interleaved (ABAB) speed gate; whole-job model for concurrency-dependent speculative decoding
- tail_exit: generation-tail replay, low-concurrency share, tail-exit run gate
- decide: exact McNemar decide-on-bounds; replayed (and rejected) sequential / informative-panel rules
- panel: LLM grader panel cost accounting (dedupe, lane health, breaker replay, re-ask cache potential)
- synth: synthetic traces for demos and tests
"""
__version__ = "0.1.0"
