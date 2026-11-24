"""Token-equivalence gate for a generation lever, anchored on a rounding-only repeat floor.

Idea: a lever that should not change outputs (a different batch size, a kernel flag, a KV-cache dtype, speculative
decoding) is compared with the baseline on greedy decoding of a fixed prompt set. Greedy outputs of the *same* engine
already diverge between runs whose forward shapes differ (fp16/bf16 near-tie argmax flips), so "zero divergence" is
the wrong bar. The bar is the baseline's own divergence against a rounding-only control run of the same engine:

    floor   = D(base_run_1, base_run_2)     # e.g. same engine, larger batch
    lever   = D(base_run_1, lever_run)
    PASS iff lever.diverged <= floor.diverged
         and lever.tokens_compared >= min_coverage * floor.tokens_compared

D counts prompts whose greedy token ids differ anywhere on their common prefix. Pure Python, no engine imports.
"""
from __future__ import annotations

from typing import Dict, Mapping, Sequence

Ids = Mapping[str, Sequence[int]]


def divergence(x: Ids, y: Ids) -> Dict[str, object]:
    """Compare two {prompt_id: token_ids} maps over their common prompts and common token prefix."""
    n_cmp = n_div = cmp_tok = 0
    firsts = []
    for k, a in x.items():
        if k not in y:
            continue
        b = y[k]
        m = min(len(a), len(b))
        if m == 0:
            continue
        n_cmp += 1
        cmp_tok += m
        f = next((i for i in range(m) if a[i] != b[i]), None)
        if f is not None:
            n_div += 1
            firsts.append(f)
    firsts.sort()
    return {
        "prompts_compared": n_cmp,
        "diverged": n_div,
        "tokens_compared": cmp_tok,
        "first_div_median": firsts[len(firsts) // 2] if firsts else None,
        "first_div_positions": firsts,
    }


def token_gate(base_a: Ids, base_b: Ids, lever: Ids, min_coverage: float = 0.5) -> Dict[str, object]:
    """Gate a lever run against the floor given by two baseline runs. Returns floor, lever stats and a verdict."""
    if not base_a or not base_b:
        raise ValueError("both baseline runs are required to set the floor")
    floor = divergence(base_a, base_b)
    vs = divergence(base_a, lever)
    coverage_ok = vs["tokens_compared"] >= min_coverage * floor["tokens_compared"]
    passed = bool(coverage_ok and vs["diverged"] <= floor["diverged"])
    return {
        "floor": floor,
        "lever": vs,
        "coverage_ok": coverage_ok,
        "verdict": "PASS" if passed else "FAIL",
    }


def shared_positions(floor_positions: Sequence[int], lever_positions: Sequence[int]) -> Dict[str, object]:
    """Diagnostic (not a gate): how many of the lever's first-divergence positions coincide with the rounding
    control's. Many shared positions point to near-tie argmax flips rather than state corruption."""
    fs = set(floor_positions)
    shared = [p for p in lever_positions if p in fs]
    return {"lever_positions": len(lever_positions), "shared_with_floor": len(shared),
            "share": round(len(shared) / len(lever_positions), 3) if lever_positions else None}


def select_arms(candidates: Mapping[str, object]) -> Dict[str, object]:
    """Keep only result files that are real arms: a dict whose "arm" equals its file stem and that carries greedy ids.

    A results directory also holds merge records, earlier comparison outputs and broken files; a comparison that globs
    every JSON either crashes or, worse, silently compares the wrong thing. `candidates` maps file stem -> parsed JSON
    (or None for an unparsable file). Returns {"arms": {...}, "skipped": [...]}.
    """
    arms, skipped = {}, []
    for stem, r in sorted(candidates.items()):
        if isinstance(r, dict) and r.get("arm") == stem and isinstance((r.get("A1") or {}).get("ids"), dict):
            arms[stem] = r
        else:
            skipped.append(stem)
    return {"arms": arms, "skipped": skipped}
