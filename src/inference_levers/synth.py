"""Synthetic generation traces for demos and tests (no real data anywhere in this repository).

`drain_trace` models one batch generation job: `n` attempts dispatched over `dispatch_s` seconds, most finishing after
a log-normal number of tokens, a few "runaways" that would only stop at the wall deadline, and optionally a few long
late finishers that do end with `stop`.
"""
from __future__ import annotations

import math
import random
from typing import List


def drain_trace(n: int = 200, seed: int = 0, decode_tok_s: float = 30.0, median_tokens: float = 6000.0,
                sigma: float = 0.6, runaways: int = 2, late_finishers: int = 1, deadline_s: float = 5400.0,
                dispatch_s: float = 20.0) -> List[dict]:
    rng = random.Random(seed)
    rows = []
    for i in range(n):
        start = dispatch_s * i / max(n - 1, 1)
        if i < runaways:
            toks = int(decode_tok_s * (deadline_s - start))
            fin, reason = deadline_s, "deadline"
        elif i < runaways + late_finishers:
            toks = int(median_tokens * 8)
            fin, reason = min(start + toks / decode_tok_s, deadline_s), "stop"
            if fin >= deadline_s:
                reason = "deadline"
        else:
            toks = int(math.exp(rng.gauss(math.log(median_tokens), sigma)))
            fin = start + toks / decode_tok_s
            reason = "stop"
            if fin >= deadline_s:
                fin, reason = deadline_s, "deadline"
        rows.append({"attempt_id": "p%03d:0" % i, "started_at": round(start, 3), "finished_at": round(fin, 3),
                     "completion_tokens": toks, "finish_reason": reason, "not_started": None})
    return rows
