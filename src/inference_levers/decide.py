"""Decision-speed tools for paired binary comparisons (arm A vs arm B on the same problem / attempt / seed).

`bound`: decide on bounds. With b (A-only) and c (B-only) discordant pairs known and u pairs still uncertain
(a censored attempt, a missing mark, an attempt not generated yet), the pre-registered exact McNemar verdict is
INVARIANT if every completion of the u pairs gives the same verdict; the unfinished work then cannot change the
decision. Exact integer arithmetic, no probability model for the uncertain pairs.

`rule_s` (group-sequential over the attempt index) and `rule_i` (keep only "informative" problems, chosen by a
disjoint history) are the two decision-speed rules that were replayed and rejected; they are kept so the negative
result can be reproduced on synthetic tables.
"""
from __future__ import annotations

from math import comb
from typing import Dict, List, Mapping, Sequence, Tuple

Key = Tuple[str, int]  # (problem id, attempt index)
FUTILITY_MAX_U = 400


def mcnemar_p(b: int, c: int) -> float:
    """Exact two-sided binomial McNemar p-value."""
    if b < 0 or c < 0:
        raise ValueError("negative count")
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    return min(1.0, 2.0 * sum(comb(n, i) for i in range(k + 1)) / 2 ** n)


def verdict(b: int, c: int, alpha: float = 0.05) -> str:
    p = mcnemar_p(b, c)
    if p < alpha and b > c:
        return "A_BETTER"
    if p < alpha and c > b:
        return "B_BETTER"
    return "NO_DIFF"


def bound(b: int, c: int, u: int, alpha: float = 0.05) -> Dict[str, object]:
    if u < 0:
        raise ValueError("negative u")
    seen: Dict[str, Tuple[int, int]] = {}
    for i in range(u + 1):
        for j in range(u - i + 1):
            seen.setdefault(verdict(b + i, c + j, alpha), (i, j))
    vs = sorted(seen)
    return {"b": b, "c": c, "u": u, "alpha": alpha, "now": verdict(b, c, alpha), "p_now": mcnemar_p(b, c),
            "verdicts_reachable": vs, "witness": {k: list(v) for k, v in seen.items()},
            "status": "INVARIANT" if len(vs) == 1 else "OPEN"}


def _counts(ta: Mapping[Key, object], tb: Mapping[Key, object], keys) -> Tuple[int, int, int]:
    b = c = u = 0
    for k in keys:
        x, y = ta.get(k), tb.get(k)
        if x is None or y is None:
            u += 1
        elif x == 1 and y == 0:
            b += 1
        elif x == 0 and y == 1:
            c += 1
    return b, c, u


def spend(alpha: float, looks: int) -> List[float]:
    """Alpha split j / (K(K+1)/2): small early, large late; sums to alpha (union bound)."""
    tot = looks * (looks + 1) / 2
    return [alpha * j / tot for j in range(1, looks + 1)]


def rule_s(ta: Mapping[Key, object], tb: Mapping[Key, object], alpha: float = 0.05) -> Dict[str, object]:
    keys = sorted(set(ta) | set(tb))
    if not keys:
        raise ValueError("no attempts")
    K = 1 + max(k[1] for k in keys)
    probs = sorted({k[0] for k in keys})
    al = spend(alpha, K)
    bF, cF, _ = _counts(ta, tb, keys)
    full = verdict(bF, cF, alpha)
    looks, decided = [], None
    for j in range(1, K + 1):
        b, c, _ = _counts(ta, tb, [k for k in keys if k[1] < j])
        v = verdict(b, c, al[j - 1])
        rem = len(probs) * (K - j)
        fut = rem <= FUTILITY_MAX_U and bound(b, c, rem, al[K - 1])["verdicts_reachable"] == ["NO_DIFF"]
        looks.append({"look": j, "b": b, "c": c, "alpha_j": al[j - 1], "verdict": v, "futile": fut})
        if decided is None and (v != "NO_DIFF" or fut):
            decided = {"look": j, "verdict": v, "by": "reject" if v != "NO_DIFF" else "futility"}
    if decided is None:
        decided = {"look": K, "verdict": looks[-1]["verdict"], "by": "end"}
    return {"K": K, "problems": len(probs), "full_test": {"b": bF, "c": cF, "verdict": full}, "looks": looks,
            "decided": decided, "gens_saved_share": round(1 - decided["look"] / K, 3),
            "agrees_with_full": decided["verdict"] == full}


def rule_i(tables: Mapping[str, Mapping[Key, object]], a: str, b: str, history: Sequence[str],
           lo: float = 0.05, hi: float = 0.95) -> Dict[str, object]:
    if not history:
        raise ValueError("no history configs")
    if a in history or b in history:
        raise ValueError("history must be disjoint from the compared configs (selection bias)")
    for h in history:
        if h not in tables:
            raise ValueError("unknown history config %s" % h)
    ta, tb = tables[a], tables[b]
    probs = sorted({k[0] for k in set(ta) | set(tb)})
    rate = {}
    for p in probs:
        v = [x for h in history for k, x in tables[h].items() if k[0] == p and x is not None]
        rate[p] = sum(v) / len(v) if v else None
    keep = {p for p in probs if rate[p] is not None and lo <= rate[p] <= hi}
    keys = sorted(set(ta) | set(tb))
    b0, c0, _ = _counts(ta, tb, keys)
    b1, c1, _ = _counts(ta, tb, [k for k in keys if k[0] in keep])
    return {"problems": len(probs), "kept": len(keep),
            "discordant_all": b0 + c0, "discordant_kept": b1 + c1,
            "discordant_retained_share": round((b1 + c1) / (b0 + c0), 3) if b0 + c0 else None,
            "full_p": mcnemar_p(b0, c0), "kept_p": mcnemar_p(b1, c1)}
