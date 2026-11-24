"""Generation-tail analysis and the tail-exit rule.

A batch generation job (many attempts, continuous batching) drains: most attempts finish early, a few long ones keep
the GPU busy at very low concurrency until a wall deadline. The tail-exit rule:

    after dispatch is exhausted, at the first time t with active(t) <= N, wait HOLD seconds;
    exit at t + HOLD (attempts still in flight are censored exactly like a wall deadline).

`replay` evaluates the rule offline on a per-attempt timing trace (start, finish, tokens, finish reason) and lists
the attempts the rule would censor although the real run finished them (the comparability cost). `low_concurrency_share`
measures how much of the wall time runs with few sequences in flight. `compare_runs` is the gate for a real tail-exit
run versus a full-deadline reference run of the same model on the same prompts.

Attempt record fields used: attempt_id, started_at, finished_at, completion_tokens, finish_reason, not_started.
Text and references are never needed.
"""
from __future__ import annotations

import collections
from typing import Dict, Iterable, List, Mapping, Optional

CENSORED = ("deadline", "stall", "length")
SAME_REGIME = ("k", "context", "dtype", "engine_version", "gpu", "sampling")


def _started(rows: Iterable[Mapping]) -> List[Mapping]:
    return [r for r in rows if not r.get("not_started") and r.get("started_at") is not None
            and r.get("finished_at") is not None]


def replay(rows: List[Mapping], n: int = 4, hold: float = 300.0) -> Dict[str, object]:
    st = _started(rows)
    if not st:
        return {"error": "no started attempts"}
    t0 = min(r["started_at"] for r in st)
    t_end = max(r["finished_at"] for r in st)
    t_disp = max(r["started_at"] for r in st)
    not_started = sum(1 for r in rows if r.get("not_started"))
    ends = sorted(r["finished_at"] for r in st)

    def active(t: float) -> int:
        return sum(1 for r in st if r["started_at"] <= t < r["finished_at"])

    exit_t: Optional[float] = None
    if not_started == 0:  # never censor while work is still waiting to be dispatched
        for t in [t_disp] + [e for e in ends if e >= t_disp]:
            if active(t) <= n:  # active is non-increasing once dispatch is exhausted
                exit_t = t + hold
                break
    res: Dict[str, object] = {
        "attempts": len(rows), "started": len(st), "not_started": not_started, "gen_s": round(t_end - t0, 1),
        "real_censored": sum(1 for r in st if r["finish_reason"] in CENSORED),
        "max_tokens_finished_stop": max([r.get("completion_tokens") or 0 for r in st if r["finish_reason"] == "stop"] or [0]),
    }
    if exit_t is None or exit_t >= t_end:
        res.update(rule_fires=False, exit_s=res["gen_s"], saved_s=0.0, newly_censored=[], censored_at_exit=0)
        return res
    infl = [r for r in st if r["started_at"] <= exit_t < r["finished_at"]]
    newly = [r for r in infl if r["finish_reason"] not in CENSORED]
    res.update(
        rule_fires=True, exit_s=round(exit_t - t0, 1), saved_s=round(t_end - exit_t, 1), censored_at_exit=len(infl),
        newly_censored=[{"attempt_id": r["attempt_id"], "finish_reason": r["finish_reason"],
                         "completion_tokens": r.get("completion_tokens"),
                         "finished_after_exit_s": round(r["finished_at"] - exit_t, 1)} for r in newly],
        tail_tokens_at_real_end=[r.get("completion_tokens") for r in infl],
    )
    return res


def low_concurrency_share(rows: List[Mapping], thresholds=(3, 8)) -> Dict[str, object]:
    """Share of wall time (first start .. last finish) with 1..k attempts in flight, for each k in thresholds."""
    st = _started(rows)
    if not st:
        raise ValueError("no started attempts")
    ev = sorted([(r["started_at"], 1) for r in st] + [(r["finished_at"], -1) for r in st])
    t0, total = ev[0][0], ev[-1][0] - ev[0][0]
    acc = {k: 0.0 for k in thresholds}
    a = mx = 0
    prev = t0
    for t, d in ev:
        for k in thresholds:
            if 0 < a <= k:
                acc[k] += t - prev
        a += d
        mx = max(mx, a)
        prev = t
    out = {"wall_s": total, "max_active": mx}
    for k in thresholds:
        out["share_le%d" % k] = round(acc[k] / total, 4) if total else 0.0
    return out


def compare_runs(meta_tb: Mapping, rows_tb: Mapping[str, Mapping], meta_ref: Mapping, rows_ref: Mapping[str, Mapping],
                 max_new: int = 3, max_stop_diff: int = 12, max_share: float = 0.6) -> Dict[str, object]:
    """Gate a tail-exit run (TB) against a full-deadline reference (REF). rows_*: attempt_id -> attempt record.

    C1 tail-exit data recorded and status deadline/completed (regime fields must match, else REFUSE)
    C2 identical attempt-id sets
    C3 newly censored attempts <= max_new
    C4 no loss of undispatched attempts
    C5 finish-reason mix within max_stop_diff stops and 6 repetitions
    C6 gen time <= max_share * reference gen time (only when the reference hit its deadline)
    """
    out: Dict[str, object] = {"checks": {}, "notes": []}
    diff = [k for k in SAME_REGIME if meta_tb.get(k) != meta_ref.get(k)]
    if diff:
        return dict(out, verdict="REFUSE", reason="regime differs: " + ",".join(diff))
    c = out["checks"]
    c["C1_tail_exit_recorded"] = isinstance(meta_tb.get("tail_exit"), dict) and meta_tb.get("status") in ("deadline", "completed")
    c["C2_same_attempt_ids"] = set(rows_tb) == set(rows_ref)
    cens_t = {a for a, r in rows_tb.items() if r["finish_reason"] in CENSORED}
    cens_r = {a for a, r in rows_ref.items() if r["finish_reason"] in CENSORED}
    new = sorted(cens_t - cens_r)
    out["newly_censored"] = new
    c["C3_newly_censored_le_max"] = len(new) <= max_new
    ns_t = sum(1 for r in rows_tb.values() if r.get("not_started"))
    ns_r = sum(1 for r in rows_ref.values() if r.get("not_started"))
    c["C4_no_undispatched_loss"] = ns_t <= ns_r
    ft = collections.Counter(r["finish_reason"] for r in rows_tb.values())
    fr = collections.Counter(r["finish_reason"] for r in rows_ref.values())
    c["C5_finish_mix"] = abs(ft["stop"] - fr["stop"]) <= max_stop_diff and abs(ft["repetition"] - fr["repetition"]) <= 6
    gt, gr = meta_tb.get("gen_seconds"), meta_ref.get("gen_seconds")
    out["gen_seconds"] = {"tb": gt, "ref": gr, "ratio": round(gt / gr, 3) if gt and gr else None}
    if meta_ref.get("status") == "deadline" and gt and gr:
        c["C6_time_share"] = gt <= max_share * gr
    else:
        out["notes"].append("reference did not hit its deadline: time share reported only")
    out["verdict"] = "PASS" if all(c.values()) else "FAIL"
    return out
