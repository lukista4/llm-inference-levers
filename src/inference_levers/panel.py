"""Cost accounting for an LLM grader panel (LLM-as-judge marking): duplicates, dead lanes, re-asks.

- `dedupe_split` / `dedupe_fanout`: grade each distinct (problem, submission text) once and fan the grade out to every
  original submission id. Whitespace runs are collapsed before hashing; nothing else is normalised.
- `parse_log` / `lane_health`: per-lane OK / BAD counts, call-seconds wasted in failed calls, and circuit-breaker
  advice (OPEN-QUOTA after a quota/429 failure, OPEN-FAIL after `trip` consecutive failures). Advice only.
- `breaker_replay`: replay a mid-run breaker (stop a lane after `trip` hard failures, probe again after `hold_s`)
  on a call timeline and report calls skipped, seconds saved and OK grades that would have been delayed.
- `reask_stats`: exact-verdict cache potential. A call whose key (lane, prompt version, problem, proof hash) already
  had an OK grade is cache-servable unless the run was a deliberate re-ask (self-consistency, controls, repeats).

Log line format accepted by `parse_log`:
    'YYYY-MM-DD HH:MM:SS <stage> <sid> <lane> attempt <n> rc=<rc> <secs>s chars=<n> <OK|BAD> [message]'
"""
from __future__ import annotations

import collections
import hashlib
import re
import statistics
from datetime import datetime
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

LINE = re.compile(r"^(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d) (\w+) (\S+) (\w+) attempt (\d+) rc=(-?\d+) (\d+)s chars=(\d+) (\w+)(.*)$")
QUOTA = re.compile(r"RESOURCE_EXHAUSTED|\b429\b|quota|rate.?limit", re.I)
RESET = re.compile(r"[Rr]esets? in ([0-9hms]+)")


def _key(pid: str, text: str) -> str:
    return hashlib.sha256((str(pid) + "|" + re.sub(r"\s+", " ", text).strip()).encode("utf-8")).hexdigest()


def dedupe_split(rows: Iterable[Mapping]) -> Tuple[List[dict], Dict[str, List[str]]]:
    groups: Dict[str, dict] = {}
    order: List[str] = []
    seen = set()
    for r in rows:
        if "sid" not in r or "pid" not in r or not isinstance(r.get("submission"), str):
            raise ValueError("row without sid/pid/submission")
        if r["sid"] in seen:
            raise ValueError("duplicate sid %s" % r["sid"])
        seen.add(r["sid"])
        k = "U" + _key(r["pid"], r["submission"])[:12]
        if k not in groups:
            groups[k] = {"rep": dict(r, sid=k), "sids": []}
            order.append(k)
        groups[k]["sids"].append(r["sid"])
    return [groups[k]["rep"] for k in order], {k: groups[k]["sids"] for k in order}


def dedupe_fanout(grades: Iterable[Mapping], mapping: Mapping[str, Sequence[str]]) -> List[dict]:
    out = []
    for g in grades:
        if g.get("sid") not in mapping:
            raise ValueError("grade for unknown unique sid %r" % g.get("sid"))
        for sid in mapping[g["sid"]]:
            out.append(dict(g, sid=sid, dedup_of=g["sid"]))
    return out


def parse_log(lines: Iterable[str], stage: str = "grade") -> List[dict]:
    calls = []
    for i, l in enumerate(lines):
        m = LINE.match(l.rstrip("\n"))
        if not m or (stage and m.group(2) != stage):
            continue
        calls.append({"t": m.group(1), "i": i, "sid": m.group(3), "lane": m.group(4), "attempt": int(m.group(5)),
                      "rc": int(m.group(6)), "secs": int(m.group(7)), "chars": int(m.group(8)),
                      "ok": m.group(9) == "OK", "msg": m.group(10).strip()})
    return calls


def lane_health(calls: Sequence[Mapping], trip: int = 3) -> Dict[str, dict]:
    lanes: Dict[str, List[Mapping]] = collections.defaultdict(list)
    for c in calls:
        lanes[c["lane"]].append(c)
    out = {}
    for lane, cs in sorted(lanes.items()):
        cs = sorted(cs, key=lambda c: (c["t"], c["i"]))
        trail, last_bad = 0, ""
        for c in reversed(cs):
            if c["ok"]:
                break
            trail += 1
            last_bad = last_bad or c["msg"]
        state = "OK"
        if trail >= 1 and QUOTA.search(last_bad):
            state = "OPEN-QUOTA"
        elif trail >= trip:
            state = "OPEN-FAIL"
        ok_s = [c["secs"] for c in cs if c["ok"]]
        bad_s = [c["secs"] for c in cs if not c["ok"]]
        tot = sum(ok_s) + sum(bad_s)
        r = RESET.search(last_bad)
        out[lane] = {"ok": len(ok_s), "bad": len(bad_s),
                     "median_ok_s": statistics.median(ok_s) if ok_s else None,
                     "wasted_s": sum(bad_s), "wasted_share": round(sum(bad_s) / tot, 3) if tot else None,
                     "trailing_bad": trail, "state": state, "reset_in": r.group(1) if r else None}
    return out


def _hard_fail(c: Mapping) -> bool:
    return c["rc"] != 0 or c["chars"] == 0


def breaker_replay(calls: Sequence[Mapping], trip: int = 3, hold_s: float = 1800.0) -> Dict[str, object]:
    """Per lane: after `trip` consecutive hard failures (rc != 0 or empty answer) the lane is skipped for `hold_s`
    seconds; the first call after the hold is a probe. Skipped OK calls count as 'delayed' (the item would be asked
    again later), never as lost."""
    fmt = "%Y-%m-%d %H:%M:%S"
    by_lane: Dict[str, List[Mapping]] = collections.defaultdict(list)
    for c in calls:
        by_lane[c["lane"]].append(c)
    res = {"skipped": 0, "saved_s": 0, "ok_delayed": 0, "per_lane": {}}
    for lane, cs in by_lane.items():
        cs = sorted(cs, key=lambda c: (c["t"], c["i"]))
        streak, open_until = 0, None
        sk = sv = dl = 0
        for c in cs:
            t = datetime.strptime(c["t"], fmt).timestamp()
            if open_until is not None and t < open_until:
                sk += 1
                sv += c["secs"]
                dl += int(c["ok"])
                continue
            open_until = None
            if _hard_fail(c):
                streak += 1
                if streak >= trip:
                    open_until = t + hold_s
                    streak = 0
            else:
                streak = 0
        res["per_lane"][lane] = {"skipped": sk, "saved_s": sv, "ok_delayed": dl}
        res["skipped"] += sk
        res["saved_s"] += sv
        res["ok_delayed"] += dl
    return res


DELIBERATE = ("_sc", "ctl", "selfcons", "rep")


def reask_stats(calls: Sequence[Mapping]) -> Dict[str, dict]:
    """calls: dicts with t, run, prompt_version, lane, pid, proof_sha, ok, secs. Returns per-lane cache potential."""
    seen_ok: Dict[tuple, str] = {}
    out: Dict[str, dict] = collections.defaultdict(lambda: {"calls": 0, "servable": 0, "servable_s": 0, "cross_run": 0})
    for c in sorted(calls, key=lambda c: c["t"]):
        k = (c["lane"], c["prompt_version"], c["pid"], c["proof_sha"])
        d = out[c["lane"]]
        d["calls"] += 1
        intended = any(x in c["run"] for x in DELIBERATE)
        if k in seen_ok and not intended:
            d["servable"] += 1
            d["servable_s"] += c["secs"]
            d["cross_run"] += int(seen_ok[k] != c["run"])
        elif c["ok"] and not intended:
            seen_ok[k] = c["run"]
    return dict(out)
