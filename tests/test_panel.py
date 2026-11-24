import pytest

from inference_levers import panel as P

LOG = [
    "2026-01-05 14:00:00 grade s1 laneA attempt 1 rc=0 300s chars=900 OK",
    "2026-01-05 14:01:00 grade s1 laneB attempt 1 rc=3 220s chars=0 BAD ERROR: RESOURCE_EXHAUSTED (code 429) Resets in 148h56m.",
    "2026-01-05 14:02:00 grade s2 laneB attempt 1 rc=3 210s chars=0 BAD ERROR: RESOURCE_EXHAUSTED (code 429) Resets in 148h55m.",
    "2026-01-05 14:03:00 grade s1 laneC attempt 1 rc=1 100s chars=0 BAD malformed",
    "2026-01-05 14:04:00 grade s2 laneC attempt 1 rc=1 100s chars=0 BAD malformed",
    "2026-01-05 14:05:00 grade s3 laneC attempt 1 rc=1 100s chars=0 BAD malformed",
    "2026-01-05 14:06:00 grade s1 laneD attempt 1 rc=1 100s chars=0 BAD too long",
    "2026-01-05 14:07:00 grade s1 laneD attempt 2 rc=0 250s chars=800 OK",
    "2026-01-05 14:08:00 prepare P1 laneE attempt 1 rc=3 50s chars=0 BAD RESOURCE_EXHAUSTED",
]


def test_lane_states():
    h = P.lane_health(P.parse_log(LOG))
    assert h["laneB"]["state"] == "OPEN-QUOTA" and h["laneB"]["wasted_s"] == 430
    assert h["laneB"]["reset_in"] == "148h55m"
    assert h["laneC"]["state"] == "OPEN-FAIL"
    assert h["laneA"]["state"] == "OK"


def test_neg_recovered_lane_ok_and_stages_not_mixed():
    h = P.lane_health(P.parse_log(LOG))
    assert h["laneD"]["state"] == "OK"
    assert "laneE" not in h


def test_neg_two_bad_below_trip():
    h = P.lane_health(P.parse_log(LOG[:1] + LOG[3:5]))
    assert h["laneC"]["state"] == "OK"


def test_dedupe_split_and_fanout():
    rows = [{"sid": "a", "pid": "P1", "submission": "proof  one\n"}, {"sid": "b", "pid": "P1", "submission": "proof one"},
            {"sid": "c", "pid": "P2", "submission": "proof one"}, {"sid": "d", "pid": "P1", "submission": "proof two"}]
    u, mp = P.dedupe_split(rows)
    assert len(u) == 3 and sorted(len(v) for v in mp.values()) == [1, 1, 2]
    out = P.dedupe_fanout([{"sid": k, "score": 3} for k in mp], mp)
    assert sorted(g["sid"] for g in out) == ["a", "b", "c", "d"]


def test_dedupe_neg_cases():
    with pytest.raises(ValueError):
        P.dedupe_split([{"sid": "a", "pid": "P", "submission": "x"}, {"sid": "a", "pid": "P", "submission": "y"}])
    with pytest.raises(ValueError):
        P.dedupe_split([{"sid": "a", "pid": "P"}])
    with pytest.raises(ValueError):
        P.dedupe_fanout([{"sid": "Uunknown"}], {"Uother": ["a"]})


def mk(t, lane, ok, secs=100, rc=None):
    rc = (0 if ok else 3) if rc is None else rc
    return "2026-01-05 %s grade s lane%s attempt 1 rc=%d %ds chars=%d %s" % (t, lane, rc, secs, 500 if ok else 0, "OK" if ok else "BAD")


def test_breaker_replay_skips_dead_lane_and_counts_delays():
    lines = [mk("10:00:00", "X", False), mk("10:01:00", "X", False), mk("10:02:00", "X", False),
             mk("10:05:00", "X", False), mk("10:10:00", "X", True), mk("10:40:00", "X", True)]
    r = P.breaker_replay(P.parse_log(lines), trip=3, hold_s=1800)
    assert r["skipped"] == 2 and r["ok_delayed"] == 1 and r["saved_s"] == 200


def test_breaker_neg_ok_resets_streak():
    lines = [mk("10:00:00", "Y", False), mk("10:01:00", "Y", False), mk("10:02:00", "Y", True), mk("10:03:00", "Y", False)]
    assert P.breaker_replay(P.parse_log(lines))["skipped"] == 0


def test_reask_stats_excludes_deliberate_reruns():
    calls = [
        {"t": "1", "run": "r1", "prompt_version": "v1", "lane": "A", "pid": "p", "proof_sha": "h", "ok": True, "secs": 10},
        {"t": "2", "run": "r2", "prompt_version": "v1", "lane": "A", "pid": "p", "proof_sha": "h", "ok": True, "secs": 12},
        {"t": "3", "run": "r2_selfcons", "prompt_version": "v1", "lane": "A", "pid": "p", "proof_sha": "h", "ok": True, "secs": 9},
        {"t": "4", "run": "r3", "prompt_version": "v2", "lane": "A", "pid": "p", "proof_sha": "h", "ok": True, "secs": 9},
    ]
    s = P.reask_stats(calls)["A"]
    assert s["calls"] == 4 and s["servable"] == 1 and s["cross_run"] == 1 and s["servable_s"] == 12
