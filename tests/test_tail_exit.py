import json

import pytest

from inference_levers import tail_exit as TE
from inference_levers.synth import drain_trace


def att(i, s, e, reason="stop", tok=100):
    return {"attempt_id": "p:%d:0" % i, "started_at": s, "finished_at": e, "completion_tokens": tok,
            "finish_reason": reason, "not_started": None}


def rows():
    r = [att(i, i, 50 + i) for i in range(8)]
    r.append(att(8, 8, 900, "stop", 50000))          # late finisher: the comparability cost of the rule
    r.append(att(9, 9, 5400, "deadline", 140000))    # runaway censored by the wall deadline anyway
    return r


def test_rule_fires_and_lists_newly_censored():
    r = TE.replay(rows(), n=4, hold=300)
    assert r["rule_fires"]
    assert abs(r["exit_s"] - 309) <= 60
    assert r["censored_at_exit"] == 2
    assert [x["attempt_id"] for x in r["newly_censored"]] == ["p:8:0"]
    assert r["saved_s"] > 4000


def test_neg_no_fire_when_run_completes_before_hold():
    r = TE.replay([att(i, i, 50 + i) for i in range(8)], n=4, hold=300)
    assert not r["rule_fires"] and r["saved_s"] == 0.0 and r["newly_censored"] == []


def test_neg_never_fires_with_undispatched_attempts():
    extra = {"attempt_id": "p:99:3", "started_at": None, "finished_at": None, "completion_tokens": 0,
             "finish_reason": "deadline", "not_started": True}
    assert not TE.replay(rows() + [extra], n=4, hold=300)["rule_fires"]


def test_neg_n_zero_never_censors_running_work():
    assert not TE.replay(rows(), n=0, hold=300)["rule_fires"]


def test_neg_empty():
    assert "error" in TE.replay([], 4, 300)


def test_synthetic_drain_has_a_long_tail():
    tr = drain_trace(n=200, seed=1)
    lc = TE.low_concurrency_share(tr)
    assert lc["max_active"] == 200
    assert lc["share_le3"] > 0.3            # the runaways dominate wall time at low concurrency
    r = TE.replay(tr, n=4, hold=300)
    assert r["rule_fires"] and r["saved_s"] / r["gen_s"] > 0.3
    assert len(r["newly_censored"]) == 1     # the single late finisher


def test_low_concurrency_share_simple():
    r = [att(0, 0, 10), att(1, 0, 20)]
    lc = TE.low_concurrency_share(r, thresholds=(1, 2))
    assert lc["share_le1"] == 0.5 and lc["share_le2"] == 1.0


META = {"k": 4, "context": 262144, "dtype": "float16", "engine_version": "x", "gpu": "L40S", "sampling": {"t": 0.7}}


def pair(new_cens=0, tb_gen=1700.0):
    rr = {"a:%d" % i: {"finish_reason": "stop", "not_started": None} for i in range(100)}
    rr["a:0"]["finish_reason"] = "deadline"
    rt = json.loads(json.dumps(rr))
    for i in range(1, 1 + new_cens):
        rt["a:%d" % i]["finish_reason"] = "deadline"
    mr = dict(META, status="deadline", gen_seconds=5400.0)
    mt = dict(META, status="deadline", gen_seconds=tb_gen, tail_exit={"n": 4, "hold_s": 300})
    return mt, rt, mr, rr


def test_compare_pass():
    assert TE.compare_runs(*pair(new_cens=1))["verdict"] == "PASS"


@pytest.mark.parametrize("kw,check", [({"new_cens": 5}, "C3_newly_censored_le_max"), ({"tb_gen": 5000.0}, "C6_time_share")])
def test_compare_neg_fail(kw, check):
    r = TE.compare_runs(*pair(**kw))
    assert r["verdict"] == "FAIL" and not r["checks"][check]


def test_compare_neg_regime_mismatch_refused():
    mt, rt, mr, rr = pair()
    mt["gpu"] = "A100-40GB"
    assert TE.compare_runs(mt, rt, mr, rr)["verdict"] == "REFUSE"


def test_compare_neg_lost_attempt_and_missing_tail_exit():
    mt, rt, mr, rr = pair()
    del rt["a:50"]
    assert TE.compare_runs(mt, rt, mr, rr)["verdict"] == "FAIL"
    mt, rt, mr, rr = pair()
    del mt["tail_exit"]
    assert TE.compare_runs(mt, rt, mr, rr)["verdict"] == "FAIL"
