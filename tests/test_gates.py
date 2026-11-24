import pytest

from inference_levers import speedup as SP
from inference_levers import token_gate as TG

SAME = {str(i): [1, 2, 3, 4] for i in range(50)}


def flip(ids, which, pos=2):
    out = {k: list(v) for k, v in ids.items()}
    for k in which:
        out[k][pos] = 99
    return out


def test_floor_gate_pass_by_floor():
    base_b = flip(SAME, ["0", "1", "2"])           # rounding-only control diverges on 3 prompts
    lever = flip(SAME, ["3", "4"], pos=1)
    r = TG.token_gate(SAME, base_b, lever)
    assert r["floor"]["diverged"] == 3 and r["lever"]["diverged"] == 2 and r["verdict"] == "PASS"


def test_floor_gate_neg_above_floor():
    r = TG.token_gate(SAME, flip(SAME, ["0"]), flip(SAME, ["1", "2"]))
    assert r["verdict"] == "FAIL"


def test_floor_gate_neg_low_coverage():
    short = {k: v[:1] for k, v in list(SAME.items())[:10]}     # lever produced far fewer comparable tokens
    r = TG.token_gate(SAME, flip(SAME, ["0"]), short)
    assert not r["coverage_ok"] and r["verdict"] == "FAIL"


def test_floor_gate_neg_requires_floor():
    with pytest.raises(ValueError):
        TG.token_gate(SAME, {}, SAME)


def test_shared_positions_diagnostic():
    r = TG.shared_positions([163, 768, 1364], [163, 768, 482])
    assert r["shared_with_floor"] == 2


def test_select_arms_skips_non_arms():
    arm = {"arm": "G0", "A1": {"ids": SAME}}
    cands = {"G0": arm, "M3": {"arm": "M3", "A1": {"ids": SAME}}, "M9": {"arm": "M3", "A1": {"ids": SAME}},
             "MERGED_FROM": {"base": "x"}, "broken": None}
    r = TG.select_arms(cands)
    assert set(r["arms"]) == {"G0", "M3"} and set(r["skipped"]) == {"M9", "MERGED_FROM", "broken"}


def test_abab_adopts_clear_gain():
    r = SP.abab_gate([35.95, 35.92], [64.83, 64.83])
    assert r["adopt_speed"] and r["ratio"] == pytest.approx(1.804, abs=1e-3)


def test_abab_neg_small_gain():
    assert not SP.abab_gate([100, 101], [110, 111])["adopt_speed"]


def test_abab_neg_inside_noise():
    r = SP.abab_gate([80, 120], [118, 140])          # +29 % mean, but overlaps the base spread
    assert r["gain"] > 0.15 and not r["outside_noise"] and not r["adopt_speed"]


def test_abab_neg_single_block():
    with pytest.raises(ValueError):
        SP.abab_gate([1.0], [2.0, 2.0])


def test_whole_job_model_reproduces_recorded_dynamic_schedule_numbers():
    # Recorded: s_m 0.604 at 256 active, s_t 1.977 (min over n = 1, 4, 16), tail share f = 0.322
    r = SP.dynamic_schedule_verdict(0.604, {"1": 2.18, "4": 2.17, "16": 1.977}, 0.322)
    assert r["T_ratio"] == pytest.approx(1.285, abs=1e-3)
    assert r["f_breakeven"] == pytest.approx(0.61, abs=0.01)
    assert not r["pass"] and not r["s_main_pass"] and not r["T_pass"]


def test_whole_job_model_pass_case_and_bounds():
    assert SP.dynamic_schedule_verdict(1.0, {"1": 2.0}, 0.322)["pass"]
    with pytest.raises(ValueError):
        SP.whole_job_ratio(1.0, 1.0, 1.5)
    assert SP.breakeven_tail_share(1.0, 1.0) is None
