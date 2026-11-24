import pytest

from inference_levers import decide as D


def test_mcnemar_known_values():
    assert D.mcnemar_p(30, 10) == pytest.approx(0.0022, abs=1e-4)
    assert D.mcnemar_p(19, 16) == pytest.approx(0.7359, abs=1e-4)
    assert D.mcnemar_p(0, 0) == 1.0


def test_bound_invariant_cases():
    assert D.bound(19, 16, 4)["verdicts_reachable"] == ["NO_DIFF"]
    r = D.bound(30, 10, 4)
    assert r["status"] == "INVARIANT" and r["now"] == "A_BETTER"


def test_bound_neg_borderline_is_open():
    r = D.bound(28, 14, 4)
    assert r["now"] == "A_BETTER" and r["status"] == "OPEN" and "NO_DIFF" in r["verdicts_reachable"]


def test_bound_neg_bad_input():
    with pytest.raises(ValueError):
        D.bound(1, 1, -1)
    with pytest.raises(ValueError):
        D.mcnemar_p(-1, 0)
    assert D.bound(0, 0, 40)["status"] == "OPEN"


def table(n=60, K=4, win=0):
    ta, tb = {}, {}
    for p in range(n):
        for k in range(K):
            key = ("q%d" % p, k)
            base = 1 if p % 2 == 0 else 0
            ta[key] = 1 if p < win else base
            tb[key] = 0 if p < win else base
    return ta, tb


def test_rule_s_early_reject_and_no_false_early_stop():
    ta, tb = table(win=20)
    r = D.rule_s(ta, tb)
    assert r["decided"]["verdict"] == "A_BETTER" and r["decided"]["look"] == 1 and r["agrees_with_full"]
    ta, tb = table(win=0)
    r = D.rule_s(ta, tb)
    assert r["decided"]["verdict"] == "NO_DIFF" and r["decided"]["by"] != "reject"


def test_spend_sums_to_alpha():
    assert sum(D.spend(0.05, 4)) == pytest.approx(0.05)


def test_rule_i_history_must_be_disjoint():
    ta, tb = table(win=5)
    t = {"A": ta, "B": tb, "H": ta}
    for hist in (["A"], [], ["nope"]):
        with pytest.raises(ValueError):
            D.rule_i(t, "A", "B", hist)


def test_rule_i_keeps_informative_problems():
    ta, tb = table(n=60, win=10)
    h = {("q%d" % p, k): (k % 2) if p < 20 else (1 if p % 2 == 0 else 0) for p in range(60) for k in range(4)}
    r = D.rule_i({"A": ta, "B": tb, "H": h}, "A", "B", ["H"])
    assert r["kept"] == 20 and r["discordant_retained_share"] == 1.0
