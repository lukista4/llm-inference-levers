"""Speed-up estimators used by the adopt/reject gates.

1. `abab_gate`: interleaved same-GPU benchmark. Blocks run in the order base, lever, base, lever (ABAB) on the same
   rows with the same seeds; engines are rebuilt per block and engine start is excluded. The noise band is the spread
   between the two base blocks (and between the two lever blocks). Adopt the speed claim only if
     - the mean gain is >= min_gain (default 15 %), and
     - the slower lever block is faster than the faster base block (the gain is outside the noise band).

2. `whole_job_ratio`: a speculative-decoding config changes the speed differently at high and low concurrency.
   With f = share of generation GPU time spent at low concurrency (the "tail"), s_m = speed ratio in the main
   (high-concurrency) phase and s_t = speed ratio in the tail:
       T(lever) / T(base) = (1 - f) / s_m + f / s_t
   and the break-even tail share for a target ratio r is f* = (1/s_m - r) / (1/s_m - 1/s_t).
"""
from __future__ import annotations

from statistics import mean
from typing import Dict, Optional, Sequence


def abab_gate(base_blocks: Sequence[float], lever_blocks: Sequence[float], min_gain: float = 0.15) -> Dict[str, object]:
    """base_blocks / lever_blocks: per-block throughput (e.g. per-sequence decode tok/s), >= 2 blocks each."""
    if len(base_blocks) < 2 or len(lever_blocks) < 2:
        raise ValueError("need at least two blocks per arm to estimate the noise band")
    if min(base_blocks) <= 0 or min(lever_blocks) <= 0:
        raise ValueError("throughput must be positive")
    b, l = mean(base_blocks), mean(lever_blocks)
    gain = l / b - 1.0
    noise_base = (max(base_blocks) - min(base_blocks)) / b
    noise_lever = (max(lever_blocks) - min(lever_blocks)) / l
    outside = min(lever_blocks) > max(base_blocks)
    return {
        "ratio": round(l / b, 4),
        "gain": round(gain, 4),
        "noise_band_base": round(noise_base, 4),
        "noise_band_lever": round(noise_lever, 4),
        "outside_noise": outside,
        "adopt_speed": bool(gain >= min_gain and outside),
    }


def whole_job_ratio(s_main: float, s_tail: float, tail_share: float) -> float:
    if s_main <= 0 or s_tail <= 0:
        raise ValueError("speed ratios must be positive")
    if not 0.0 <= tail_share <= 1.0:
        raise ValueError("tail_share must be in [0, 1]")
    return (1.0 - tail_share) / s_main + tail_share / s_tail


def breakeven_tail_share(s_main: float, s_tail: float, target: float = 0.95) -> Optional[float]:
    """Smallest tail share f at which the whole-job ratio reaches `target`; None if s_main == s_tail."""
    a, c = 1.0 / s_main, 1.0 / s_tail
    if a == c:
        return None
    return (a - target) / (a - c)


def dynamic_schedule_verdict(s_main: float, s_tail_by_n: Dict[str, float], tail_share: float,
                             min_s_main: float = 0.97, max_ratio: float = 0.95) -> Dict[str, object]:
    """Gate for a concurrency-dependent speculation schedule: the main phase may not get slower by more than 3 %
    (s_main >= 0.97) and the whole job must get >= 5 % faster (ratio <= 0.95). s_tail is the conservative minimum
    over the low-concurrency levels."""
    if not s_tail_by_n:
        raise ValueError("need at least one tail level")
    s_t = min(s_tail_by_n.values())
    t = whole_job_ratio(s_main, s_t, tail_share)
    sm_pass, t_pass = s_main >= min_s_main, t <= max_ratio
    return {"s_main": s_main, "s_tail": s_t, "T_ratio": round(t, 4),
            "f_breakeven": breakeven_tail_share(s_main, s_t, max_ratio),
            "s_main_pass": sm_pass, "T_pass": t_pass, "pass": bool(sm_pass and t_pass)}
