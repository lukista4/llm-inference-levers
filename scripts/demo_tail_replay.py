"""Demo: tail-exit replay on a SYNTHETIC drain trace (no real data). Run: python scripts/demo_tail_replay.py"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from inference_levers.synth import drain_trace  # noqa: E402
from inference_levers.tail_exit import low_concurrency_share, replay  # noqa: E402


def main():
    rows = drain_trace(n=200, seed=1)
    print("low-concurrency share:", json.dumps(low_concurrency_share(rows)))
    for n, hold in ((4, 300), (4, 600), (8, 300)):
        r = replay(rows, n=n, hold=hold)
        print("N=%d hold=%ds: gen %.0f s -> exit %.0f s (saved %.0f%%), newly censored %d"
              % (n, hold, r["gen_s"], r["exit_s"], 100 * r["saved_s"] / r["gen_s"], len(r["newly_censored"])))


if __name__ == "__main__":
    main()
