#!/usr/bin/env python3
"""Validate normscreen against the audit's own frozen census.

The instrument is only worth shipping if, run independently, it reproduces the numbers
the paper obtained by hand. This test takes the frozen per-dataset minimum variances
from the paper's benchmark map and checks that normscreen's floor_share agrees to the
printed precision, and that its conditioning bands agree with the paper's categories.

It also round-trips the real per-window variances of the two checkpoint-audited
datasets through the screen, so the agreement is not only on a summary statistic.

Run:  python3 test_against_paper.py
"""
import glob
import gzip
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from normscreen import band_for, floor_share  # noqa: E402


def _find_fixtures() -> str:
    """Locate the audit's frozen fixtures from either checkout layout.

    The instrument ships inside the harness repository but is also developed
    beside it, so ``fixtures/`` sits either in this directory's parent or one
    level further up. Search rather than hard-code, and fail loudly if absent:
    a missing fixture tree must not silently degrade this suite into a
    zero-check pass.
    """
    probe = os.path.join("fixtures", "generalization", "benchmark_map", "MAP.json")
    d = HERE
    for _ in range(4):
        for base in (d, os.path.join(d, "repro-harness")):
            if os.path.exists(os.path.join(base, probe)):
                return os.path.join(base, "fixtures")
        d = os.path.dirname(d)
    raise SystemExit(
        "cannot locate the audit fixtures (fixtures/generalization/benchmark_map/"
        "MAP.json) above " + HERE + ". This suite validates the screen against the "
        "audit's frozen census; without the fixtures it verifies nothing, so it "
        "fails rather than reporting a pass."
    )


FIXTURES = _find_fixtures()
MAP = os.path.join(FIXTURES, "generalization", "benchmark_map", "MAP.json")
RB = os.path.join(FIXTURES, "rb_models")
EPS_LIB = 1e-7
# The census sizes quoted in README.md and in the paper. Pinned so a truncated or
# partially-fetched fixture tree fails instead of passing on a subset.
EXPECT_MAP_ROWS = 17
EXPECT_RB_WINDOWS = 1728
# RELATIVE tolerance. An absolute 5e-6 would accept a row whose stored share is 8e-6 at
# any value below 1.3e-5 -- i.e. barely one significant figure -- while the claim being
# made is agreement to the printed precision. Relative tolerance means the claim and the
# check are the same statement.
TOL = 5e-6


def check_map() -> tuple[int, int, float]:
    with open(MAP, encoding="utf-8") as f:
        m = json.load(f)
    rows = m["rows"] if isinstance(m, dict) and "rows" in m else m
    ok = bad = 0
    worst = 0.0
    print(f"{'dataset':38s} {'min Var':>11}  {'paper':>9}  {'normscreen':>10}  band")
    for r in rows:
        mv, paper = r.get("min_var"), r.get("floor_share_epslib")
        if mv is None or paper is None:
            continue
        got = float(floor_share(mv, EPS_LIB))
        rel = abs(got - paper) / abs(paper) if paper else abs(got)
        worst = max(worst, rel)
        agree = rel <= TOL
        ok, bad = (ok + 1, bad) if agree else (ok, bad + 1)
        print(f"{r['dataset'][:38]:38s} {mv:11.3e}  {paper:9.6f}  {got:10.6f}  "
              f"{band_for(got):18s}{'' if agree else '   <-- MISMATCH'}")
    return ok, bad, worst


def check_quoted_agreement(worst: float) -> int:
    """Pin the worst-case relative agreement the paper prints for this suite.

    The manuscript quotes this figure in the sentence that introduces the
    instrument. Asserting it here means the printed number cannot drift away
    from the code that produces it without a test going red.
    """
    ref = os.path.join(FIXTURES, "numbers_reference.json")
    if not os.path.exists(ref):
        print(f"  FAIL: {ref} not found; cannot check the quoted agreement")
        return 1
    with open(ref, encoding="utf-8") as f:
        quoted = json.load(f)["instrument_agreement"]["worst_relative_difference"]
    agree = abs(worst - quoted) <= 1e-9 * max(abs(quoted), 1e-12)
    print(f"  worst relative difference {worst:.6e}  vs numbers.json {quoted:.6e}  "
          f"-> {'OK' if agree else 'DRIFT'}")
    return 0 if agree else 1


def check_rb_windows() -> tuple[int, int]:
    """Feed the real per-window target variances through the screen."""
    rows = []
    for fp in sorted(glob.glob(os.path.join(RB, "*.json.gz"))):
        with gzip.open(fp, "rt", encoding="utf-8") as f:
            rows += json.load(f)["rows"]
    if len(rows) != EXPECT_RB_WINDOWS:
        print(f"  FAIL: expected {EXPECT_RB_WINDOWS} windows under {RB}, found {len(rows)}")
        return 0, 1
    var = np.array([min(r["target_variance_ddof1"]) for r in rows], dtype=np.float64)
    shares = np.asarray(floor_share(var, EPS_LIB))
    dominated = int(np.sum(shares >= 0.90))
    worst = float(shares.max())
    print(f"  {len(rows)} Rayleigh-Benard windows, least-conditioned field per window")
    print(f"  worst floor share {worst:.6f} ({band_for(worst)})")
    print(f"  floor-dominated in {dominated}/{len(rows)} windows "
          f"({100*dominated/len(rows):.1f}%)")
    # the paper's claim for this dataset is a >=0.9999 worst-case share; assert the
    # instrument sees the same regime rather than a different one
    agree = worst >= 0.99
    print(f"  agrees with the paper's floor-dominated verdict: {agree}")
    return (1, 0) if agree else (0, 1)


if __name__ == "__main__":
    print("=== normscreen vs the audit's frozen benchmark map ===\n")
    ok1, bad1, worst1 = check_map()
    print(f"\n  {ok1} rows agree, {bad1} disagree (tolerance {TOL:g})")
    if ok1 + bad1 != EXPECT_MAP_ROWS:
        print(f"  FAIL: expected {EXPECT_MAP_ROWS} comparable rows in MAP.json, "
              f"got {ok1 + bad1}")
        bad1 += 1
    bad1 += check_quoted_agreement(worst1)
    print()
    print("=== normscreen on the real per-window variances ===")
    ok2, bad2 = check_rb_windows()
    bad = bad1 + bad2
    print("\n" + ("PASS: the instrument reproduces the audit's conditioning numbers."
                  if not bad else f"FAIL: {bad} disagreement(s)."))
    raise SystemExit(1 if bad else 0)
