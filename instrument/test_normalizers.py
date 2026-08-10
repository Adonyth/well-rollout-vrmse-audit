#!/usr/bin/env python3
"""Regression tests for the two normalizer conventions.

These encode the finding that motivated normalizer support: two peer physics-ML
benchmarks divide by different quantities, and the choice decides whether a quiescent
field is scoreable at all.

  The Well   VRMSE  : divide by (Var + eps),  eps = 1e-7   -> saturates at 1/sqrt(eps)
  PDEBench   nRMSE  : divide by sqrt(mean(x^2)), no eps    -> unbounded at zero signal

Run:  python3 test_normalizers.py
"""
import sys, os
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from normscreen.screen import floor_share, screen_fields, spatial_rms2, spatial_variance

fails = []


def check(name, cond, detail=""):
    print(f"  [{'OK  ' if cond else 'FAIL'}] {name}" + (f"  {detail}" if detail else ""))
    if not cond:
        fails.append(name)


print("=== RMS vs variance: the same field, opposite verdicts ===")
rng = np.random.default_rng(0)
# oscillation about a large offset: RMS ~ 1, variance ~ 1e-12
f = {"offset": 1.0 + 1e-6 * rng.standard_normal((10, 64, 64))}
rv = screen_fields(f, eps=1e-7, n_spatial=2, normalizer="variance").fields[0]
rr = screen_fields(f, eps=1e-7, n_spatial=2, normalizer="rms").fields[0]
check("variance normalizer sees it as floor-determined", rv.band == "floor-determined",
      f"share={rv.floor_share_at_min_variance:.6f}")
check("rms normalizer sees it as well-conditioned", rr.band == "well-conditioned",
      f"share={rr.floor_share_at_min_variance:.6f}")
check("RMS^2 = Var + mean^2 identity holds",
      np.allclose(spatial_rms2(f["offset"], 2),
                  spatial_variance(f["offset"], 2, ddof=0) + f["offset"].mean(axis=(1, 2))**2))

print("\n=== eps = 0 (a metric with no stabilizing constant) ===")
z = {"identically_zero": np.zeros((5, 32, 32)), "healthy": rng.standard_normal((5, 32, 32))}
r0 = screen_fields(z, eps=0.0, n_spatial=2, normalizer="rms")
zf = [x for x in r0.fields if x.name == "identically_zero"][0]
check("floor share is 0 at eps=0, never NaN", np.isfinite(zf.floor_share_at_min_variance)
      and zf.floor_share_at_min_variance == 0.0)
check("zero-denominator frames counted", zf.frames_zero_denominator == 5,
      f"n={zf.frames_zero_denominator}")
check("band reports degeneracy, not a floor band", zf.band == "degenerate-denominator")
check("verdict names the unboundedness", "unbounded" in r0.verdict)
check("scalar floor_share at eps=0 is 0.0", floor_share(0.0, 0.0) == 0.0)

print("\n=== Lesson-1 threshold statistic (regression: caught by the Codex gate leg) ===")
# The statistic a benchmark should publish is the fraction with Var <= eps (share >= 0.5).
# An earlier build reported only the >=0.90 band (Var <= eps/9) and therefore returned 0.0
# for a field sitting exactly at the floor -- silently answering a different question.
x = rng.standard_normal((6, 32, 32))
x *= (1e-7 / x.var(axis=(1, 2), ddof=1).mean()) ** 0.5      # drive Var to ~eps
at_eps = screen_fields({"at_eps": x}, eps=1e-7, n_spatial=2).fields[0]
check("floor share is ~0.5 when Var ~ eps", 0.4 < at_eps.floor_share_at_min_variance < 0.6,
      f"{at_eps.floor_share_at_min_variance:.4f}")
check("Lesson-1 fraction (Var<=eps) is non-zero there",
      at_eps.fraction_at_or_below_floor > 0.0,
      f"{at_eps.fraction_at_or_below_floor:.3f}")
check("the 0.90 band is a STRICTER, separate statistic",
      at_eps.fraction_floor_dominated < at_eps.fraction_at_or_below_floor,
      f"{at_eps.fraction_floor_dominated:.3f} < {at_eps.fraction_at_or_below_floor:.3f}")
# and a mutation test: swapping the threshold back to 0.90 must change the answer
check("the two thresholds are not interchangeable",
      at_eps.fraction_at_or_below_floor != at_eps.fraction_floor_dominated)

print("\n=== saturation vs unboundedness ===")
# with a floor the worst achievable score is bounded by 1/sqrt(eps); without one it is not
mse = np.array([1.0])
bounded = float(np.sqrt(mse / (0.0 + 1e-7))[0])
check("floored metric saturates at 1/sqrt(eps)", abs(bounded - 1 / np.sqrt(1e-7)) < 1e-6,
      f"{bounded:.1f}")
with np.errstate(divide="ignore"):
    unbounded = float(np.sqrt(mse / (0.0 + 0.0))[0])
check("floorless metric is genuinely infinite", not np.isfinite(unbounded))

print("\n" + ("PASS" if not fails else f"FAIL: {fails}"))
raise SystemExit(1 if fails else 0)
