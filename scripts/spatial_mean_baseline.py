#!/usr/bin/env python3
"""Trivial "always predict this frame's own spatial mean" VRMSE baseline.

Why this script exists: UNetClassic's documented-floor (eps=1e-5) rollout
scores (1.927 at window 6-12, 1.267 at window 13-30) are sometimes described
loosely as "roughly the quality of predicting the field mean." This script
computes the actual mean-predictor baseline and shows the eps=1e-5 floor
score is several times *worse* than a trivial spatial-mean predictor scored
under the identical metric, not comparable to it.

Exact identity used (not an approximation): for N grid points x_1..x_N in one
frame, predicting the constant c = mean(x) gives
    MSE = (1/N) * sum (x_i - c)^2 = population variance (ddof=0) of that frame.
The pipeline stores target_variance_ddof1 (dividing by N-1) per field/row in
results/models/*.json.gz (see aggregate_results.py), so
    mse_meanpred = var_ddof1 * (N-1) / N                     (exact identity)
N = 128**3 = 2,097,152 (RT grid; see fast_reader.py FRAME_VOX / reshape
asserts), so (N-1)/N = 1 - 4.768e-7 -- negligible next to eps=1e-5, but kept
exact rather than dropped since it costs nothing.

VRMSE_meanpred(frame, field) = sqrt(mse_meanpred / (var_ddof1 + eps)),
averaged over the 4 fields per row (matching aggregate_results.field_mean_vrmse)
and window-aggregated the same way aggregate_results.rollout_summary.agg()
aggregates the real model scores: mean over trajectories of each trajectory's
window mean. Same aggregation path that produces the 1.927/1.267 numbers
already machine-checked by verify.py -- so the two numbers are apples-to-apples
comparable, not computed by different methodology.

Usage:
    python3 spatial_mean_baseline.py                 # uses ./results
    P3_RESULTS_DIR=../fixtures python3 spatial_mean_baseline.py
"""
from __future__ import annotations

import glob
import gzip
import json
import os
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
RESULTS_DIR = Path(os.environ.get("P3_RESULTS_DIR", str(HERE.parent / "results")))

EPS_REQ = 1e-5
W1 = (6, 12)
W2 = (13, 30)
N_GRID = 128 ** 3  # confirmed grid size, scripts/fast_reader.py FRAME_VOX
DDOF_CORRECTION = (N_GRID - 1) / N_GRID


def load_model_rows(model: str, results_dir: Path = RESULTS_DIR) -> list[dict]:
    rows = []
    for path in sorted(glob.glob(str(results_dir / "models" / f"{model}_rayleigh*.json.gz"))):
        with gzip.open(path, "rt") as handle:
            payload = json.load(handle)
        rows.extend(payload["rows"])
    return rows


def meanpred_field_vrmse(row: dict, eps: float) -> float:
    var = np.asarray(row["target_variance_ddof1"], dtype=np.float64)
    mse_meanpred = var * DDOF_CORRECTION  # exact identity, see module docstring
    return float(np.mean(np.sqrt(mse_meanpred / (var + eps))))


def window_baseline(model: str, w: tuple[int, int], results_dir: Path = RESULTS_DIR) -> tuple[float, int]:
    rows = load_model_rows(model, results_dir)
    roll = [r for r in rows if r["mode"] == "rollout"]
    trajs = sorted({(r["file"], r["trajectory"]) for r in roll})
    per_traj_means = []
    n_windows = 0
    for tk in trajs:
        sub = sorted([r for r in roll if (r["file"], r["trajectory"]) == tk],
                     key=lambda r: r["rollout_step"])
        vals = [meanpred_field_vrmse(r, EPS_REQ) for r in sub if w[0] <= r["rollout_step"] <= w[1]]
        n_windows += len(vals)
        if vals:
            per_traj_means.append(float(np.mean(vals)))
    return float(np.mean(per_traj_means)), n_windows


def compute(results_dir: Path = RESULTS_DIR, model: str = "UNetClassic") -> dict:
    b1, n1 = window_baseline(model, W1, results_dir)
    b2, n2 = window_baseline(model, W2, results_dir)
    return {
        "model": model,
        "eps": EPS_REQ,
        "n_grid": N_GRID,
        "window_6_12": {"baseline_vrmse": b1, "n_windows": n1},
        "window_13_30": {"baseline_vrmse": b2, "n_windows": n2},
    }


def main() -> int:
    if not RESULTS_DIR.exists():
        print(f"FAIL: results dir not found at {RESULTS_DIR}")
        return 1
    out = compute()
    reported = {"window_6_12": 1.927, "window_13_30": 1.267}
    print(f"Spatial-mean-predictor VRMSE baseline under eps={EPS_REQ}, model={out['model']}")
    print(f"  N_grid = {out['n_grid']} ((N-1)/N = {DDOF_CORRECTION:.10f})")
    for wname, label in [("window_6_12", "6-12"), ("window_13_30", "13-30")]:
        w = out[wname]
        r = reported[wname]
        print(f"  window {label} ({w['n_windows']} windows): baseline = {w['baseline_vrmse']:.4f}   "
              f"reported eps5 model score = {r}   ratio = {r / w['baseline_vrmse']:.3f}x")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
