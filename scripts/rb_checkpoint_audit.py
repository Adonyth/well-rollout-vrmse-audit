#!/usr/bin/env python3
"""Aggregate the Rayleigh-Benard CHECKPOINT audit into the cells the paper cites.

Rayleigh-Benard is the paper's second audited dataset. Where the benchmark-wide census
(scripts/well_denominator_census.py) is data-only and can therefore only establish that the
NECESSARY condition for the artifact is present, this pass runs the four public RB baselines
themselves, so the mechanism can be confirmed rather than flagged.

Everything here is derived from the packaged per-window scalars in fixtures/rb_models/
(and fixtures/rb_spread/ for the sampling control); nothing is transcribed by hand.
All scores are in RAW PHYSICAL UNITS, the convention the published validation loop uses
(it denormalizes predictions before scoring) and the one rt_model_eval.py/rb_model_eval.py
store: mse and target_variance_ddof1 are computed on denormalized fields.

Published RB comparison cells are QUOTED from the benchmark paper, not recomputed:
  Table 2 (test, one-step): FNO 0.8395, TFNO 0.6566, U-net 1.4860, CNextU-net 0.6699
  Table 3 (test, rollout windows 6-12 and 13-30): ">10" for all four baselines, both windows
"""
import glob
import gzip
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
HARNESS = os.path.dirname(HERE)
sys.path.insert(0, HERE)
from aggregate_results import field_mean_vrmse, per_field_vrmse  # noqa: E402

FLOORS = {"definitional": None, "eps_1e9": 1e-9, "library": 1e-7, "eps_fix": 1e-5}
WINDOWS = {"6-12": (6, 12), "13-30": (13, 30)}
FIELDS = ["buoyancy", "pressure", "velocity_x", "velocity_y"]
# Quoted from the benchmark paper's Table 2 (test split, one step).
PUBLISHED_ONESTEP = {"FNO": 0.8395, "TFNO": 0.6566,
                     "UNetClassic": 1.4860, "UNetConvNext": 0.6699}
# A start is called quiescent when the least-conditioned field's spatial variance sits at or
# below the better-conditioned floor the benchmark's own successor documents.
QUIESCENT_THRESHOLD = 1e-5
# Test-split geometry, QUOTED from the library's own test index rather than recomputed here:
# WellDataset(rayleigh_benard, split="test", n_steps_input=4, n_steps_output=1) reports 34300
# windows = 35 files x 5 trajectories x 196 starts. Recorded so the paper's coverage
# denominators are sourced rather than asserted in prose.
SPLIT_GEOMETRY = {"test_files": 35, "trajectories_per_file": 5, "test_trajectories": 175,
                  "starts_per_trajectory": 196, "library_test_index_windows": 34300}
# Provenance of the audited weights, QUOTED from the model hub's commit API (verified
# 2026-08-03) and from arXiv. Every audited checkpoint's initial commit is 2025-03-28;
# the benchmark's first arXiv version is 2024-11-30. The paper states this before any
# result because it bounds what the reproduction gap can be attributed to.
CHECKPOINT_PROVENANCE = {"initial_commit": "2025-03-28",
                         "benchmark_arxiv_v1": "2024-11-30",
                         "days_after_benchmark_paper": 118}


def load(subdir):
    rows, prov = [], {}
    for fp in sorted(glob.glob(os.path.join(HARNESS, "fixtures", subdir, "*.json.gz"))):
        with gzip.open(fp, "rt", encoding="utf-8") as f:
            d = json.load(f)
        rows += d["rows"]
        p = d.get("provenance", {})
        if p.get("repo"):
            prov[p["repo"]] = p.get("revision")
    return rows, prov


def traj_key(r):
    return (r["file"], r["trajectory"])


def rollout_window(rows, model, window, eps):
    """Mean over trajectories of the within-window mean, matching library aggregation."""
    sel = [r for r in rows if r["model"] == model and r["mode"] == "rollout"]
    per = []
    for tk in sorted({traj_key(r) for r in sel}):
        vals = [field_mean_vrmse(r, eps) for r in sel
                if traj_key(r) == tk and window[0] <= r["rollout_step"] <= window[1]]
        if vals:
            per.append(float(np.mean(vals)))
    return float(np.mean(per)) if per else float("nan")


def onestep_estimate(rows, model, eps):
    """Trapezoid over each trajectory's sampled start grid, then mean over trajectories.

    The published cell averages uniformly over ALL starts; our grid is dense early (where the
    score moves fastest) and coarser later, so a plain mean would over-weight the quiescent
    phase. The trapezoid weights each sampled start by the span it represents.
    """
    sel = [r for r in rows if r["model"] == model and r["mode"] == "onestep"]
    per = []
    for tk in sorted({traj_key(r) for r in sel}):
        sub = sorted([r for r in sel if traj_key(r) == tk], key=lambda r: r["input_start"])
        xs = np.array([r["input_start"] for r in sub], dtype=float)
        ys = np.array([field_mean_vrmse(r, eps) for r in sub], dtype=float)
        if len(xs) > 1:
            per.append(float(np.trapezoid(ys, xs) / (xs[-1] - xs[0])))
    return float(np.mean(per)) if per else float("nan")


def _phase_mean(sub, eps, weighted):
    """Mean score over one phase of one trajectory.

    The 42-point start grid is deliberately NON-UNIFORM (dense early, where the score moves
    fastest, coarse later). onestep_estimate weights by the span each sampled start represents
    for exactly that reason, so an unweighted phase mean would be inconsistent with it and
    would over-weight the densely sampled quiescent starts. We therefore report the
    span-weighted mean as primary and keep the unweighted one alongside it.
    """
    if not sub:
        return float("nan")
    ys = np.array([field_mean_vrmse(r, eps) for r in sub], dtype=float)
    if not weighted or len(sub) < 2:
        return float(np.mean(ys))
    xs = np.array([r["input_start"] for r in sub], dtype=float)
    span = xs[-1] - xs[0]
    if span <= 0:
        return float(np.mean(ys))
    return float(np.trapezoid(ys, xs) / span)


def quiescent_split(rows, model, eps, weighted=True):
    """Partition the one-step starts by whether the flow has developed, per trajectory.

    The partition itself is a selection rule on the DATA (least-conditioned field variance vs
    QUIESCENT_THRESHOLD); eps only sets the floor the two halves are scored under.
    """
    sel = [r for r in rows if r["model"] == model and r["mode"] == "onestep"]
    qs, ds, nq, nd = [], [], 0, 0
    for tk in sorted({traj_key(r) for r in sel}):
        sub = sorted([r for r in sel if traj_key(r) == tk], key=lambda r: r["input_start"])
        qsub = [r for r in sub if min(r["target_variance_ddof1"]) <= QUIESCENT_THRESHOLD]
        dsub = [r for r in sub if min(r["target_variance_ddof1"]) > QUIESCENT_THRESHOLD]
        nq += len(qsub); nd += len(dsub)
        if qsub:
            qs.append(_phase_mean(qsub, eps, weighted))
        if dsub:
            ds.append(_phase_mean(dsub, eps, weighted))
    out = {"quiescent_mean": float(np.mean(qs)) if qs else float("nan"), "n_quiescent": nq,
           "developed_mean": float(np.mean(ds)) if ds else float("nan"), "n_developed": nd,
           "weighting": "span-weighted (trapezoid)" if weighted else "unweighted mean"}
    out["ratio"] = out["quiescent_mean"] / out["developed_mean"] if qs and ds else float("nan")
    return out


def build():
    rows, prov = load("rb_models")
    models = sorted({r["model"] for r in rows})
    trajs = sorted({traj_key(r) for r in rows})
    out = {
        "protocol": {
            "units": "raw physical (predictions denormalized before scoring)",
            "n_rows": len(rows),
            "n_models": len(models),
            "n_trajectories": len(trajs),
            "files": sorted({t[0] for t in trajs}),
            "quiescent_threshold": QUIESCENT_THRESHOLD,
            "split_geometry": SPLIT_GEOMETRY,
            "checkpoint_provenance": CHECKPOINT_PROVENANCE,
            "revisions": prov,
        },
        "published_onestep_table2": PUBLISHED_ONESTEP,
        "rollout_windows": {}, "onestep": {}, "quiescent_developed": {},
        "per_field_rollout_6_12": {},
    }
    for m in models:
        out["rollout_windows"][m] = {
            lab: {fl: rollout_window(rows, m, w, eps) for fl, eps in FLOORS.items()}
            for lab, w in WINDOWS.items()}
        out["onestep"][m] = {fl: onestep_estimate(rows, m, eps) for fl, eps in FLOORS.items()}
        out["onestep"][m]["published"] = PUBLISHED_ONESTEP[m]
        out["onestep"][m]["ratio_library_over_published"] = (
            out["onestep"][m]["library"] / PUBLISHED_ONESTEP[m])
        # BOTH floors: the direction of the split is floor-free, its magnitude is not, and
        # the paper's own rule is to report the floor-free counterpart rather than only the
        # most favourable one. Recording a single floor here is what let an unlabelled ratio
        # reach the abstract.
        out["quiescent_developed"][m] = {
            "library": quiescent_split(rows, m, 1e-7),
            "eps_fix": quiescent_split(rows, m, 1e-5),
            "library_unweighted": quiescent_split(rows, m, 1e-7, weighted=False),
            "eps_fix_unweighted": quiescent_split(rows, m, 1e-5, weighted=False),
            "split_threshold": QUIESCENT_THRESHOLD,
        }
        sel = [r for r in rows if r["model"] == m and r["mode"] == "rollout"
               and WINDOWS["6-12"][0] <= r["rollout_step"] <= WINDOWS["6-12"][1]]
        pf = np.array([per_field_vrmse(r, 1e-7) for r in sel]).mean(axis=0)
        out["per_field_rollout_6_12"][m] = dict(zip(FIELDS, [float(v) for v in pf]))

    # How many of the eight published ">10" rollout cells this audit reproduces, and how many
    # survive the better-conditioned floor.
    cells = [(m, lab) for m in models for lab in WINDOWS]
    out["gt10_summary"] = {
        "n_cells": len(cells),
        "n_reproduced_at_library_floor":
            sum(out["rollout_windows"][m][lab]["library"] > 10 for m, lab in cells),
        "n_still_gt10_at_eps_fix":
            sum(out["rollout_windows"][m][lab]["eps_fix"] > 10 for m, lab in cells),
        "cell_below_10_at_library_floor":
            [f"{m} {lab}" for m, lab in cells
             if out["rollout_windows"][m][lab]["library"] <= 10],
    }

    # Sampling control: does the one-step estimate move when the sampled files span the whole
    # Ra x Pr grid instead of the Prandtl-1 spread the main pass uses?
    srows, _ = load("rb_spread")
    if srows:
        smodels = sorted({r["model"] for r in srows})
        out["spread_control"] = {
            "n_files": len({r["file"] for r in srows}),
            "estimates": {m: {"narrow": onestep_estimate(rows, m, 1e-7),
                              "broad": onestep_estimate(srows, m, 1e-7)}
                          for m in smodels},
        }
    return out


if __name__ == "__main__":
    res = build()
    dest = os.path.join(HARNESS, "fixtures", "rb_models", "rb_checkpoint_audit.json")
    with open(dest, "w", encoding="utf-8") as f:
        json.dump(res, f, indent=1, sort_keys=True, allow_nan=False)
    g = res["gt10_summary"]
    print(f"wrote {dest}")
    print(f"  {res['protocol']['n_rows']} rows, {res['protocol']['n_models']} models, "
          f"{res['protocol']['n_trajectories']} trajectories")
    print(f"  published '>10' rollout cells reproduced at the library floor: "
          f"{g['n_reproduced_at_library_floor']}/{g['n_cells']}"
          f"  (still >10 under eps=1e-5: {g['n_still_gt10_at_eps_fix']})")
