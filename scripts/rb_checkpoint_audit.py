#!/usr/bin/env python3
"""Aggregate the Rayleigh-Benard CHECKPOINT audit into the cells the paper cites.

Rayleigh-Benard is the paper's second checkpoint-audited dataset, audited on a STRATIFIED
SAMPLE of its test split -- never on the whole split, and the coverage block below carries
the exact denominators. Where the benchmark-wide census (scripts/well_denominator_census.py)
is data-only and can therefore only establish that the NECESSARY condition for the artifact
is present, this pass runs the public RB baselines themselves, so the mechanism can be
confirmed rather than flagged.

Two passes, deliberately traded off against each other:
  * the DEPTH pass (fixtures/rb_models/): four baselines x three Rayleigh numbers at
    Prandtl 1 x two trajectories -- all four models, one corner of the parameter grid;
  * the STRATIFIED pass (fixtures/rb_spread/): two baselines x ten files spanning
    Ra 1e6-1e10 and Pr 0.1-10 x one trajectory -- fewer models, the whole grid.
The stratified pass is what lets the mechanism be tested where the mechanism itself
predicts it should WEAKEN (high Rayleigh number, short quiescent phase), rather than only
where it is strongest. Its quiescent-fraction predictor is a property of the DATA alone
(verified here: identical start grids and target variances across models), so it also
serves as the one place where the paper's data-only screen is checked against
model-scored outcomes at all -- an exploratory check on the same ten strata that present it,
not an estimated screen-to-outcome relation.

Everything here is derived from the packaged per-window scalars in fixtures/rb_models/
and fixtures/rb_spread/; nothing is transcribed by hand.
All scores are in RAW PHYSICAL UNITS, the convention the published validation loop uses
(it denormalizes predictions before scoring) and the one rt_model_eval.py/rb_model_eval.py
store: mse and target_variance_ddof1 are computed on denormalized fields.

Published RB comparison cells are QUOTED from the benchmark paper, not recomputed:
  Table 2 (test, one-step): FNO 0.8395, TFNO 0.6566, U-net 1.4860, CNextU-net 0.6699
  Table 3 (test, rollout windows 6-12 and 13-30): ">10" for all four baselines, both windows
"""
import glob
import gzip
import math
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
# Quoted from the benchmark paper's Table 2 (test split, one step), arXiv:2412.00568v2.
# COLUMN SEMANTICS, recorded for the same reason the Rayleigh-Taylor record carries them:
# Table 2 has one column per model in the header order "FNO | TFNO | U-net | CNextU-net",
# with a single value per model (no window sub-columns), and the rayleigh_benard row reads
# 0.8395, 0.6566, 1.4860, 0.6699 in that order. The published/computed ratios in the paper
# are order-sensitive, so the mapping is stated rather than assumed.
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
# Loaded rather than restated: scripts/fetch_checkpoint_chronology.py records it from the
# model hub's commit API and the benchmark's arXiv submission history.
CHRONOLOGY_FIXTURE = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "fixtures", "provenance", "checkpoint_chronology.json")
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


def quiescent_fraction(rows, fname):
    """Share of a file's sampled one-step starts whose least-conditioned field is quiescent.

    A property of the DATA only -- no model enters it. The caller asserts that every model's
    start grid and target variances on this file agree, so the fraction is well defined for
    the file rather than for a (file, model) pair. This is the quantity the paper's data-only
    screen actually measures, which is why it is the natural predictor to calibrate the
    screen against the model-scored cells.
    """
    grids = {}
    for m in sorted({r["model"] for r in rows}):
        sub = sorted([r for r in rows if r["file"] == fname and r["model"] == m
                      and r["mode"] == "onestep"], key=lambda r: r["input_start"])
        grids[m] = ([r["input_start"] for r in sub],
                    [min(r["target_variance_ddof1"]) for r in sub])
    keys = list(grids)
    for m in keys[1:]:
        if grids[m] != grids[keys[0]]:
            raise AssertionError(f"{fname}: start grid/target variance differs across models; "
                                 "the quiescent fraction would not be a data-only quantity")
    starts, minvars = grids[keys[0]]
    return {"n_starts": len(starts),
            "n_quiescent": int(sum(v <= QUIESCENT_THRESHOLD for v in minvars)),
            "quiescent_fraction": float(np.mean([v <= QUIESCENT_THRESHOLD for v in minvars])),
            "min_field_variance": float(min(minvars)),
            "floor_share_epslib": float(1e-7 / (min(minvars) + 1e-7))}


def _pearson(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    return float(np.corrcoef(x, y)[0, 1])


def _spearman(x, y):
    """Rank correlation without a SciPy dependency, with MIDRANKS for ties.

    The predictor DOES tie: two files share quiescent fraction 19/42 and the two
    Ra=1e10 files share 4/42. An earlier version of this function asserted "no ties" and
    used argsort positions, which assigns tied values arbitrary distinct ranks and silently
    reports a coefficient that depends on file ordering rather than on the data. Midranks
    are the standard correction and reproduce scipy.stats.spearmanr exactly on this sample.
    """
    def rank(v):
        v = np.asarray(v, float)
        order = np.argsort(v, kind="mergesort")
        r = np.empty(len(v), float)
        r[order] = np.arange(len(v), dtype=float)
        # Average the ranks within each group of equal values.
        for val in np.unique(v):
            tied = np.flatnonzero(v == val)
            if len(tied) > 1:
                r[tied] = r[tied].mean()
        return r
    return _pearson(rank(x), rank(y))


def _betacf(a, b, x, itmax=200, eps=3e-16):
    """Continued fraction for the incomplete beta (Lentz's method)."""
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c, d = 1.0, 1.0 - qab * x / qap
    if abs(d) < 1e-300:
        d = 1e-300
    d, h = 1.0 / d, 1.0 / d
    for m in range(1, itmax + 1):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        c = 1.0 + aa / c
        if abs(d) < 1e-300:
            d = 1e-300
        if abs(c) < 1e-300:
            c = 1e-300
        d = 1.0 / d
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        c = 1.0 + aa / c
        if abs(d) < 1e-300:
            d = 1e-300
        if abs(c) < 1e-300:
            c = 1e-300
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < eps:
            break
    return h


def _betainc(a, b, x):
    """Regularized incomplete beta I_x(a,b). NumPy-only: the harness ships no SciPy."""
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    lbeta = math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
    front = math.exp(lbeta + a * math.log(x) + b * math.log1p(-x))
    if x < (a + 1.0) / (a + b + 2.0):
        return front * _betacf(a, b, x) / a
    return 1.0 - front * _betacf(b, a, 1.0 - x) / b


def _corr_pvalue(r, n):
    """Two-sided p for a correlation coefficient, via the usual t transform.

    Reported because the sample is TEN strata: a coefficient of 0.57 on n=10 is not
    distinguishable from zero at conventional levels, and the paper says so rather than
    quoting the coefficient bare. Validated against scipy.stats.spearmanr/pearsonr.

    Naming: the emitted JSON key is `calibration_window_6_12` and is kept for stability of
    the frozen records and of numbers.json paths, but the manuscript deliberately calls this
    an EXPLORATORY MECHANISM CHECK, not a calibration -- the separation is evaluated on the
    same ten strata that present it and estimates no screen-to-outcome relation.
    """
    if n <= 2 or abs(r) >= 1.0:
        return float("nan")
    df = n - 2
    t2 = r * r * df / (1.0 - r * r)
    return float(_betainc(0.5 * df, 0.5, df / (df + t2)))


def stratified(rows):
    """The Ra x Pr pass: does the mechanism hold where the mechanism predicts it weakens?

    The depth pass covers one corner of the parameter grid (Prandtl 1, Ra <= 1e8), which is
    where the quiescent phase is longest and the artifact therefore easiest to find. Testing
    only there would be assuming the answer. This pass scores the same published rollout
    windows on ten files spanning Ra 1e6-1e10 and Pr 0.1-10, for the two baselines it was run
    for, and reports the cells WITH the data-only quiescent fraction of each file -- so a
    reader can see both where the headline crosses the same censored threshold broadly and
    where it stops crossing.
    """
    files = sorted({r["file"] for r in rows})
    models = sorted({r["model"] for r in rows})
    per_file = {}
    for fn in files:
        sub = [r for r in rows if r["file"] == fn]
        rec = quiescent_fraction(rows, fn)
        rec["trajectories"] = sorted({r["trajectory"] for r in sub})
        rec["cells"] = {
            m: {lab: {fl: rollout_window(sub, m, w, eps)
                      for fl, eps in (("library", 1e-7), ("eps_fix", 1e-5))}
                for lab, w in WINDOWS.items()}
            for m in models}
        per_file[fn] = rec

    summary = {"n_files": len(files), "n_models": len(models),
               "n_trajectories": sum(len(per_file[f]["trajectories"]) for f in files),
               "models": models}
    for lab in WINDOWS:
        cells = [(fn, m) for fn in files for m in models]
        gt10 = [(fn, m) for fn, m in cells if per_file[fn]["cells"][m][lab]["library"] > 10]
        summary[f"window_{lab}"] = {
            "n_cells": len(cells),
            "n_gt10_at_library_floor": len(gt10),
            "n_gt10_at_eps_fix": sum(per_file[fn]["cells"][m][lab]["eps_fix"] > 10
                                     for fn, m in cells),
            "files_with_no_cell_gt10_at_library_floor":
                sorted({fn for fn in files
                        if all(per_file[fn]["cells"][m][lab]["library"] <= 10
                               for m in models)}),
        }
    # Calibration of the data-only predictor against the model-scored cell. Ten strata, two
    # models, one dataset: a probe, not a validated threshold, and the paper says so.
    qf = [per_file[fn]["quiescent_fraction"] for fn in files]
    summary["calibration_window_6_12"] = {
        "predictor": "data-only quiescent fraction of the file's sampled one-step starts",
        "n_strata": len(files),
        "pearson_r": {m: _pearson(qf, [per_file[fn]["cells"][m]["6-12"]["library"]
                                       for fn in files]) for m in models},
        "spearman_r": {m: _spearman(qf, [per_file[fn]["cells"][m]["6-12"]["library"]
                                         for fn in files]) for m in models},
        "pearson_p": {m: _corr_pvalue(_pearson(qf, [per_file[fn]["cells"][m]["6-12"]["library"]
                                                    for fn in files]), len(files))
                      for m in models},
        "spearman_p": {m: _corr_pvalue(_spearman(qf, [per_file[fn]["cells"][m]["6-12"]["library"]
                                                      for fn in files]), len(files))
                       for m in models},
        "max_quiescent_fraction_among_files_with_no_gt10_cell":
            max([per_file[fn]["quiescent_fraction"] for fn in files
                 if all(per_file[fn]["cells"][m]["6-12"]["library"] <= 10 for m in models)],
                default=float("nan")),
        "min_quiescent_fraction_among_files_with_all_gt10_cells":
            min([per_file[fn]["quiescent_fraction"] for fn in files
                 if all(per_file[fn]["cells"][m]["6-12"]["library"] > 10 for m in models)],
                default=float("nan")),
    }
    c = summary["calibration_window_6_12"]
    c["separates_without_overlap"] = bool(
        c["max_quiescent_fraction_among_files_with_no_gt10_cell"]
        < c["min_quiescent_fraction_among_files_with_all_gt10_cells"])
    return {"per_file": per_file, "summary": summary}


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

    # How many of the eight published ">10" rollout cells this audit lands on the same side
    # of (a crossing, not a reproduction of the published estimate), and how many
    # survive the better-conditioned floor.
    #
    # The DENOMINATOR of "seven of eight" is a claim about the PUBLISHED table, so it is read
    # from the frozen per-cell record rather than inferred from this audit's own grid. The
    # dataset-level ">10" flag used by the benchmark-wide map ("any baseline, either window")
    # would NOT entail eight published ">10" cells, and quoting it as if it did was exactly
    # the kind of premise this paper exists to object to.
    _pub_path = os.path.join(HARNESS, "fixtures", "generalization", "well_table3_gt10.json")
    with open(_pub_path, encoding="utf-8") as _f:
        _pub = json.load(_f).get("table3_rollout_rayleigh_benard", {})
    _pub_cells = {(m, lab): _pub.get(f"window_{lab.replace('-', '_')}", {}).get(m)
                  for m in models for lab in WINDOWS}
    # The published ONE-STEP row exists in two places -- the literal above and the frozen
    # record -- and the paper uses both paths (the range quoted in prose comes from the
    # frozen copy; the per-model ratios come from the literal). Uncoupled copies of the same
    # premise can drift without any check firing, so couple them here.
    with open(_pub_path, encoding="utf-8") as _f2:
        _t2 = json.load(_f2).get("table2_onestep_rayleigh_benard", {})
    _t2_map = {"FNO": "fno", "TFNO": "tfno",
               "UNetClassic": "unet", "UNetConvNext": "cnextunet"}
    for _m, _k in _t2_map.items():
        if _k in _t2 and abs(_t2[_k] - PUBLISHED_ONESTEP[_m]) > 1e-12:
            raise AssertionError(
                f"published one-step row disagrees between the frozen record "
                f"({_k}={_t2[_k]}) and PUBLISHED_ONESTEP ({_m}={PUBLISHED_ONESTEP[_m]})")
    out["published_rollout_cells"] = {
        "source": "fixtures/generalization/well_table3_gt10.json"
                  " -> table3_rollout_rayleigh_benard",
        "cells": {f"{m} {lab}": v for (m, lab), v in _pub_cells.items()},
        "n_cells": len(_pub_cells),
        "n_published_gt10": sum(v == ">10" for v in _pub_cells.values()),
    }
    if out["published_rollout_cells"]["n_published_gt10"] != len(_pub_cells):
        raise AssertionError(
            "the paper's 'seven of the eight published >10 cells' premise assumes every "
            "published RB rollout cell is '>10'; the frozen record says otherwise")

    cells = [(m, lab) for m in models for lab in WINDOWS]
    out["gt10_summary"] = {
        "n_cells": len(cells),
        "n_same_side_crossings_at_library_floor":
            sum(out["rollout_windows"][m][lab]["library"] > 10 for m, lab in cells),
        "n_still_gt10_at_eps_fix":
            sum(out["rollout_windows"][m][lab]["eps_fix"] > 10 for m, lab in cells),
        "cell_below_10_at_library_floor":
            [f"{m} {lab}" for m, lab in cells
             if out["rollout_windows"][m][lab]["library"] <= 10],
    }

    # Sampling control: does the one-step estimate move when the sampled files span the whole
    # Ra x Pr grid instead of the Prandtl-1 spread the main pass uses?
    srows, sprov = load("rb_spread")
    if srows:
        smodels = sorted({r["model"] for r in srows})
        out["spread_control"] = {
            "n_files": len({r["file"] for r in srows}),
            "estimates": {m: {"narrow": onestep_estimate(rows, m, 1e-7),
                              "broad": onestep_estimate(srows, m, 1e-7)}
                          for m in smodels},
        }
        # The same ten files, scored on the published ROLLOUT windows rather than only used
        # as a one-step file-mix control. This is the stratified half of the audit.
        out["stratified"] = stratified(srows)

        # Coverage denominators, computed rather than asserted in prose, so the paper's
        # scope sentences cannot drift from what was actually read.
        depth_files = {t[0] for t in trajs}
        strat_files = {r["file"] for r in srows}
        strat_trajs = {(r["file"], r["trajectory"]) for r in srows}
        out["coverage"] = {
            "depth_pass": {"files": len(depth_files), "trajectories": len(trajs),
                           "models": len(models), "trajectories_per_file": 2},
            "stratified_pass": {"files": len(strat_files), "trajectories": len(strat_trajs),
                                "models": len(smodels), "trajectories_per_file": 1},
            "combined": {
                "files": len(depth_files | strat_files),
                "trajectories": len(set(trajs) | strat_trajs),
                "of_test_files": SPLIT_GEOMETRY["test_files"],
                "of_test_trajectories": SPLIT_GEOMETRY["test_trajectories"],
            },
            "models_on_every_covered_file": sorted(set(models) & set(smodels)),
            "models_depth_only": sorted(set(models) - set(smodels)),
        }
        cov = out["coverage"]["combined"]
        cov["file_share"] = cov["files"] / cov["of_test_files"]
        cov["trajectory_share"] = cov["trajectories"] / cov["of_test_trajectories"]

    # Upload chronology of every audited checkpoint (both datasets), loaded from the frozen
    # network fetch. Kept in this aggregate because the reproduction gap cannot be read
    # without it.
    if os.path.exists(CHRONOLOGY_FIXTURE):
        with open(CHRONOLOGY_FIXTURE, encoding="utf-8") as f:
            out["checkpoint_chronology"] = json.load(f)
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
    print(f"  published '>10' rollout cells crossed on the same side at the library floor: "
          f"{g['n_same_side_crossings_at_library_floor']}/{g['n_cells']}"
          f"  (still >10 under eps=1e-5: {g['n_still_gt10_at_eps_fix']})")
