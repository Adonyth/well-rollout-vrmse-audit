#!/usr/bin/env python3
"""One-command verification: regenerate P3's summary.json from the packaged
raw fixtures (public-RT-data-derived per-frame/per-window scalars) and check
that the recomputed UNetClassic census numbers match the frozen paper
numbers.json (paper/extracted/numbers.json in the source repo) to a
relative tolerance of 1e-4 (about four significant figures).

This does NOT re-fetch data from the network or re-run model inference — it
re-executes the deterministic aggregation step (aggregate_results.py) that
turns stored raw MSE / target-variance scalars into the 142 enumerated VRMSE values
the P3 paper cites. See README.md "Tier 2" for the full cold-start commands
that regenerate the raw fixtures themselves from public Well HTTP data and
public Hugging Face checkpoints.

Usage:
    python3 verify.py
Exit code 0 = all checked numbers match within tolerance; 1 = mismatch.
"""
from __future__ import annotations

import json
import math
import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).parent
SCRIPTS = HERE / "scripts"
FIXTURES = HERE / "fixtures"
REFERENCE = FIXTURES / "numbers_reference.json"

sys.path.insert(0, str(SCRIPTS))
import spatial_mean_baseline  # noqa: E402

# (dotted path into recomputed summary.json, dotted path into numbers_reference.json)
# UNetClassic: the paper's headline one-step/rollout/issue-78 cells (full
# ten-trajectory coverage), covering both the library-floor rollout cells and
# the well-conditioned onestep split (the ">10" numbers) and the collapse
# numbers (the documented-floor eps5 and density-only cells) that are the
# paper's actual claim, plus the ill-conditioned onestep split and the
# issue-78 window_13_30 counterfactual -- so the load-bearing "collapse"
# numbers are machine-checked, not just documented.
CHECKS = [
    ("models.UNetClassic.onestep.interp115_mean_lib", "models.UNetClassic.onestep.interp115_mean_lib"),
    ("models.UNetClassic.onestep.interp115_mean_eps5", "models.UNetClassic.onestep.interp115_mean_eps5"),
    ("models.UNetClassic.onestep.subset_mean_lib", "models.UNetClassic.onestep.subset_mean_lib"),
    ("models.UNetClassic.onestep.well_conditioned_windows_mean_lib", "models.UNetClassic.onestep.well_conditioned_windows_mean_lib"),
    ("models.UNetClassic.onestep.n_well_conditioned", "models.UNetClassic.onestep.n_well_conditioned"),
    ("models.UNetClassic.onestep.ill_conditioned_windows_mean_lib", "models.UNetClassic.onestep.ill_conditioned_windows_mean_lib"),
    ("models.UNetClassic.onestep.n_ill_conditioned", "models.UNetClassic.onestep.n_ill_conditioned"),
    ("models.UNetClassic.rollout.window_6_12.lib", "models.UNetClassic.rollout_6_12.lib"),
    ("models.UNetClassic.rollout.window_6_12.eps5", "models.UNetClassic.rollout_6_12.eps5"),
    ("models.UNetClassic.rollout.window_6_12.density_only_lib", "models.UNetClassic.rollout_6_12.density_only_lib"),
    ("models.UNetClassic.rollout.window_13_30.lib", "models.UNetClassic.rollout_13_30.lib"),
    ("models.UNetClassic.rollout.window_13_30.eps5", "models.UNetClassic.rollout_13_30.eps5"),
    ("models.UNetClassic.rollout.window_13_30.density_only_lib", "models.UNetClassic.rollout_13_30.density_only_lib"),
    ("models.UNetClassic.rollout.issue78.last_batch_window_6_12", "models.UNetClassic.issue78.last_batch_window_6_12"),
    ("models.UNetClassic.rollout.issue78.all_batch_window_6_12", "models.UNetClassic.issue78.all_batch_window_6_12"),
    ("models.UNetClassic.rollout.issue78.last_batch_window_13_30", "models.UNetClassic.issue78.last_batch_window_13_30"),
    ("models.UNetClassic.rollout.issue78.all_batch_window_13_30", "models.UNetClassic.issue78.all_batch_window_13_30"),
    # FNO: the two-trajectory divergence-diagnostic cells cited in the abstract /
    # sec7 coverage-tier footnote (215.8/39.28 one-step; the two rollout-window
    # cells, both finite before FNO's per-step curve turns non-finite; the
    # density-only 6-12 cell that Table~tab:rollout reports to show FNO's ">10"
    # survives even the always-well-conditioned field).
    ("models.FNO.onestep.interp115_mean_lib", "models.FNO.onestep.interp115_mean_lib"),
    ("models.FNO.onestep.interp115_mean_eps5", "models.FNO.onestep.interp115_mean_eps5"),
    ("models.FNO.rollout.window_6_12.lib", "models.FNO.rollout_6_12.lib"),
    ("models.FNO.rollout.window_6_12.eps5", "models.FNO.rollout_6_12.eps5"),
    ("models.FNO.rollout.window_6_12.density_only_lib", "models.FNO.rollout_6_12.density_only_lib"),
    # UNetConvNext: the single-trajectory qualitative-replication cells cited in
    # the abstract / sec7 coverage-tier footnote (1.796/0.4572 one-step; both
    # rollout windows, which are finite for this checkpoint unlike FNO's), plus
    # the same eps5/density_only/ill-conditioned/issue78-window_13_30 cells added
    # for UNetClassic above.
    ("models.UNetConvNext.onestep.interp115_mean_lib", "models.UNetConvNext.onestep.interp115_mean_lib"),
    ("models.UNetConvNext.onestep.interp115_mean_eps5", "models.UNetConvNext.onestep.interp115_mean_eps5"),
    ("models.UNetConvNext.onestep.ill_conditioned_windows_mean_lib", "models.UNetConvNext.onestep.ill_conditioned_windows_mean_lib"),
    ("models.UNetConvNext.onestep.n_ill_conditioned", "models.UNetConvNext.onestep.n_ill_conditioned"),
    ("models.UNetConvNext.rollout.window_6_12.lib", "models.UNetConvNext.rollout_6_12.lib"),
    ("models.UNetConvNext.rollout.window_6_12.eps5", "models.UNetConvNext.rollout_6_12.eps5"),
    ("models.UNetConvNext.rollout.window_6_12.density_only_lib", "models.UNetConvNext.rollout_6_12.density_only_lib"),
    ("models.UNetConvNext.rollout.window_13_30.lib", "models.UNetConvNext.rollout_13_30.lib"),
    ("models.UNetConvNext.rollout.window_13_30.eps5", "models.UNetConvNext.rollout_13_30.eps5"),
    ("models.UNetConvNext.rollout.window_13_30.density_only_lib", "models.UNetConvNext.rollout_13_30.density_only_lib"),
    ("models.UNetConvNext.rollout.issue78.last_batch_window_13_30", "models.UNetConvNext.issue78.last_batch_window_13_30"),
    ("models.UNetConvNext.rollout.issue78.all_batch_window_13_30", "models.UNetConvNext.issue78.all_batch_window_13_30"),
]

# Table~\ref{tab:fieldsplit}: all 16 printed cells --
# 2 models (UNetClassic, UNetConvNext) x 2 windows (6-12, 13-30) x 4 columns
# (density_only_lib, density_only_eps5, velocity_only_lib, velocity_only_eps5).
# Machine-checked against numbers.json, and self-contained (it duplicates 4
# cells -- the density_only_lib column -- that are already covered
# incidentally by the general CHECKS list above; kept here too so the
# fieldsplit table has one complete, independently-readable group of exactly
# 16 rather than being split across two lists).
FIELDSPLIT_CHECKS = [
    (f"models.{model}.rollout.{window_key}.{col}", f"models.{model}.{ref_window_key}.{col}")
    for model in ("UNetClassic", "UNetConvNext")
    for window_key, ref_window_key in (("window_6_12", "rollout_6_12"), ("window_13_30", "rollout_13_30"))
    for col in ("density_only_lib", "density_only_eps5", "velocity_only_lib", "velocity_only_eps5")
]

# Denominator census (Table~\ref{tab:census} in the paper): all five printed
# columns, for all ten trajectories -- the full denominator-census set (50 of the
# 142 checks) referenced by sec7_boundaries.tex item 6, so the assertion set below is sized to
# match it exactly rather than being a token sample. Audit trajectory keys
# contain a literal "." inside the filename (e.g.
# "rayleigh_taylor_instability_At_0625.hdf5:0"), which breaks naive dotted-path
# splitting, so these are checked with an explicit walker instead of dig().
AUDIT_TRAJ_KEYS = [
    "rayleigh_taylor_instability_At_0625.hdf5:0",
    "rayleigh_taylor_instability_At_0625.hdf5:1",
    "rayleigh_taylor_instability_At_125.hdf5:0",
    "rayleigh_taylor_instability_At_125.hdf5:1",
    "rayleigh_taylor_instability_At_25.hdf5:0",
    "rayleigh_taylor_instability_At_25.hdf5:1",
    "rayleigh_taylor_instability_At_50.hdf5:0",
    "rayleigh_taylor_instability_At_50.hdf5:1",
    "rayleigh_taylor_instability_At_75.hdf5:0",
    "rayleigh_taylor_instability_At_75.hdf5:1",
]
AUDIT_FIELD_SHORT = {"velocity_x": "vx", "velocity_y": "vy", "velocity_z": "vz"}

# One cross-check per rollout window. Named as a module-level constant, like
# CHECKS/FIELDSPLIT_CHECKS/AUDIT_TRAJ_KEYS above, so extract_numbers.py's
# repro_harness_total_checks_derived block can read
# len(SPATIAL_MEAN_BASELINE_WINDOWS) instead of hand-typing this sub-count.
SPATIAL_MEAN_BASELINE_WINDOWS = ("window_6_12", "window_13_30")


def dig(obj: dict, dotted: str):
    cur = obj
    for part in dotted.split("."):
        cur = cur[part]
    return cur


def audit_census_checks(summary: dict, reference: dict) -> list[tuple[str, object, str, object]]:
    """Every cell of Table~\\ref{tab:census}: per trajectory, var(vx)<=1e-7,
    var(vx)<=1e-5, var(vy)<=1e-5, var(vz)<=1e-5 (frame counts), and min var(density)."""
    rows: list[tuple[str, object, str, object]] = []
    for tk in AUDIT_TRAJ_KEYS:
        s_entry = summary["audit"][tk]
        r_entry = reference["audit_T5"][tk]
        rows.append((
            f"audit[{tk}].velocity_x.n_frames_var_le_1e-7", s_entry["velocity_x"]["n_frames_var_le_1e-7"],
            f"audit_T5[{tk}].vx_n_frames_var_le_1e7", r_entry["vx_n_frames_var_le_1e7"],
        ))
        rows.append((
            f"audit[{tk}].velocity_x.n_frames_var_le_1e-5", s_entry["velocity_x"]["n_frames_var_le_1e-5"],
            f"audit_T5[{tk}].vx_n_frames_var_le_1e5", r_entry["vx_n_frames_var_le_1e5"],
        ))
        rows.append((
            f"audit[{tk}].velocity_y.n_frames_var_le_1e-5", s_entry["velocity_y"]["n_frames_var_le_1e-5"],
            f"audit_T5[{tk}].vy_n_frames_var_le_1e5", r_entry["vy_n_frames_var_le_1e5"],
        ))
        rows.append((
            f"audit[{tk}].velocity_z.n_frames_var_le_1e-5", s_entry["velocity_z"]["n_frames_var_le_1e-5"],
            f"audit_T5[{tk}].vz_n_frames_var_le_1e5", r_entry["vz_n_frames_var_le_1e5"],
        ))
        rows.append((
            f"audit[{tk}].density.var_min", s_entry["density"]["var_min"],
            f"audit_T5[{tk}].density_var_min", r_entry["density_var_min"],
        ))
    return rows


def figure_table_consistency_checks(summary: dict) -> list[tuple[str, object, str, object]]:
    """Consistency check: paper/figs/make_figures.py's census_heatmap()
    (Figure~\\ref{fig:census}) must plot the same quantity Table~\\ref{tab:census}
    prints -- n_frames_var_le_1e-N, a frame COUNT -- not the sibling key last_t_var_le_1e-N
    (the 0-indexed position of the last degenerate frame), which is exactly one less in every
    trajectory. This harness does not import paper/figs/make_figures.py (repro-harness ships
    standalone and only re-executes the numeric aggregation, not figure rendering), so the
    guard here is the underlying data invariant the figure depends on: for all 10
    trajectories and the figure's 4 printed columns (vx<=1e-7, vx<=1e-5, vy<=1e-5, vz<=1e-5),
    n_frames_var_le_1e-N must equal last_t_var_le_1e-N + 1. A violation here means either the
    degenerate-region-starts-at-frame-0 contiguity this check assumes (and Table 1's own caption
    states) broke, or the figure and table would disagree."""
    cols = [("velocity_x", "1e-7"), ("velocity_x", "1e-5"),
            ("velocity_y", "1e-5"), ("velocity_z", "1e-5")]
    rows: list[tuple[str, object, str, object]] = []
    for tk in AUDIT_TRAJ_KEYS:
        entry = summary["audit"][tk]
        for fld, eps in cols:
            n_frames = entry[fld][f"n_frames_var_le_{eps}"]
            last_t = entry[fld][f"last_t_var_le_{eps}"]
            derived = None if last_t is None else last_t + 1
            rows.append((
                f"figure_vs_table[{tk}].{fld}.n_frames_var_le_{eps}", n_frames,
                f"derived_from_last_t[{tk}].{fld}.last_t_var_le_{eps}+1", derived,
            ))
    return rows


def spatial_mean_baseline_checks(reference: dict) -> list[tuple[str, object, str, object]]:
    """Cross-validates scripts/spatial_mean_baseline.py's
    computation (which derives the meanpred baseline directly from raw per-row
    target_variance_ddof1 in fixtures/models/*.json.gz) against
    paper/extracted/numbers.json's unetclassic_spatial_mean_baseline_derived
    block (which derives the same quantity indirectly, by solving for that same
    variance from the already-aggregated per_field_lib/per_field_eps5 curves in
    results/summary.json). Two independent code paths over the same underlying
    data; agreement here is a genuine cross-check, not a self-comparison."""
    out = spatial_mean_baseline.compute(results_dir=FIXTURES, model="UNetClassic")
    rows: list[tuple[str, object, str, object]] = []
    for wname in SPATIAL_MEAN_BASELINE_WINDOWS:
        got = out[wname]["baseline_vrmse"]
        want = dig(reference, f"unetclassic_spatial_mean_baseline_derived.{wname}.baseline_vrmse")
        rows.append((
            f"spatial_mean_baseline.UNetClassic.{wname}.baseline_vrmse", got,
            f"numbers.json unetclassic_spatial_mean_baseline_derived.{wname}.baseline_vrmse", want,
        ))
    return rows


def sig4_equal(a: float, b: float) -> bool:
    if isinstance(a, int) and isinstance(b, int):
        return a == b
    if isinstance(a, float) and isinstance(b, float) and math.isnan(a) and math.isnan(b):
        return True
    if a == b:
        return True
    if a == 0 or b == 0:
        return abs(a - b) < 1e-12
    return math.isclose(a, b, rel_tol=1e-4)


def main() -> int:
    env = dict(os.environ)
    env["P3_RESULTS_DIR"] = str(FIXTURES)
    print(f"[1/3] Recomputing summary.json from packaged fixtures ({FIXTURES})...")
    proc = subprocess.run(
        [sys.executable, str(SCRIPTS / "aggregate_results.py")],
        env=env, capture_output=True, text=True,
    )
    print(proc.stdout)
    if proc.returncode != 0:
        print(proc.stderr, file=sys.stderr)
        print("FAIL: aggregate_results.py exited nonzero")
        return 1

    summary = json.loads((FIXTURES / "summary.json").read_text())
    reference = json.loads(REFERENCE.read_text())

    print("[2/3] Checking recomputed values against paper/extracted/numbers.json (rel tol 1e-4, ~4 sig figs)...")
    n_ok, n_bad = 0, 0
    for summary_path, ref_path in CHECKS:
        got = dig(summary, summary_path)
        want = dig(reference, ref_path)
        ok = sig4_equal(got, want)
        status = "OK  " if ok else "FAIL"
        print(f"  [{status}] {summary_path} = {got!r}  vs numbers.json {ref_path} = {want!r}")
        n_ok += ok
        n_bad += not ok

    print("[2a/3] Checking Table~fieldsplit's 16 printed cells (density/velocity x lib/eps5 x "
          "2 models x 2 windows)...")
    for summary_path, ref_path in FIELDSPLIT_CHECKS:
        got = dig(summary, summary_path)
        want = dig(reference, ref_path)
        ok = sig4_equal(got, want)
        status = "OK  " if ok else "FAIL"
        print(f"  [{status}] {summary_path} = {got!r}  vs numbers.json {ref_path} = {want!r}")
        n_ok += ok
        n_bad += not ok

    print("[2b/3] Checking every cell of the denominator census (Table~census, 10 trajectories x 5 columns)...")
    for summary_path, got, ref_path, want in audit_census_checks(summary, reference):
        ok = sig4_equal(got, want)
        status = "OK  " if ok else "FAIL"
        print(f"  [{status}] {summary_path} = {got!r}  vs numbers_reference.json {ref_path} = {want!r}")
        n_ok += ok
        n_bad += not ok

    print("[2c/3] Figure-vs-table consistency: Figure~census's 4 columns x 10 trajectories must equal "
          "Table~census's n_frames_var_le_1e-N (not the off-by-one last_t_var_le_1e-N sibling key)...")
    for summary_path, got, ref_path, want in figure_table_consistency_checks(summary):
        ok = sig4_equal(got, want)
        status = "OK  " if ok else "FAIL"
        print(f"  [{status}] {summary_path} = {got!r}  vs {ref_path} = {want!r}")
        n_ok += ok
        n_bad += not ok

    print("[2d/3] Spatial-mean-predictor baseline: the eps=1e-5-floored "
          "UNetClassic rollout scores (1.927 / 1.267) are NOT comparable in quality to predicting "
          "the field mean -- the actual trivial per-frame spatial-mean predictor, scored under the "
          "identical eps=1e-5 floor and the identical window aggregation, scores 0.2699 / 0.3919, "
          "i.e. the model is 7.1x / 3.2x worse than that baseline...")
    for summary_path, got, ref_path, want in spatial_mean_baseline_checks(reference):
        ok = sig4_equal(got, want)
        status = "OK  " if ok else "FAIL"
        print(f"  [{status}] {summary_path} = {got!r}  vs {ref_path} = {want!r}")
        n_ok += ok
        n_bad += not ok

    print(f"[3/3] {n_ok} match, {n_bad} mismatch.")
    if n_bad:
        print("FAIL: reproduction does not match frozen numbers.json")
        return 1
    print("PASS: repro-harness regenerates the frozen P3 census numbers to a relative tolerance of 1e-4 (~4 sig figs).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
