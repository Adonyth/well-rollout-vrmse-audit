#!/usr/bin/env python3
"""One-command verification: regenerate P3's summary.json from the packaged
raw fixtures (public-RT-data-derived per-frame/per-window scalars) and check
that the recomputed P3 values (census, checkpoint, field-split, and
baseline cells) match the frozen paper numbers.json (paper/extracted/numbers.json
in the source repo) to a relative tolerance of 1e-4 (about four significant figures).

This does NOT re-fetch data from the network or re-run model inference — it
re-executes the deterministic aggregation step (aggregate_results.py) that
turns stored raw MSE / target-variance scalars into the 142 enumerated value checks
the P3 paper cites (VRMSE/census cells plus field-split, figure-vs-table, and
spatial-mean-baseline checks). See README.md "Tier 2" for the full cold-start commands
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
import gzip  # noqa: E402
import spatial_mean_baseline  # noqa: E402
from aggregate_results import field_mean_vrmse as _field_mean_vrmse  # noqa: E402

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

    # Generalization map self-consistency (separate from the 142 above; the benchmark-wide
    # denominator-conditioning census is data-only/streamed and its numbers are outside Tier 1,
    # but the assembled categories in Table tab:map must follow deterministically from the frozen
    # per-dataset census outputs and the frozen Well Table-3 status -- assemble_map.py re-derives
    # them and checks every stored field of every row against the packaged MAP.json).
    import os as _os, subprocess as _sub
    _asm = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "scripts", "assemble_map.py")
    if not _os.path.exists(_asm):
        print("FAIL: scripts/assemble_map.py missing -- the benchmark-map self-check cannot run")
        return 1
    _r = _sub.run([sys.executable, _asm], capture_output=True, text=True,
                  cwd=_os.path.dirname(_os.path.abspath(__file__)))
    print(f"[map] {_r.stdout.strip() or _r.stderr.strip()}")
    if _r.returncode != 0:
        print("FAIL: benchmark-map categories disagree with the frozen census + Table-3 status")
        return 1

    # Gate 3 (appendix): the packaged library-equivalence fixture must still say what the
    # appendix claims -- every compared tensor bit-identical, the metric within tolerance,
    # and the window/file counts unchanged. This is a consistency gate over the frozen
    # fixture, not a re-execution: reproducing it from scratch requires network access to
    # HuggingFace and the SDSC mirror plus the pinned checkpoints, which verify.py does not
    # assume. scripts/gate3_recheck.py re-runs the live comparison when those are available.
    _g3 = FIXTURES / "gate3" / "library_equivalence.json"
    if not _g3.exists():
        print("FAIL: fixtures/gate3/library_equivalence.json missing -- Gate 3 cannot be checked")
        return 1
    g3 = json.loads(_g3.read_text())
    g3ref = reference.get("gate3_library_equivalence", {})
    g3_bad = 0
    # Assert the EXPECTED comparison legs per dataset rather than globbing whichever keys
    # happen to exist: a missing leg previously passed vacuously while the manuscript claimed
    # the comparison had been made.
    _TKEY = {"model_input": "rel_diff_input", "target": "rel_diff_target",
             "model_output": "rel_diff_output"}
    for dset, d in sorted(g3["datasets"].items()):
        declared = d.get("tensors_compared", [])
        expected = {_TKEY[t] for t in declared}
        if not expected:
            print(f"  [FAIL] gate3 {dset}: declares no compared tensors")
            g3_bad += 1
        for w in d["windows"]:
            missing = expected - set(w)
            if missing:
                print(f"  [FAIL] gate3 {dset} window {w.get('start')}: missing {sorted(missing)}")
                g3_bad += 1
        ref_declared = g3ref.get("datasets", {}).get(dset, {}).get("tensors_compared")
        if ref_declared is not None and sorted(ref_declared) != sorted(declared):
            print(f"  [FAIL] gate3 {dset}: tensors_compared {declared} vs numbers.json {ref_declared}")
            g3_bad += 1
        tens = [w[k] for w in d["windows"] for k in _TKEY.values() if k in w]
        worst_t = max(tens)
        worst_v = max(w["rel_diff_vrmse"] for w in d["windows"])
        ok = (worst_t == 0.0
              and worst_v < g3["tolerance"]
              and d["n_windows"] == g3ref.get("datasets", {}).get(dset, {}).get("n_windows")
              and d["n_files"] == g3ref.get("datasets", {}).get(dset, {}).get("n_files"))
        print(f"  [{'OK  ' if ok else 'FAIL'}] gate3 {dset}: {d['n_windows']} windows / "
              f"{d['n_files']} files, worst tensor rel diff {worst_t:g}, "
              f"worst VRMSE rel diff {worst_v:.3e}")
        g3_bad += not ok
    if g3["summary"]["n_windows_total"] != g3ref.get("n_windows_total"):
        print("FAIL: gate3 window count disagrees with numbers.json")
        g3_bad += 1
    if g3_bad:
        print("FAIL: Gate 3 library-equivalence fixture no longer supports the appendix claim")
        return 1
    print(f"[gate3] library equivalence OK ({g3['summary']['n_windows_total']} windows, "
          f"all compared tensors bit-identical)")

    # ---- Gate 2: the MULTI-STEP path. Gate 3 drives the library at n_steps_output=1, so
    # it certifies a single forward pass only; every cell in the reported reproduction gap
    # is multi-step. This stage asserts the frozen result of the packaged alignment fixture
    # (scripts/gate2_alignment_fixture.py), which compares the library's own
    # Trainer.rollout_model against this audit's mirror elementwise over a 5-step
    # autoregressive rollout. Re-running it live needs the Tier-2 stack (the_well + torch),
    # so the offline harness checks the frozen copy, as it does for Gate 3.
    _g2_src = FIXTURES / "gate2" / "alignment_fixture.json"
    if not _g2_src.exists():
        print(f"FAIL: Gate 2 fixture missing at {_g2_src}; the appendix claims a packaged "
              f"multi-step equivalence check")
        return 1
    g2 = json.loads(_g2_src.read_text())
    c2 = g2.get("checks", {})
    g2_bad = 0
    # Tolerances are HARD-CODED here, never read from the artifact under test. An earlier
    # version used g2.get("d_tolerance", ...), which let the fixture declare its own passing
    # threshold: a hand-written file claiming a 1e9 tolerance passed while reporting an
    # eight-order-of-magnitude disagreement. An artifact must not set its own bar.
    G2_EXACT_KEYS = ("A_window_alignment_max_abs_diff",
                     "B_channel_order_max_abs_diff",
                     "C_rollout_identity_norm_max_abs_diff")
    G2_D_TOL = 2e-5          # float32 accumulation bound for the Z-scored leg
    G2_MIN_STEPS = 13        # must reach the second disputed rollout window
    for k in G2_EXACT_KEYS:
        v = c2.get(k)
        if not isinstance(v, (int, float)) or not math.isfinite(v) or v != 0.0:
            print(f"  [FAIL] gate2 {k}: {v!r}, expected an exact 0.0")
            g2_bad += 1
    _d = c2.get("D_rollout_zscore_norm_max_abs_diff")
    if not isinstance(_d, (int, float)) or not math.isfinite(_d) or _d < 0.0 or _d > G2_D_TOL:
        print(f"  [FAIL] gate2 Z-scored rollout max abs diff {_d!r} is not a finite "
              f"non-negative value within {G2_D_TOL}")
        g2_bad += 1
    # Geometry must be internally coherent AND actually reach the disputed windows. A
    # fixture with 1 frame, history 999 or a 2-step horizon is not evidence about the
    # cells this paper disputes, whatever it declares.
    _st, _fr = g2.get("rollout_steps"), g2.get("n_frames")
    _hi, _gr, _tr = g2.get("history"), g2.get("grid"), g2.get("n_trajectories")
    geom_ok = (all(isinstance(x, int) for x in (_st, _fr, _hi, _gr, _tr))
               and _st >= G2_MIN_STEPS and _gr > 1 and _tr > 0 and _hi > 0
               and _fr >= _hi + _st)
    if not geom_ok:
        print(f"  [FAIL] gate2 geometry incoherent or too short: steps={_st!r} "
              f"frames={_fr!r} history={_hi!r} grid={_gr!r} traj={_tr!r} "
              f"(need steps>={G2_MIN_STEPS}, frames>=history+steps, grid>1)")
        g2_bad += 1
    # The per-step series and the window count are what let the appendix say "exact at every
    # step" and "all 62 windows" rather than "we sampled a few and wrote down zero".
    _ps = c2.get("identity_per_step_max_abs_diff") if isinstance(c2, dict) else None
    _ps = g2.get("identity_per_step_max_abs_diff", _ps)
    if not (isinstance(_ps, list) and len(_ps) == _st
            and all(isinstance(x, (int, float)) and math.isfinite(x) and x == 0.0 for x in _ps)):
        print(f"  [FAIL] gate2 identity per-step series is not {_st!r} exact zeros: "
              f"{(_ps[:4] if isinstance(_ps, list) else _ps)!r}")
        g2_bad += 1
    _nw = g2.get("n_windows_checked")
    if not (isinstance(_nw, int) and _nw == (_fr - _hi) * _tr if all(
            isinstance(x, int) for x in (_nw, _fr, _hi, _tr)) else False):
        print(f"  [FAIL] gate2 checked {_nw!r} windows, not the "
              f"{(_fr - _hi) * _tr if all(isinstance(x, int) for x in (_fr,_hi,_tr)) else '?'} "
              f"its own geometry implies")
        g2_bad += 1
    _shapes = g2.get("compared_shapes", {})
    _pred, _ref = _shapes.get("rollout_pred"), _shapes.get("rollout_ref")
    # The two shapes come from the same library call, so _pred == _ref is true by
    # construction and carries no information on its own; cross-check them against the
    # geometry the fixture declares, or [31,1,1,1,1] with grid=8 passes.
    _geom_ok = (isinstance(_pred, list) and len(_pred) >= 2
                and _pred[1:1 + 3] == [_gr] * 3) if isinstance(_gr, int) else False
    if not (isinstance(_pred, list) and _pred == _ref and _pred and _pred[0] == _st
            and _geom_ok):
        print(f"  [FAIL] gate2 compared shapes {_pred!r} vs {_ref!r} do not agree with "
              f"a {_st!r}-step rollout")
        g2_bad += 1
    if g2_bad:
        print("FAIL: Gate 2 fixture no longer supports the appendix's multi-step claim")
        return 1
    # ---- window coverage: the abstract's third quantification. Previously unbound -- the
    # leaf could be perturbed and the exit code did not notice. Re-derived here from the
    # packaged audit rows rather than trusted, and the census containment leaves with it,
    # since item 8 claims those are bound and they were not.
    import gzip as _gz, glob as _gl
    _lo, _hi = (_wc0 := reference.get("window_coverage_derived", {})).get("window_frames", [4, 33])
    _fr, _vzfull, _hzfull = [], 0, 0
    for _fp in sorted(_gl.glob(str(FIXTURES / "audit" / "*.json.gz"))):
        with _gz.open(_fp, "rt", encoding="utf-8") as _f:
            _row = json.load(_f)
        _flds = _row["fields"]
        _v = {n: [] for n in _flds if "velocity" in n}
        for _r in _row["rows"]:
            if _lo <= _r["t"] <= _hi:
                for _i, _n in enumerate(_flds):
                    if _n in _v and _i < len(_r["variance_ddof1"]):
                        _v[_n].append(_r["variance_ddof1"][_i])
        _tot = len(next(iter(_v.values())))
        _fr.append(min(sum(1 for x in _v[n] if x <= 1e-7) / _tot
                       for n in _v if n != "velocity_z"))
        _vzfull += all(x <= 1e-5 for x in _v.get("velocity_z", []))
        _hzfull += all(all(x <= 1e-5 for x in _v[n]) for n in _v if n != "velocity_z")
    _wc = reference.get("window_coverage_derived", {})
    _wc_bad = 0
    for _key, _got in (("horizontal_frac_below_epslib_mean", sum(_fr) / len(_fr)),
                       ("horizontal_frac_below_epslib_min", min(_fr)),
                       ("horizontal_frac_below_epslib_max", max(_fr)),
                       ("n_traj_vz_below_epsfix_whole_window", _vzfull),
                       ("n_traj_horizontal_below_epsfix_whole_window", _hzfull),
                       ("n_trajectories", len(_fr))):
        _want = _wc.get(_key)
        if _want is None or abs(float(_got) - float(_want)) > 1e-9:
            print(f"  [FAIL] window_coverage {_key}: re-derived {_got!r} vs table {_want!r}")
            _wc_bad += 1
    if _wc_bad:
        print("FAIL: the window-coverage quantifications no longer match the packaged rows")
        return 1
    print(f"[coverage] window coverage OK ({len(_fr)} trajectories re-derived: horizontal "
          f"below the library floor over {sum(_fr)/len(_fr):.2f} of the window on average, "
          f"vertical below the better-conditioned floor whole-window in {_vzfull})")

    print(f"[gate2] multi-step alignment OK ({g2['rollout_steps']}-step rollout vs the "
          f"library's own Trainer.rollout_model, spanning both disputed windows; "
          f"identity norm exact (0.0), Z-scored max abs diff {_d:.3e})")

    # The exit code must be bound to the RELEASED table, not only to the harness-local copy.
    # verify.py compares against fixtures/numbers_reference.json while its output says
    # "numbers.json"; if the two ever drift, every [OK] above would be checking the wrong
    # file. Assert they are leaf-for-leaf identical whenever the released table is present.
    _released = HERE.parent / "paper" / "extracted" / "numbers.json"
    if _released.exists():
        with open(_released, encoding="utf-8") as _f:
            if json.load(_f) != reference:
                print(f"FAIL: {REFERENCE} has drifted from the released "
                      f"paper/extracted/numbers.json -- the checks above are bound to a "
                      f"stale copy")
                return 1
        print("[reference] fixtures/numbers_reference.json is identical to the released "
              "paper/extracted/numbers.json")

    # Bind the DERIVED blocks and the map block back to the artifacts they were derived
    # from. Adversarial testing showed the exit code bound 393 of numbers.json's ~1332
    # leaves: the printed conditioning-map floor shares, the Gate-3 quoted magnitudes, the
    # and several *_derived groups could all be perturbed in numbers.json AND its frozen twin
    # with this harness still exiting 0, because the reference-identity stage only compares
    # the two copies to each other. What follows re-derives the map and Gate-3 blocks from
    # the packaged sources. NOTE what it does NOT cover: the census containment and extremes
    # leaves are re-derived by the [coverage] stage above only for the window-coverage
    # quantifications the abstract quotes; the remaining census_extremes_derived leaves are
    # still bound only by the reference-identity comparison, and Section 8 lists them among
    # the values outside the exit code.
    map_bad = 0
    _map_ref = dig(reference, "generalization.benchmark_map.datasets") or {}
    _map_src = FIXTURES / "generalization" / "benchmark_map" / "MAP.json"
    if _map_ref and _map_src.exists():
        with open(_map_src, encoding="utf-8") as _f:
            _raw = json.load(_f)
        _map_rows = {r["dataset"]: r for r in _raw} if isinstance(_raw, list) else \
            _raw.get("datasets", {})
        for _ds, _row in sorted(_map_ref.items()):
            _src_row = _map_rows.get(_ds)
            if _src_row is None:
                print(f"  [FAIL] map block: {_ds} present in numbers.json, absent from MAP.json")
                map_bad += 1
                continue
            # floor_share_pct is the printed column; it is a percentage of the stored share.
            def _stored_eq(a, b):
                # numbers.json stores these leaves ROUNDED, and not all to the same
                # precision (floor shares at 4 significant figures, window fractions at 3
                # decimals). Compare the source rounded to whatever precision the stored
                # value actually carries, so the check is exact rather than tolerance-based.
                if not (isinstance(a, (int, float)) and isinstance(b, (int, float))):
                    return a == b
                a, b = float(a), float(b)
                for _n in range(1, 13):
                    if f"{a:.{_n}g}" == f"{a:.12g}":
                        break
                return f"{b:.{_n}g}" == f"{a:.{_n}g}"

            if "floor_share_pct" in _row and "floor_share_epslib" in _src_row:
                if not _stored_eq(_row["floor_share_pct"],
                                  100.0 * _src_row["floor_share_epslib"]):
                    print(f"  [FAIL] map block {_ds}.floor_share_pct: {_row['floor_share_pct']!r} "
                          f"vs 100x MAP.json floor_share_epslib "
                          f"{100.0 * _src_row['floor_share_epslib']!r}")
                    map_bad += 1
            for _k in ("gt10", "susceptibility", "category", "field", "min_var",
                       "frac_le_epslib", "frac_le_epsfix", "n_files"):
                _src_k = "most_susceptible_field" if _k == "field" else _k
                if _k in _row and _src_k in _src_row:
                    _a, _b = _row[_k], _src_row[_src_k]
                    if not _stored_eq(_a, _b):
                        print(f"  [FAIL] map block {_ds}.{_k}: {_a!r} vs MAP.json {_b!r}")
                        map_bad += 1
    _g3_ref = reference.get("gate3_library_equivalence", {})
    _g3_src = FIXTURES / "gate3" / "library_equivalence.json"
    if _g3_ref and _g3_src.exists():
        with open(_g3_src, encoding="utf-8") as _f:
            _g3_raw = json.load(_f)
        # numbers.json flattens the fixture's summary block; map the leaves explicitly.
        for _ref_key, _src_path in (("n_windows_total", "summary.n_windows_total"),
                                    ("worst_rel_diff_vrmse", "summary.worst_rel_diff_vrmse"),
                                    ("worst_rel_diff_vrmse_raw",
                                     "summary.worst_rel_diff_vrmse_raw"),
                                    ("worst_rel_diff_tensor", "summary.worst_rel_diff_tensor"),
                                    ("all_tensors_bit_identical",
                                     "summary.all_tensors_bit_identical"),
                                    ("raw_units_leg_on_all_windows",
                                     "summary.raw_units_leg_on_all_windows"),
                                    ("tolerance", "tolerance")):
            _a, _b = _g3_ref.get(_ref_key), dig(_g3_raw, _src_path)
            _path = _ref_key
            _eq = (f"{float(_a):.4g}" == f"{float(_b):.4g}"
                   if isinstance(_a, (int, float)) and isinstance(_b, (int, float))
                   else _a == _b)
            if _a is not None and _b is not None and not _eq:
                print(f"  [FAIL] gate3 block {_path}: {_a!r} vs fixture {_b!r}")
                map_bad += 1
    if map_bad:
        print(f"FAIL: map / gate3 reference blocks disagree with their sources ({map_bad})")
        return 1
    print("[blocks] map and Gate-3 reference blocks re-derived from their packaged sources")

    # Data provenance, asserted the way checkpoint provenance already is. The manuscript
    # states a bulk-window count, a span, and the three datasets that fall outside it; those
    # are exactly the kind of hand-typed figures that went wrong once here ("fifteen" for a
    # window of fourteen), so they are derived from the shipped record instead.
    _rev_src = FIXTURES / "generalization" / "census_source_revisions.json"
    if not _rev_src.exists():
        print(f"FAIL: {_rev_src} missing -- the data-provenance bound cannot be checked")
        return 1
    with open(_rev_src, encoding="utf-8") as _f:
        _rev = json.load(_f)
    _rev_ds = _rev.get("datasets", {})
    _BULK_DAY = "2025-04-10"
    _bulk = sorted(v["lastModified"] for v in _rev_ds.values()
                   if (v.get("lastModified") or "").startswith(_BULK_DAY))
    _outside = sorted(k for k, v in _rev_ds.items()
                      if not (v.get("lastModified") or "").startswith(_BULK_DAY))
    _span = 0
    if _bulk:
        def _secs(t):
            return int(t[11:13]) * 3600 + int(t[14:16]) * 60 + int(t[17:19])
        _span = _secs(_bulk[-1]) - _secs(_bulk[0])
    _prov_bad = 0
    if len(_rev_ds) != 17:
        print(f"  [FAIL] census revision record covers {len(_rev_ds)} datasets, expected 17")
        _prov_bad += 1
    if len(_bulk) != 14:
        print(f"  [FAIL] {_BULK_DAY} bulk window holds {len(_bulk)} datasets; the manuscript "
              f"says fourteen")
        _prov_bad += 1
    if _span > 20:
        print(f"  [FAIL] bulk window spans {_span}s; the manuscript says under twenty seconds")
        _prov_bad += 1
    if sorted(_outside) != sorted(["MHD_64", "turbulent_radiative_layer_3D",
                                   "euler_multi_quadrants_periodicBC"]):
        print(f"  [FAIL] datasets outside the bulk window are {_outside}; the manuscript "
              f"names MHD_64, turbulent_radiative_layer_3D and "
              f"euler_multi_quadrants_periodicBC")
        _prov_bad += 1
    _cap = _rev.get("_captured_utc", "")
    for _k, _v in _rev_ds.items():
        _lm = _v.get("lastModified")
        if _lm and _cap and _lm >= _cap:
            print(f"  [FAIL] {_k} last modified {_lm} is not before the census capture {_cap}")
            _prov_bad += 1
    if _prov_bad:
        print(f"FAIL: data-provenance record disagrees with the manuscript ({_prov_bad})")
        return 1
    print(f"[provenance] census source revisions OK: {len(_bulk)}/{len(_rev_ds)} datasets in "
          f"the {_BULK_DAY} window ({_span}s span); outside it: {', '.join(_outside)}")

    # UNetClassic floor sweep and definitional-limit one-step means. These carry the
    # abstract's headline ("91.46 ... 1.927", the "factor of about fifty") and the intro's
    # EddyFormer rebuttal, and were previously neither machine-checked NOR listed among the
    # things the harness does not check -- the worst of the two states. Re-derived here from
    # the same packaged per-window scalars the reported cells come from.
    _sweep_ref = reference.get("unetclassic_floor_sweep_derived", {})
    _mean_ref = reference.get("onestep_sampled_mean_by_floor_derived", {})
    sweep_bad = 0
    if not _sweep_ref or not _mean_ref:
        print("FAIL: floor-sweep / sampled-mean reference blocks missing")
        return 1
    _fixture_rows = {}
    for _fp in sorted((FIXTURES / "models").glob("*.json.gz")):
        with gzip.open(_fp, "rt", encoding="utf-8") as _f:
            for _r in json.load(_f)["rows"]:
                _fixture_rows.setdefault((_r["model"], _r.get("mode")), []).append(_r)
    _floors = {"eps_0_definitional": 0.0, "eps_1e9": 1e-9,
               "eps_lib_1e7": 1e-7, "eps_fix_1e5": 1e-5}
    for wname, (lo, hi) in (("window_6_12", (6, 12)), ("window_13_30", (13, 30))):
        sel = [r for r in _fixture_rows.get(("UNetClassic", "rollout"), [])
               if lo <= r["rollout_step"] <= hi]
        by_traj = {}
        for r in sel:
            by_traj.setdefault((r["file"], r["trajectory"]), []).append(r)
        for fname, eps in _floors.items():
            per = [sum(_field_mean_vrmse(r, eps) for r in rs) / len(rs)
                   for rs in by_traj.values()]
            got = sum(per) / len(per)
            want = dig(_sweep_ref, f"{wname}.{fname}")
            if not sig4_equal(got, want):
                print(f"  [FAIL] floor sweep UNetClassic {wname} {fname}: {got!r} vs {want!r}")
                sweep_bad += 1
    for model in ("UNetClassic", "UNetConvNext", "FNO"):
        rows_ = _fixture_rows.get((model, "onestep"), [])
        for fname, eps in (("definitional", 0.0), ("eps_1e9", 1e-9),
                           ("library", 1e-7), ("eps_fix", 1e-5)):
            got = sum(_field_mean_vrmse(r, eps) for r in rows_) / len(rows_)
            want = dig(_mean_ref, f"{model}.{fname}")
            if not sig4_equal(got, want):
                print(f"  [FAIL] onestep sampled mean {model} {fname}: {got!r} vs {want!r}")
                sweep_bad += 1
    # Two further prose-load-bearing groups that were previously neither asserted nor named
    # as unasserted: the FNO flat-extrapolation decline (which is the quantitative basis for
    # the "majority flat-extrapolated" caveat on the one-step table) and the
    # spatial-displacement field-frame count. A reviewer demonstrated that both could be
    # perturbed in numbers.json and its frozen twin with this harness still exiting 0,
    # because the reference-identity stage only compares the two copies to each other.
    _interp_ref = reference.get("onestep_interp_extrapolation_derived", {})
    if _interp_ref:
        _fno = [r for r in _fixture_rows.get(("FNO", "onestep"), [])]
        _by = {}
        for _r in _fno:
            _by.setdefault(_r["input_start"], []).append(_r)
        for _start, _key in ((20, "fno_lib_at_start_20"), (29, "fno_lib_at_start_29")):
            if _key in _interp_ref and _start in _by:
                _vals = [_field_mean_vrmse(_r, 1e-7) for _r in _by[_start]
                         if _r["file"].endswith("At_75.hdf5") and _r["trajectory"] == 1]
                if _vals:
                    _got = sum(_vals) / len(_vals)
                    if f"{_got:.4g}" != f"{float(_interp_ref[_key]):.4g}":
                        print(f"  [FAIL] onestep interp {_key}: {_got!r} vs "
                              f"{_interp_ref[_key]!r}")
                        sweep_bad += 1
    _disp_ref = reference.get("spatial_displacement_frames_derived", {})
    if _disp_ref:
        _sel = [r for r in _fixture_rows.get(("UNetClassic", "rollout"), [])
                if 6 <= r["rollout_step"] <= 12]
        _hit = _tot = 0
        for _r in _sel:
            _var, _mse = _r["target_variance_ddof1"], _r["mse"]
            _pv = _r.get("pred_variance_ddof1") or [0.0] * len(_var)
            for _i in range(1, len(_var)):          # index 0 is density
                _tot += 1
                if math.sqrt(_mse[_i]) > math.sqrt(_var[_i]) + math.sqrt(_pv[_i]):
                    _hit += 1
        for _k, _got in (("velocity_field_frames", _tot),
                         ("velocity_field_frames_exceeding", _hit)):
            if _k in _disp_ref and _disp_ref[_k] != _got:
                print(f"  [FAIL] spatial displacement {_k}: {_got!r} vs {_disp_ref[_k]!r}")
                sweep_bad += 1

    _max_ref = reference.get("onestep_sampled_max_derived", {})
    for model in ("UNetClassic", "UNetConvNext"):
        rows_ = _fixture_rows.get((model, "onestep"), [])
        if not rows_ or not dig(_max_ref, model):
            continue
        vals_lib = [_field_mean_vrmse(r, 1e-7) for r in rows_]
        vals_fix = [_field_mean_vrmse(r, 1e-5) for r in rows_]
        for key, got in (("max_lib", max(vals_lib)), ("max_fix", max(vals_fix)),
                         ("max_e9", max(_field_mean_vrmse(r, 1e-9) for r in rows_)),
                         ("max_defn", max(_field_mean_vrmse(r, 0.0) for r in rows_)),
                         ("n_sampled_starts", len(rows_)),
                         ("n_above_ten_e9",
                          sum(_field_mean_vrmse(r, 1e-9) > 10 for r in rows_)),
                         ("n_above_ten_defn",
                          sum(_field_mean_vrmse(r, 0.0) > 10 for r in rows_))):
            want = dig(_max_ref, f"{model}.{key}")
            if want is None:
                continue
            # These leaves are STORED rounded to four significant figures by the extractor,
            # so compare at that precision rather than with a 1e-4 relative tolerance.
            ok = (got == want if isinstance(want, int)
                  else f"{got:.4g}" == f"{float(want):.4g}")
            if not ok:
                print(f"  [FAIL] onestep sampled {model} {key}: {got!r} vs {want!r}")
                sweep_bad += 1
    if sweep_bad:
        print(f"FAIL: floor sweep / sampled means disagree with the reference ({sweep_bad})")
        return 1
    print("[sweep] floor sweep + definitional-limit one-step means OK "
          "(8 sweep cells, 12 sampled means, re-derived from the packaged scalars)")

    # Rayleigh-Benard checkpoint audit (second audited dataset, paper section 'generalize'):
    # re-derive the cited cells from the packaged per-window scalars and compare with
    # numbers.json, rather than trusting the stored aggregate.
    _rb = SCRIPTS / "rb_checkpoint_audit.py"
    if not _rb.exists():
        print("FAIL: scripts/rb_checkpoint_audit.py missing -- the RB audit cannot be re-derived")
        return 1
    sys.path.insert(0, str(SCRIPTS))
    import importlib
    rbmod = importlib.import_module("rb_checkpoint_audit")
    rb_now = rbmod.build()
    rb_ref = reference.get("rb_checkpoint_audit", {})
    rb_bad = 0
    for model in sorted(rb_now["rollout_windows"]):
        for win in ("6-12", "13-30"):
            for floor in ("library", "eps_fix"):
                got = rb_now["rollout_windows"][model][win][floor]
                want = dig(rb_ref, f"rollout_windows.{model}.{win}.{floor}")
                if not sig4_equal(got, want):
                    print(f"  [FAIL] rb rollout {model} {win} {floor}: {got!r} vs {want!r}")
                    rb_bad += 1
        for key in ("library", "ratio_library_over_published"):
            got = rb_now["onestep"][model][key]
            want = dig(rb_ref, f"onestep.{model}.{key}")
            if not sig4_equal(got, want):
                print(f"  [FAIL] rb onestep {model} {key}: {got!r} vs {want!r}")
                rb_bad += 1
        # both floors AND both weightings: the span-weighted figures are the ones the paper
        # quotes, the unweighted ones are reported alongside them, and a single-floor or
        # single-weighting assertion is exactly what let an unlabelled ratio reach the abstract
        for floor in ("library", "eps_fix", "library_unweighted", "eps_fix_unweighted"):
            for key in ("quiescent_mean", "developed_mean", "ratio"):
                got = rb_now["quiescent_developed"][model][floor][key]
                want = dig(rb_ref, f"quiescent_developed.{model}.{floor}.{key}")
                if not sig4_equal(got, want):
                    print(f"  [FAIL] rb quiescent/developed {model} {floor}.{key}: "
                          f"{got!r} vs {want!r}")
                    rb_bad += 1
    # The published-cell denominator behind "seven of eight": asserted against the frozen
    # per-cell record, so the premise is traceable rather than quoted from prose.
    pub_now, pub_ref = rb_now.get("published_rollout_cells", {}), rb_ref.get(
        "published_rollout_cells", {})
    for key in ("n_cells", "n_published_gt10", "cells"):
        if pub_now.get(key) != pub_ref.get(key):
            print(f"  [FAIL] rb published_rollout_cells {key}: "
                  f"{pub_now.get(key)!r} vs {pub_ref.get(key)!r}")
            rb_bad += 1
    if pub_now.get("n_published_gt10") != 8:
        print("  [FAIL] the 'seven of eight published >10 cells' premise requires 8 "
              f"published '>10' cells; frozen record has {pub_now.get('n_published_gt10')!r}")
        rb_bad += 1

    g10, g10ref = rb_now["gt10_summary"], rb_ref.get("gt10_summary", {})
    for key in ("n_cells", "n_same_side_crossings_at_library_floor", "n_still_gt10_at_eps_fix"):
        if g10[key] != g10ref.get(key):
            print(f"  [FAIL] rb gt10_summary {key}: {g10[key]!r} vs {g10ref.get(key)!r}")
            rb_bad += 1

    # Stratified pass (Ra x Pr): the cells the paper's generalization claim rests on, the
    # coverage denominators its scope sentences quote, and the calibration of the data-only
    # predictor. Asserted here because every one of those is a claim a reader can check.
    st_now = rb_now.get("stratified", {}).get("summary", {})
    st_ref = dig(rb_ref, "stratified.summary") or {}
    if not st_now:
        print("FAIL: rb stratified pass missing -- the Ra x Pr claim cannot be re-derived")
        return 1
    for key in ("n_files", "n_models", "n_trajectories"):
        if st_now[key] != st_ref.get(key):
            print(f"  [FAIL] rb stratified {key}: {st_now[key]!r} vs {st_ref.get(key)!r}")
            rb_bad += 1
    for win in ("6-12", "13-30"):
        for key in ("n_cells", "n_gt10_at_library_floor", "n_gt10_at_eps_fix",
                    "files_with_no_cell_gt10_at_library_floor"):
            got, want = st_now[f"window_{win}"][key], dig(st_ref, f"window_{win}.{key}")
            if got != want:
                print(f"  [FAIL] rb stratified window {win} {key}: {got!r} vs {want!r}")
                rb_bad += 1
    # Every PRINTED cell of the stratified table, not just its summary counts. Asserting
    # only the counts would leave the 40 rollout cells and 10 quiescent shares the paper
    # prints unchecked, while the paper claims the harness re-derives its cited cells.
    pf_now = dig(rb_now, "stratified.per_file") or {}
    pf_ref = dig(rb_ref, "stratified.per_file") or {}
    if sorted(pf_now) != sorted(pf_ref):
        print(f"  [FAIL] rb stratified per-file set: {sorted(pf_now)} vs {sorted(pf_ref)}")
        rb_bad += 1
    # NOTE: the keys are file names and contain dots, so they cannot go through dig()'s
    # dotted-path lookup -- index the reference dict directly.
    for fn in sorted(pf_now):
        ref_fn = pf_ref.get(fn, {})
        for key in ("quiescent_fraction", "min_field_variance", "floor_share_epslib"):
            if not sig4_equal(pf_now[fn][key], ref_fn.get(key)):
                print(f"  [FAIL] rb stratified {fn} {key}: {pf_now[fn][key]!r} vs "
                      f"{ref_fn.get(key)!r}")
                rb_bad += 1
        for model in sorted(pf_now[fn]["cells"]):
            for win in ("6-12", "13-30"):
                for floor in ("library", "eps_fix"):
                    got = pf_now[fn]["cells"][model][win][floor]
                    want = dig(ref_fn.get("cells", {}), f"{model}.{win}.{floor}")
                    if not sig4_equal(got, want):
                        print(f"  [FAIL] rb stratified {fn} {model} {win} {floor}: "
                              f"{got!r} vs {want!r}")
                        rb_bad += 1
    cal_now, cal_ref = st_now["calibration_window_6_12"], st_ref.get(
        "calibration_window_6_12", {})
    for key in ("n_strata", "separates_without_overlap"):
        if cal_now[key] != cal_ref.get(key):
            print(f"  [FAIL] rb mechanism-check {key}: {cal_now[key]!r} vs {cal_ref.get(key)!r}")
            rb_bad += 1
    for stat in ("pearson_r", "spearman_r", "pearson_p", "spearman_p"):
        for model in sorted(cal_now[stat]):
            if not sig4_equal(cal_now[stat][model], dig(cal_ref, f"{stat}.{model}")):
                print(f"  [FAIL] rb mechanism-check {stat} {model}: {cal_now[stat][model]!r} "
                      f"vs {dig(cal_ref, f'{stat}.{model}')!r}")
                rb_bad += 1
    for key in ("max_quiescent_fraction_among_files_with_no_gt10_cell",
                "min_quiescent_fraction_among_files_with_all_gt10_cells"):
        if not sig4_equal(cal_now[key], cal_ref.get(key)):
            print(f"  [FAIL] rb mechanism-check {key}: {cal_now[key]!r} vs {cal_ref.get(key)!r}")
            rb_bad += 1
    cov_now, cov_ref = rb_now.get("coverage", {}), rb_ref.get("coverage", {})
    for path in ("combined.files", "combined.trajectories", "combined.of_test_files",
                 "combined.of_test_trajectories", "depth_pass.files",
                 "depth_pass.trajectories", "depth_pass.models", "stratified_pass.files",
                 "stratified_pass.trajectories", "stratified_pass.models",
                 "models_depth_only", "models_on_every_covered_file"):
        got, want = dig(cov_now, path), dig(cov_ref, path)
        if got != want:
            print(f"  [FAIL] rb coverage {path}: {got!r} vs {want!r}")
            rb_bad += 1

    # Checkpoint chronology: a provenance fact about our own audited artifacts, so it is
    # checked like a number, not trusted as prose.
    chron_now = rb_now.get("checkpoint_chronology", {})
    chron_ref = rb_ref.get("checkpoint_chronology", {})
    if not chron_now:
        print("FAIL: checkpoint chronology fixture missing -- "
              "fixtures/provenance/checkpoint_chronology.json")
        return 1
    for path in ("benchmark_arxiv_v1", "quoted_tables_version", "quoted_tables_version_date",
                 "summary.n_repos", "summary.all_initial_commits_same_day",
                 "summary.any_revised_after_upload", "summary.initial_commit_dates",
                 "summary.min_days_after_benchmark_paper",
                 "summary.max_days_after_benchmark_paper",
                 "summary.min_days_after_quoted_tables_version",
                 "summary.max_days_after_quoted_tables_version"):
        got, want = dig(chron_now, path), dig(chron_ref, path)
        if got != want:
            print(f"  [FAIL] checkpoint chronology {path}: {got!r} vs {want!r}")
            rb_bad += 1
    if len(chron_now.get("repos", {})) != 8:
        print(f"  [FAIL] checkpoint chronology covers {len(chron_now.get('repos', {}))} "
              "repos, expected all 8 audited checkpoints")
        rb_bad += 1
    if rb_bad:
        print(f"FAIL: Rayleigh-Benard checkpoint audit disagrees with numbers.json ({rb_bad})")
        return 1
    _c, _s = rb_now["coverage"]["combined"], st_now["window_6-12"]
    print(f"[rb] stratified pass OK: {_s['n_gt10_at_library_floor']}/{_s['n_cells']} "
          f"Ra x Pr cells >10 at the library floor (window 6-12), "
          f"{_s['n_gt10_at_eps_fix']} still >10 under the better-conditioned floor; "
          f"predictor separates without overlap: "
          f"{st_now['calibration_window_6_12']['separates_without_overlap']}")
    print(f"[rb] coverage OK: {_c['files']}/{_c['of_test_files']} test files, "
          f"{_c['trajectories']}/{_c['of_test_trajectories']} test trajectories")
    print(f"[chronology] OK: {chron_now['summary']['n_repos']} audited checkpoints, all "
          f"committed {chron_now['summary']['initial_commit_dates']}, "
          f"+{chron_now['summary']['min_days_after_quoted_tables_version']}d after the "
          f"quoted tables' arXiv {chron_now['quoted_tables_version']}")
    print(f"[rb] checkpoint audit OK: {rb_now['protocol']['n_rows']} rows re-derived, "
          f"{g10['n_same_side_crossings_at_library_floor']}/{g10['n_cells']} published '>10' rollout "
          f"cells crossed on the same side at the library floor, {g10['n_still_gt10_at_eps_fix']} still "
          f">10 under the better-conditioned floor")

    print("PASS: repro-harness regenerates the frozen P3 values to a relative tolerance of 1e-4 (~4 sig figs).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
