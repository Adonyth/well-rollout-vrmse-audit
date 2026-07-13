"""Aggregate P3-quant RT results into every metric variant + counterfactual.

Inputs:  results/audit/*.json.gz, results/models/*.json.gz
Outputs: results/summary.json (+ markdown tables on stdout)

Metric variants (all derived from stored float64 MSE / target-variance scalars):
  lib    = sqrt(mse / (var_ddof1 + 1e-7))   # published semantics (default eps)
  eps5   = sqrt(mse / (var_ddof1 + 1e-5))   # issue-#75-requested eps, propagated
  noeps  = sqrt(mse / var_ddof1)
  cond   = lib restricted to well-conditioned windows (all-field var > 1e-5)

Aggregations mirror the_well 1.2.0 validation_loop:
  * per sample: mean over fields (paper: "averaged over all physical fields")
  * rollout Table-3 windows: steps 6..12, 13..30 (1-indexed) == temporal_split
    slices [5:12], [12:30] of the 0-indexed per-step curve
  * batch structure: rollout_test_dataloader = batch_size 1, shuffle False,
    lexicographic file order -> last batch = At_75 trajectory 1
"""

from __future__ import annotations

import glob
import gzip
import json
import math
import os
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
# repro-harness note: original always read/wrote HERE/"results". This env var
# lets the harness point the aggregator at the packaged fixtures/ directory
# (or any fresh cold-run output directory) without changing call sites below.
RESULTS_DIR = Path(os.environ.get("P3_RESULTS_DIR", str(HERE / "results")))
FIELDS = ["density", "velocity_x", "velocity_y", "velocity_z"]
EPS_LIB = 1e-7
EPS_REQ = 1e-5
W1 = (6, 12)   # rollout steps, 1-indexed inclusive
W2 = (13, 30)
PAPER = {
    "table2_onestep": {"FNO": ">10", "TFNO": ">10", "UNetClassic": ">10", "UNetConvNext": ">10"},
    "table3_6_12": {"FNO": ">10", "TFNO": "6.72", "UNetClassic": ">10", "UNetConvNext": ">10"},
    "table3_13_30": {"FNO": ">10", "TFNO": ">10", "UNetClassic": "2.84", "UNetConvNext": "7.43"},
    "table5_validation": {"FNO": 0.4013, "TFNO": 0.2251, "UNetClassic": 0.6140, "UNetConvNext": 0.3771},
}


def vr(mse: np.ndarray, var: np.ndarray, eps: float) -> np.ndarray:
    return np.sqrt(mse / (var + eps))


def load_audit() -> list[dict]:
    rows = []
    for path in sorted(glob.glob(str(RESULTS_DIR / "audit" / "*.json.gz"))):
        with gzip.open(path, "rt") as handle:
            payload = json.load(handle)
        for row in payload["rows"]:
            row["file"] = payload["file"]
            row["trajectory"] = payload["trajectory"]
            rows.append(row)
    return rows


def load_models() -> dict[str, list[dict]]:
    out: dict[str, list[dict]] = {}
    for path in sorted(glob.glob(str(RESULTS_DIR / "models" / "*_rayleigh*.json.gz"))):
        with gzip.open(path, "rt") as handle:
            payload = json.load(handle)
        for row in payload["rows"]:
            out.setdefault(row["model"], []).append(row)
    return out


def field_mean_vrmse(row: dict, eps: float | None) -> float:
    mse = np.asarray(row["mse"], dtype=np.float64)
    var = np.asarray(row["target_variance_ddof1"], dtype=np.float64)
    if eps is None:
        return float(np.mean(np.sqrt(mse / var)))
    return float(np.mean(vr(mse, var, eps)))


def per_field_vrmse(row: dict, eps: float) -> np.ndarray:
    mse = np.asarray(row["mse"], dtype=np.float64)
    var = np.asarray(row["target_variance_ddof1"], dtype=np.float64)
    return vr(mse, var, eps)


def well_conditioned(row: dict, threshold: float = EPS_REQ) -> bool:
    return bool(np.all(np.asarray(row["target_variance_ddof1"]) > threshold))


def traj_key(row: dict) -> tuple[str, int]:
    return (row["file"], row["trajectory"])


def onestep_summary(rows: list[dict]) -> dict:
    one = [r for r in rows if r["mode"] == "onestep"]
    trajs = sorted({traj_key(r) for r in one})
    curves = {}
    for tk in trajs:
        sub = sorted([r for r in one if traj_key(r) == tk], key=lambda r: r["input_start"])
        curves[tk] = {
            "starts": [r["input_start"] for r in sub],
            "lib": [field_mean_vrmse(r, EPS_LIB) for r in sub],
            "eps5": [field_mean_vrmse(r, EPS_REQ) for r in sub],
            "per_field_lib": [per_field_vrmse(r, EPS_LIB).tolist() for r in sub],
            "conditioned": [well_conditioned(r) for r in sub],
        }

    def subset_mean(kind: str) -> float:
        return float(np.mean([v for tk in trajs for v in curves[tk][kind]]))

    def interp_mean(kind: str) -> float:
        # per trajectory: linear interpolation of per-start curve onto all 115
        # starts, then global mean (documented estimator, not the exact census)
        vals = []
        for tk in trajs:
            s = np.asarray(curves[tk]["starts"], dtype=float)
            v = np.asarray(curves[tk][kind], dtype=float)
            grid = np.arange(0, 115, dtype=float)
            vals.append(np.interp(grid, s, v))
        return float(np.mean(np.stack(vals)))

    cond_vals = [
        v
        for tk in trajs
        for v, ok in zip(curves[tk]["lib"], curves[tk]["conditioned"])
        if ok
    ]
    uncond_vals = [
        v
        for tk in trajs
        for v, ok in zip(curves[tk]["lib"], curves[tk]["conditioned"])
        if not ok
    ]
    return {
        "n_trajectories": len(trajs),
        "trajectories": [f"{f}:{t}" for f, t in trajs],
        "n_windows": sum(len(curves[tk]["starts"]) for tk in trajs),
        "subset_mean_lib": subset_mean("lib"),
        "subset_mean_eps5": subset_mean("eps5"),
        "interp115_mean_lib": interp_mean("lib"),
        "interp115_mean_eps5": interp_mean("eps5"),
        "well_conditioned_windows_mean_lib": float(np.mean(cond_vals)) if cond_vals else math.nan,
        "n_well_conditioned": len(cond_vals),
        "ill_conditioned_windows_mean_lib": float(np.mean(uncond_vals)) if uncond_vals else math.nan,
        "n_ill_conditioned": len(uncond_vals),
        "curves": {f"{f}:{t}": curves[(f, t)] for f, t in trajs},
    }


def rollout_summary(rows: list[dict], last_batch: tuple[str, int]) -> dict:
    roll = [r for r in rows if r["mode"] == "rollout"]
    trajs = sorted({traj_key(r) for r in roll})
    curves = {}
    for tk in trajs:
        sub = sorted([r for r in roll if traj_key(r) == tk], key=lambda r: r["rollout_step"])
        curves[tk] = {
            "steps": [r["rollout_step"] for r in sub],
            "lib": [field_mean_vrmse(r, EPS_LIB) for r in sub],
            "eps5": [field_mean_vrmse(r, EPS_REQ) for r in sub],
            "per_field_lib": [per_field_vrmse(r, EPS_LIB).tolist() for r in sub],
            "per_field_eps5": [per_field_vrmse(r, EPS_REQ).tolist() for r in sub],
            "conditioned": [well_conditioned(r) for r in sub],
        }

    def window_mean(tk: tuple[str, int], kind: str, w: tuple[int, int]) -> float:
        steps = np.asarray(curves[tk]["steps"])
        vals = np.asarray(curves[tk][kind])
        mask = (steps >= w[0]) & (steps <= w[1])
        return float(vals[mask].mean())

    def window_field_mean(kind: str, w: tuple[int, int], field_idx: list[int]) -> float:
        # mean over trajectories of window means of the per-field VRMSE subset
        per_traj = []
        for tk in trajs:
            steps = np.asarray(curves[tk]["steps"])
            vals = np.asarray(curves[tk][kind])  # [n_steps, F]
            mask = (steps >= w[0]) & (steps <= w[1])
            per_traj.append(vals[mask][:, field_idx].mean())
        return float(np.mean(per_traj))

    def agg(kind: str, w: tuple[int, int]) -> float:
        # library semantics: mean over batches (=trajectories) of window means
        return float(np.mean([window_mean(tk, kind, w) for tk in trajs]))

    def cond_window(w: tuple[int, int]) -> dict:
        vals, n_used, n_all = [], 0, 0
        for tk in trajs:
            steps = np.asarray(curves[tk]["steps"])
            lib = np.asarray(curves[tk]["lib"])
            ok = np.asarray(curves[tk]["conditioned"])
            mask = (steps >= w[0]) & (steps <= w[1])
            n_all += int(mask.sum())
            take = mask & ok
            n_used += int(take.sum())
            vals.extend(lib[take].tolist())
        return {
            "mean_lib_well_conditioned_only": float(np.mean(vals)) if vals else math.nan,
            "n_windows_used": n_used,
            "n_windows_total": n_all,
        }

    divergence = {}
    for tk in trajs:
        sub = sorted(
            [r for r in roll if traj_key(r) == tk], key=lambda r: r["rollout_step"]
        )
        first_bad = None
        for r in sub:
            mse = np.asarray(r["mse"])
            if not np.all(np.isfinite(mse)) or np.any(mse > 1e6):
                first_bad = r["rollout_step"]
                break
        divergence[f"{tk[0]}:{tk[1]}"] = first_bad

    out = {
        "n_trajectories": len(trajs),
        "trajectories": [f"{f}:{t}" for f, t in trajs],
        "first_rollout_step_with_mse_gt_1e6_or_nonfinite": divergence,
        "window_6_12": {
            "lib": agg("lib", W1),
            "eps5": agg("eps5", W1),
            "density_only_lib": window_field_mean("per_field_lib", W1, [0]),
            "density_only_eps5": window_field_mean("per_field_eps5", W1, [0]),
            "velocity_only_lib": window_field_mean("per_field_lib", W1, [1, 2, 3]),
            "velocity_only_eps5": window_field_mean("per_field_eps5", W1, [1, 2, 3]),
            "conditioned": cond_window(W1),
            "per_trajectory_lib": {f"{f}:{t}": window_mean((f, t), "lib", W1) for f, t in trajs},
            "per_trajectory_eps5": {f"{f}:{t}": window_mean((f, t), "eps5", W1) for f, t in trajs},
        },
        "window_13_30": {
            "lib": agg("lib", W2),
            "eps5": agg("eps5", W2),
            "density_only_lib": window_field_mean("per_field_lib", W2, [0]),
            "density_only_eps5": window_field_mean("per_field_eps5", W2, [0]),
            "velocity_only_lib": window_field_mean("per_field_lib", W2, [1, 2, 3]),
            "velocity_only_eps5": window_field_mean("per_field_eps5", W2, [1, 2, 3]),
            "conditioned": cond_window(W2),
            "per_trajectory_lib": {f"{f}:{t}": window_mean((f, t), "lib", W2) for f, t in trajs},
            "per_trajectory_eps5": {f"{f}:{t}": window_mean((f, t), "eps5", W2) for f, t in trajs},
        },
        "curves": {f"{f}:{t}": curves[(f, t)] for f, t in trajs},
    }

    # ---- #78 counterfactual: long_time curve = last batch only ----
    if last_batch in curves:
        lb = curves[last_batch]
        all_lib = np.stack([curves[tk]["lib"] for tk in trajs])
        out["issue78"] = {
            "last_batch_trajectory": f"{last_batch[0]}:{last_batch[1]}",
            "last_batch_curve_lib": lb["lib"],
            "all_batch_mean_curve_lib": all_lib.mean(0).tolist(),
            "curve_min": all_lib.min(0).tolist(),
            "curve_max": all_lib.max(0).tolist(),
            "last_batch_window_6_12": window_mean(last_batch, "lib", W1),
            "all_batch_window_6_12": out["window_6_12"]["lib"],
            "last_batch_window_13_30": window_mean(last_batch, "lib", W2),
            "all_batch_window_13_30": out["window_13_30"]["lib"],
        }
    return out


def audit_summary(audit_rows: list[dict]) -> dict:
    files = sorted({r["file"] for r in audit_rows})
    per_traj = {}
    for f in files:
        for traj in sorted({r["trajectory"] for r in audit_rows if r["file"] == f}):
            sub = sorted(
                [r for r in audit_rows if r["file"] == f and r["trajectory"] == traj],
                key=lambda r: r["t"],
            )
            var = np.asarray([r["variance_ddof1"] for r in sub])  # [T, F]
            t = np.asarray([r["t"] for r in sub])
            entry = {}
            for i, field in enumerate(FIELDS):
                below7 = t[var[:, i] <= EPS_LIB]
                below5 = t[var[:, i] <= EPS_REQ]
                entry[field] = {
                    "n_frames_var_le_1e-7": int((var[:, i] <= EPS_LIB).sum()),
                    "n_frames_var_le_1e-5": int((var[:, i] <= EPS_REQ).sum()),
                    "last_t_var_le_1e-7": int(below7.max()) if below7.size else None,
                    "last_t_var_le_1e-5": int(below5.max()) if below5.size else None,
                    "var_min": float(var[:, i].min()),
                    "var_at_t4": float(var[t == 4, i][0]) if (t == 4).any() else None,
                    "var_at_t33": float(var[t == 33, i][0]) if (t == 33).any() else None,
                    "var_final": float(var[-1, i]),
                }
            # persistence VRMSE curves (Lane 3 regression anchor)
            pers = [
                (
                    r["t"],
                    (
                        np.sqrt(
                            np.asarray(r["persistence_mse"])
                            / (np.asarray(r["variance_ddof1"]) + EPS_LIB)
                        )
                    ).tolist(),
                )
                for r in sub
                if "persistence_mse" in r
            ]
            entry["_persistence_vrmse_lib_by_t"] = pers
            entry["_variance_by_t"] = [(int(tt), vv) for tt, vv in zip(t.tolist(), var.tolist())]
            per_traj[f"{f}:{traj}"] = entry
    return per_traj


def main() -> None:
    audit_rows = load_audit()
    model_rows = load_models()
    last_batch = ("rayleigh_taylor_instability_At_75.hdf5", 1)

    summary: dict = {
        "paper_values": PAPER,
        "eps_lib": EPS_LIB,
        "eps_requested_issue75": EPS_REQ,
        "windows": {"table3_first": W1, "table3_second": W2},
        "last_batch_identity": "At_75 trajectory 1 (lexicographic file sort, batch_size=1, shuffle=False)",
        "audit": audit_summary(audit_rows) if audit_rows else {},
        "models": {},
    }
    for model, rows in model_rows.items():
        summary["models"][model] = {
            "onestep": onestep_summary(rows),
            "rollout": rollout_summary(rows, last_batch),
        }

    out = RESULTS_DIR / "summary.json"
    out.write_text(json.dumps(summary, indent=1) + "\n")
    print(f"wrote {out}")

    # ---- compact human tables ----
    for model, s in summary["models"].items():
        one, roll = s["onestep"], s["rollout"]
        print(f"\n=== {model} ===")
        print(
            f" Table2 analog (one-step): subset lib={one['subset_mean_lib']:.4g} "
            f"eps5={one['subset_mean_eps5']:.4g} interp115 lib={one['interp115_mean_lib']:.4g} "
            f"eps5={one['interp115_mean_eps5']:.4g}  paper={PAPER['table2_onestep'].get(model)}"
        )
        print(
            f"   well-cond lib={one['well_conditioned_windows_mean_lib']:.4g} (n={one['n_well_conditioned']}) "
            f"ill-cond lib={one['ill_conditioned_windows_mean_lib']:.4g} (n={one['n_ill_conditioned']})"
        )
        for wname, w in [("6-12", "window_6_12"), ("13-30", "window_13_30")]:
            ww = roll[w]
            print(
                f" Table3 {wname}: lib={ww['lib']:.4g} eps5={ww['eps5']:.4g} "
                f"cond-only={ww['conditioned']['mean_lib_well_conditioned_only']:.4g} "
                f"({ww['conditioned']['n_windows_used']}/{ww['conditioned']['n_windows_total']}) "
                f"paper={PAPER['table3_' + wname.replace('-', '_')].get(model)}"
            )
        if "issue78" in roll:
            i78 = roll["issue78"]
            print(
                f" #78: last-batch 6-12={i78['last_batch_window_6_12']:.4g} vs all={i78['all_batch_window_6_12']:.4g}; "
                f"13-30 last={i78['last_batch_window_13_30']:.4g} vs all={i78['all_batch_window_13_30']:.4g}"
            )


def emit_markdown() -> None:
    """Emit report-ready markdown tables from results/summary.json."""
    s = json.loads((RESULTS_DIR / "summary.json").read_text())
    models = s["models"]

    def g(x: float | None, digits: int = 4) -> str:
        if x is None or (isinstance(x, float) and math.isnan(x)):
            return "—"
        return f"{x:.{digits}g}"

    print("\n### T1. Paper-table analogs: library metric vs fixed metric\n")
    print("| model | coverage (traj) | one-step lib (T2 analog) | one-step eps5 | paper T2 | 6-12 lib | 6-12 eps5 | 6-12 cond-only | paper 6:12 | 13-30 lib | 13-30 eps5 | 13-30 cond-only | paper 13:30 |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for m, sm in models.items():
        one, roll = sm["onestep"], sm["rollout"]
        w1, w2 = roll["window_6_12"], roll["window_13_30"]
        print(
            f"| {m} | {roll['n_trajectories']}/10 | {g(one['interp115_mean_lib'])} | {g(one['interp115_mean_eps5'])} "
            f"| {s['paper_values']['table2_onestep'].get(m)} "
            f"| {g(w1['lib'])} | {g(w1['eps5'])} | {g(w1['conditioned']['mean_lib_well_conditioned_only'])} ({w1['conditioned']['n_windows_used']}/{w1['conditioned']['n_windows_total']}) "
            f"| {s['paper_values']['table3_6_12'].get(m)} "
            f"| {g(w2['lib'])} | {g(w2['eps5'])} | {g(w2['conditioned']['mean_lib_well_conditioned_only'])} ({w2['conditioned']['n_windows_used']}/{w2['conditioned']['n_windows_total']}) "
            f"| {s['paper_values']['table3_13_30'].get(m)} |"
        )

    print("\n### T1b. Rollout windows split by field group (lib / eps5)\n")
    print("| model | window | density-only lib | density-only eps5 | velocity-only lib | velocity-only eps5 |")
    print("|---|---|---|---|---|---|")
    for m, sm in models.items():
        for wname, w in [("6-12", "window_6_12"), ("13-30", "window_13_30")]:
            ww = sm["rollout"][w]
            print(
                f"| {m} | {wname} | {g(ww['density_only_lib'])} | {g(ww['density_only_eps5'])} "
                f"| {g(ww['velocity_only_lib'])} | {g(ww['velocity_only_eps5'])} |"
            )

    print("\n### T2. #78 counterfactual (long-time curve = last batch = At_75 traj 1)\n")
    print("| model | window | last-batch-only (published plot path) | all-batch mean (correct) | ratio |")
    print("|---|---|---|---|---|")
    for m, sm in models.items():
        if "issue78" not in sm["rollout"]:
            continue
        i78 = sm["rollout"]["issue78"]
        for wname, lb, ab in [
            ("6-12", i78["last_batch_window_6_12"], i78["all_batch_window_6_12"]),
            ("13-30", i78["last_batch_window_13_30"], i78["all_batch_window_13_30"]),
        ]:
            print(f"| {m} | {wname} | {g(lb)} | {g(ab)} | {g(lb / ab, 3)} |")

    print("\n### T3. One-step time decomposition (per-start field-mean VRMSE, mean over covered trajectories)\n")
    for m, sm in models.items():
        one = sm["onestep"]
        starts = sorted({st for c in one["curves"].values() for st in c["starts"]})
        print(f"\n**{m}** (trajectories: {one['n_trajectories']})\n")
        print("| start | lib | eps5 | lib/eps5 |")
        print("|---|---|---|---|")
        for st in starts:
            libs, eps5s = [], []
            for c in one["curves"].values():
                if st in c["starts"]:
                    k = c["starts"].index(st)
                    libs.append(c["lib"][k])
                    eps5s.append(c["eps5"][k])
            lm, em = float(np.mean(libs)), float(np.mean(eps5s))
            print(f"| {st} | {g(lm)} | {g(em)} | {g(lm / em, 3)} |")

    print("\n### T4. Rollout per-step decomposition (field-mean VRMSE, mean over covered trajectories)\n")
    for m, sm in models.items():
        roll = sm["rollout"]
        steps = [1, 2, 3, 6, 9, 12, 16, 20, 25, 30]
        print(f"\n**{m}**\n")
        print("| rollout step | lib | eps5 | lib/eps5 |")
        print("|---|---|---|---|")
        for st in steps:
            libs, eps5s = [], []
            for c in roll["curves"].values():
                if st in c["steps"]:
                    k = c["steps"].index(st)
                    libs.append(c["lib"][k])
                    eps5s.append(c["eps5"][k])
            if not libs:
                continue
            lm, em = float(np.mean(libs)), float(np.mean(eps5s))
            print(f"| {st} | {g(lm)} | {g(em)} | {g(lm / em, 3)} |")

    print("\n### T5. Denominator audit (per trajectory: velocity variance near-zero extent)\n")
    print("| file:traj | vx last t var<=1e-7 | vx last t var<=1e-5 | vy <=1e-5 | vz <=1e-5 | density var min |")
    print("|---|---|---|---|---|---|")
    for tk, a in s["audit"].items():
        vx, vy, vz, de = a["velocity_x"], a["velocity_y"], a["velocity_z"], a["density"]
        print(
            f"| {tk.replace('rayleigh_taylor_instability_', '').replace('.hdf5', '')} "
            f"| {vx['last_t_var_le_1e-7']} | {vx['last_t_var_le_1e-5']} "
            f"| {vy['last_t_var_le_1e-5']} | {vz['last_t_var_le_1e-5']} | {g(de['var_min'], 3)} |"
        )


if __name__ == "__main__":
    import sys as _sys

    if "--markdown" in _sys.argv:
        emit_markdown()
    else:
        main()
