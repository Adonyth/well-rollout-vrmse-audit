#!/usr/bin/env python3
"""Assemble the benchmark-wide denominator-conditioning map from per-dataset censuses.
Crosses census susceptibility (min spatial variance vs library floor 1e-7 and the
better-conditioned 1e-5) with the published >10 rollout status (Well Table 3)."""
import json, glob, os

EPSLIB, EPSFIX = 1e-7, 1e-5

# Published rollout ">10" status from The Well Table 3 (arXiv:2412.00568v2), verified:
#   "all"  = all four baselines >10 (both windows)
#   "some" = some baselines >10
#   "none" = no baseline >10
GT10 = {
    "rayleigh_taylor_instability": "all",
    "rayleigh_benard": "all",
    "shear_flow": "some",
    "active_matter": "some",
    "gray_scott_reaction_diffusion": "some",
    "euler_multi_quadrants_openBC": "some",
    "supernova_explosion_64": "some",
    "turbulent_radiative_layer_3D": "some",
    "viscoelastic_instability": "some",
    "turbulent_radiative_layer_2D": "none",
    "MHD_64": "none",
    "acoustic_scattering_inclusions": "none",
    "helmholtz_staircase": "none",
    "planetswe": "none",
    "turbulence_gravity_cooling": "none",
    "convective_envelope_rsg": "none",
    "post_neutron_star_merger": "none",
}
AUDITED = {"rayleigh_taylor_instability"}  # checkpoint-level audit performed in this paper


def per_dataset(fp):
    d = json.load(open(fp))
    ds = d["dataset"]
    best = {}
    for fn, fd in d.get("files", {}).items():
        if "fields" not in fd:
            continue
        for k, v in fd["fields"].items():
            b = best.setdefault(k, {"min": 1e99, "flib": 0.0, "ffix": 0.0})
            b["min"] = min(b["min"], v["var_min"])
            b["flib"] = max(b["flib"], v["frac_below_epslib"])
            b["ffix"] = max(b["ffix"], v["frac_below_epsfix"])
    if not best:
        return None
    # the most-susceptible field = lowest min variance
    fld = min(best, key=lambda k: best[k]["min"])
    b = best[fld]
    gmin = b["min"]
    if gmin <= EPSLIB:
        suscept = "severe"          # variance reaches the library floor
    elif gmin <= EPSFIX:
        suscept = "susceptible"     # variance below the better-conditioned floor
    else:
        suscept = "healthy"
    return {
        "dataset": ds, "n_files": len(d.get("files", {})),
        "n_total": d.get("n_test_files_total"), "frame_stride": d.get("frame_stride", 1),
        "gt10": GT10.get(ds, "?"),
        "most_susceptible_field": fld, "min_var": gmin,
        "frac_le_epslib": b["flib"], "frac_le_epsfix": b["ffix"],
        "susceptibility": suscept,
    }


def category(r):
    s, g = r["susceptibility"], r["gt10"]
    floored = s in ("severe", "susceptible")
    gt = g in ("all", "some")
    if floored and gt:
        return "triggered_audited" if r["dataset"] in AUDITED else "artifact_suspect"
    if floored and not gt:
        return "latent"
    if not floored and gt:
        return "genuine_failure"
    return "well_conditioned"


rows = [per_dataset(fp) for fp in sorted([f for f in glob.glob(".gate-work/census/*.json") if not f.endswith("MAP.json")])]
rows = [r for r in rows if r]
for r in rows:
    r["category"] = category(r)

order = {"triggered_audited": 0, "artifact_suspect": 1, "severe": 1,
         "latent": 2, "genuine_failure": 3, "well_conditioned": 4}
rows.sort(key=lambda r: (order.get(r["category"], 9), r["min_var"]))

CATLABEL = {"triggered_audited": "TRIGGERED (audited)", "artifact_suspect": "artifact-suspect",
            "latent": "latent (nec. cond., not triggered)", "genuine_failure": "GENUINE failure",
            "well_conditioned": "well-conditioned"}
print(f"{'dataset':34s} {'>10':4s} {'field':14s} {'min_var':>9s} {'≤lib':>4s} {'≤fix':>4s} {'files':>6s}  category")
print("-" * 115)
for r in rows:
    print(f"{r['dataset']:34s} {r['gt10']:4s} {r['most_susceptible_field']:14s} "
          f"{r['min_var']:9.1e} {r['frac_le_epslib']*100:3.0f}% {r['frac_le_epsfix']*100:3.0f}% "
          f"{r['n_files']}/{str(r['n_total']):>3s}  {CATLABEL[r['category']]}")

# tallies
from collections import Counter
c = Counter(r["category"] for r in rows)
print("\nTALLY:", dict(c))
json.dump(rows, open(".gate-work/census/MAP.json", "w"), indent=1)
print("saved .gate-work/census/MAP.json  (", len(rows), "datasets )")
