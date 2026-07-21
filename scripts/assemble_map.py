#!/usr/bin/env python3
"""Assemble the benchmark-wide denominator-conditioning map (paper Table tab:map)
from the per-dataset census fixtures and the verified Well Table-3 ">10" status.

Reproducible: reads the frozen per-dataset census outputs and the frozen
Table-3 status, recomputes every dataset's susceptibility and category, and
(by default) checks the result against the frozen MAP.json, exiting non-zero on
any disagreement. Run from the repo root or the harness root.

  python3 scripts/assemble_map.py            # verify frozen MAP.json
  python3 scripts/assemble_map.py --write     # regenerate MAP.json
"""
import json, glob, os, sys

EPSLIB, EPSFIX = 1e-7, 1e-5


def find_dir():
    for d in ("fixtures/generalization/benchmark_map",
              "repro-harness/fixtures/generalization/benchmark_map"):
        if os.path.isdir(d):
            return d
    sys.exit("benchmark_map fixture dir not found")


def gt10_path(base):
    for p in (os.path.join(base, "..", "well_table3_gt10.json"),
              "fixtures/generalization/well_table3_gt10.json",
              "repro-harness/fixtures/generalization/well_table3_gt10.json"):
        if os.path.exists(p):
            return p
    sys.exit("well_table3_gt10.json not found")


def per_dataset(fp, gt10):
    d = json.load(open(fp))
    ds = d["dataset"]
    best = {}
    for fd in d.get("files", {}).values():
        for k, v in fd.get("fields", {}).items():
            b = best.setdefault(k, {"min": 1e99, "flib": 0.0, "ffix": 0.0})
            b["min"] = min(b["min"], v["var_min"])
            b["flib"] = max(b["flib"], v["frac_below_epslib"])
            b["ffix"] = max(b["ffix"], v["frac_below_epsfix"])
    if not best:
        return None
    fld = min(best, key=lambda k: best[k]["min"])
    b = best[fld]
    gmin = b["min"]
    suscept = "severe" if gmin <= EPSLIB else ("susceptible" if gmin <= EPSFIX else "healthy")
    gt = gt10.get(ds, "?")
    floored = suscept in ("severe", "susceptible")
    is_gt = (gt == "yes")
    if floored and is_gt:
        cat = "triggered_audited" if ds == "rayleigh_taylor_instability" else "artifact_suspect"
    elif floored and not is_gt:
        cat = "latent"
    elif (not floored) and is_gt:
        cat = "genuine_failure"
    else:
        cat = "well_conditioned"
    return {"dataset": ds, "n_files": len(d.get("files", {})), "n_total": d.get("n_test_files_total"),
            "frame_stride": d.get("frame_stride", 1), "gt10": gt,
            "most_susceptible_field": fld, "min_var": gmin,
            "frac_le_epslib": b["flib"], "frac_le_epsfix": b["ffix"],
            "susceptibility": suscept, "category": cat}


def main():
    base = find_dir()
    gt10 = json.load(open(gt10_path(base)))["status"]
    rows = [per_dataset(fp, gt10) for fp in sorted(glob.glob(os.path.join(base, "*.json")))
            if not fp.endswith("MAP.json")]
    rows = [r for r in rows if r]
    order = {"triggered_audited": 0, "artifact_suspect": 1, "latent": 2,
             "genuine_failure": 3, "well_conditioned": 4}
    rows.sort(key=lambda r: (order.get(r["category"], 9), r["min_var"]))
    mp = os.path.join(base, "MAP.json")
    if "--write" in sys.argv:
        json.dump(rows, open(mp, "w"), indent=1)
        print(f"wrote {mp} ({len(rows)} datasets)")
        return
    frozen = json.load(open(mp))
    fz = {r["dataset"]: (r["gt10"], r["susceptibility"], r["category"]) for r in frozen}
    bad = [r["dataset"] for r in rows
           if fz.get(r["dataset"]) != (r["gt10"], r["susceptibility"], r["category"])]
    if bad:
        print("MAP MISMATCH on:", bad); sys.exit(1)
    from collections import Counter
    print("MAP OK:", dict(Counter(r["category"] for r in rows)), f"({len(rows)} datasets)")


if __name__ == "__main__":
    main()
