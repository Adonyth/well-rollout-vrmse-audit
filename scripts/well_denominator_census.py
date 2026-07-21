#!/usr/bin/env python3
"""General data-only denominator census for ANY Well dataset.

Streams a spread of a dataset's public test split by exact-offset range reads
(no full download), auto-discovers every field (t0 scalars, t1 vectors, t2
tensors) by walking the HDF5 groups, and reports per-field/per-component spatial
variance over the benchmark rollout window together with the fraction of that
window at or below each epsilon floor. No model in the loop.

Usage: python3 well_denominator_census.py <dataset_name> [n_files] [win0] [win1]
Writes .gate-work/census/<dataset_name>.json
"""
import fsspec, h5py, numpy as np, json, time, sys, os, urllib.request

EPS_LIB, EPS_FIX = 1e-7, 1e-5
DEF_WIN0, DEF_WIN1 = 4, 33
API = "https://huggingface.co/api/datasets/polymathic-ai/{ds}/tree/main/data/test"
RESOLVE = "https://huggingface.co/datasets/polymathic-ai/{ds}/resolve/main/data/test/"


def list_test_files(ds):
    url = API.format(ds=ds)
    with urllib.request.urlopen(url, timeout=60) as r:
        tree = json.load(r)
    files = sorted(x["path"].split("/")[-1] for x in tree
                   if x.get("type") == "file" and x["path"].endswith(".hdf5"))
    return files


def pick_spread(files, n):
    """Pick n files evenly spread across the sorted list (parameter spread)."""
    if len(files) <= n or n <= 1:
        return files[:max(1, n)]
    idx = [round(i * (len(files) - 1) / (n - 1)) for i in range(n)]
    return [files[i] for i in sorted(set(idx))]


def spatial_var_per_frame(arr):
    a = arr.astype(np.float64).reshape(arr.shape[0], -1)
    return a.var(axis=1, ddof=1)


def census_field_group(h5, grp, rank, win0, win1, out, fstride=1):
    if grp not in h5:
        return
    for name in h5[grp]:
        ds = h5[grp][name]
        T = ds.shape[1]                      # (n_traj, T, *spatial[, comp...])
        w1 = min(win1, T - 1)
        block = ds[0, win0:w1 + 1:fstride]   # trajectory 0, window frames (strided)
        nf = block.shape[0]
        if rank == 0:
            comps = {name: block}
        elif rank == 1:
            d = block.shape[-1]
            comps = {f"{name}.{i}": block[..., i] for i in range(d)}
        else:                                # rank-2 tensor
            d0, d1 = block.shape[-2], block.shape[-1]
            comps = {f"{name}.{i}{j}": block[..., i, j]
                     for i in range(d0) for j in range(d1)}
        for cname, carr in comps.items():
            v = spatial_var_per_frame(carr)
            out[cname] = {
                "var_min": float(v.min()), "var_max": float(v.max()),
                "var_mean": float(v.mean()),
                "frac_below_epslib": float(np.mean(v <= EPS_LIB)),
                "frac_below_epsfix": float(np.mean(v <= EPS_FIX)),
                "n_frames": int(nf), "window": [int(win0), int(w1)],
            }


def census_file(fs, url, win0, win1, fstride=1):
    fields = {}
    with fs.open(url, "rb", block_size=4 * 1024 * 1024) as fo:
        with h5py.File(fo, "r") as h5:
            census_field_group(h5, "t0_fields", 0, win0, win1, fields, fstride)
            census_field_group(h5, "t1_fields", 1, win0, win1, fields, fstride)
            census_field_group(h5, "t2_fields", 2, win0, win1, fields, fstride)
    return fields


def main():
    ds = sys.argv[1]
    n_files = int(sys.argv[2]) if len(sys.argv) > 2 else 3
    win0 = int(sys.argv[3]) if len(sys.argv) > 3 else DEF_WIN0
    win1 = int(sys.argv[4]) if len(sys.argv) > 4 else DEF_WIN1
    fstride = int(sys.argv[5]) if len(sys.argv) > 5 else 1
    all_files = list_test_files(ds)
    picked = pick_spread(all_files, n_files)
    base = RESOLVE.format(ds=ds)
    fs = fsspec.filesystem("http")
    out = {"dataset": ds, "n_test_files_total": len(all_files),
           "window_frames": [win0, win1], "frame_stride": fstride,
           "eps_lib": EPS_LIB, "eps_fix": EPS_FIX,
           "files_censused": picked, "files": {}}
    for fn in picked:
        t = time.time()
        try:
            fields = census_file(fs, base + fn, win0, win1, fstride)
        except Exception as e:
            out["files"][fn] = {"error": str(e)}
            print(f"  {fn}: ERROR {e}", flush=True)
            continue
        sec = round(time.time() - t, 1)
        out["files"][fn] = {"fields": fields, "sec": sec}
        # compact per-file line: which fields dip to/below each floor
        flagged = [f"{k}(lib{d['frac_below_epslib']:.0%}/fix{d['frac_below_epsfix']:.0%},min{d['var_min']:.1e})"
                   for k, d in fields.items() if d["frac_below_epsfix"] > 0 or d["var_min"] <= EPS_FIX]
        healthy = [k for k, d in fields.items() if d["frac_below_epsfix"] == 0 and d["var_min"] > EPS_FIX]
        print(f"  {fn}: {sec}s | FLOORED: {flagged or 'none'} | healthy: {healthy}", flush=True)
    os.makedirs(".gate-work/census", exist_ok=True)
    json.dump(out, open(f".gate-work/census/{ds}.json", "w"), indent=1)
    print(f"saved .gate-work/census/{ds}.json  (total test files: {len(all_files)})", flush=True)


if __name__ == "__main__":
    main()
