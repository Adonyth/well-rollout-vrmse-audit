#!/usr/bin/env python3
"""Data-only denominator census for Rayleigh-Benard (second artifact case).

Streams The Well RB test files via HuggingFace range reads (no full download);
computes per-frame spatial sample variance (ddof=1) for velocity components
(degenerate, near-zero denominator candidate) and buoyancy (well-conditioned
reference), over the benchmark rollout window (frames 4-33 = steps 1-30,
history=4). No model in the loop.
"""
import fsspec, h5py, numpy as np, json, sys, time

HF = "https://huggingface.co/datasets/polymathic-ai/rayleigh_benard/resolve/main/data/test/"
# a spread of Rayleigh/Prandtl test files
FILES = [
    "rayleigh_benard_Rayleigh_1e6_Prandtl_1.hdf5",
    "rayleigh_benard_Rayleigh_1e7_Prandtl_1.hdf5",
    "rayleigh_benard_Rayleigh_1e8_Prandtl_1.hdf5",
]
WIN0, WIN1 = 4, 33           # benchmark rollout target frames (like RT)
EPS_LIB, EPS_FIX = 1e-7, 1e-5

def spatial_var(field):  # field: (nt, H, W) -> per-frame ddof=1 variance
    f = field.astype(np.float64).reshape(field.shape[0], -1)
    return f.var(axis=1, ddof=1)

def main():
    fs = fsspec.filesystem("http")
    out = {"window_frames": [WIN0, WIN1], "eps_lib": EPS_LIB, "eps_fix": EPS_FIX, "files": {}}
    for fname in FILES:
        url = HF + fname
        t = time.time()
        rec = {"trajectories": []}
        with fs.open(url, "rb", block_size=4 * 1024 * 1024) as fo:
            with h5py.File(fo, "r") as h5:
                velo = h5["t1_fields/velocity"]     # (5,200,512,128,2)
                buoy = h5["t0_fields/buoyancy"]      # (5,200,512,128)
                ntraj = velo.shape[0]
                for tr in range(min(2, ntraj)):      # 2 trajectories per file
                    v = velo[tr, WIN0:WIN1 + 1]      # (nwin,512,128,2)
                    b = buoy[tr, WIN0:WIN1 + 1]       # (nwin,512,128)
                    vx = spatial_var(v[..., 0]); vy = spatial_var(v[..., 1]); bv = spatial_var(b)
                    def frac_below(arr, eps): return float(np.mean(arr <= eps))
                    rec["trajectories"].append({
                        "traj": tr,
                        "vx_var_min": float(vx.min()), "vx_var_max": float(vx.max()), "vx_var_mean": float(vx.mean()),
                        "vy_var_min": float(vy.min()), "vy_var_max": float(vy.max()), "vy_var_mean": float(vy.mean()),
                        "buoy_var_min": float(bv.min()), "buoy_var_max": float(bv.max()),
                        "vx_frac_below_epslib": frac_below(vx, EPS_LIB), "vx_frac_below_epsfix": frac_below(vx, EPS_FIX),
                        "vy_frac_below_epslib": frac_below(vy, EPS_LIB), "vy_frac_below_epsfix": frac_below(vy, EPS_FIX),
                        "buoy_frac_below_epsfix": frac_below(bv, EPS_FIX),
                    })
        rec["read_seconds"] = round(time.time() - t, 1)
        out["files"][fname] = rec
        print(f"{fname}: {rec['read_seconds']}s")
        for tr in rec["trajectories"]:
            print(f"  traj{tr['traj']}: vx_var[{tr['vx_var_min']:.2e},{tr['vx_var_max']:.2e}] "
                  f"vy_var[{tr['vy_var_min']:.2e},{tr['vy_var_max']:.2e}] "
                  f"buoy_var[{tr['buoy_var_min']:.2e},{tr['buoy_var_max']:.2e}] | "
                  f"vx<=epsfix {tr['vx_frac_below_epsfix']:.0%} vy<=epsfix {tr['vy_frac_below_epsfix']:.0%} "
                  f"buoy<=epsfix {tr['buoy_frac_below_epsfix']:.0%}")
        sys.stdout.flush()
    json.dump(out, open(".gate-work/rb_census_out.json", "w"), indent=2)
    print("\nsaved .gate-work/rb_census_out.json")

if __name__ == "__main__":
    main()
