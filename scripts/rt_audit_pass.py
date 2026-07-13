"""Pass A — full-test-split denominator audit (no models).

Extends Lane 3's rt_denominator_audit.py (first file, 2 trajectories) to ALL
5 official RT test objects / 10 trajectories: per (file, trajectory, frame,
field): spatial sample variance (ddof=1), spatial mean, persistence MSE.
All computation float64; reuses Lane 3's verified spatial_mse /
spatial_sample_variance. Data via fast_reader (contiguous ranged GETs).

Output: results/audit/<file>_traj<k>.json.gz + provenance.
"""

from __future__ import annotations

import argparse
import gzip
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
# repro-harness note: original had a hardcoded absolute path to lane-3/;
# independent_metrics.py is now colocated in this scripts/ dir instead.
from independent_metrics import spatial_mse, spatial_sample_variance  # noqa: E402

from fast_reader import discover_layout, read_frames  # noqa: E402

BASE = (
    "https://sdsc-users.flatironinstitute.org/~polymathic/data/the_well/"
    "datasets/rayleigh_taylor_instability/data/test"
)
TEST_FILES = [
    "rayleigh_taylor_instability_At_75.hdf5",
    "rayleigh_taylor_instability_At_25.hdf5",
    "rayleigh_taylor_instability_At_125.hdf5",
    "rayleigh_taylor_instability_At_50.hdf5",
    "rayleigh_taylor_instability_At_0625.hdf5",
]
FIELDS = ["density", "velocity_x", "velocity_y", "velocity_z"]
BLOCK = 10  # frames per bulk fetch (~336 MB)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--files", default="0,1,2,3,4")
    parser.add_argument("--output-dir", type=Path, default=HERE / "results" / "audit")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    provenance = {"access_date": time.strftime("%Y-%m-%d"), "fields": FIELDS, "files": {}}
    started = time.perf_counter()
    for fi in [int(i) for i in args.files.split(",")]:
        fname = TEST_FILES[fi]
        url = f"{BASE}/{fname}"
        layout = discover_layout(url)
        provenance["files"][fname] = {
            "url": url,
            "bytes": layout.size,
            "etag": layout.etag,
            "n_trajectories": layout.n_trajectories,
            "n_frames": layout.n_frames,
            "density_offset": layout.density_offset,
            "velocity_offset": layout.velocity_offset,
            "range_read_crossvalidated_vs_h5py": True,
        }
        print(f"[audit] {fname}: {layout.n_trajectories} traj x {layout.n_frames} frames", flush=True)
        for traj in range(layout.n_trajectories):
            out_path = args.output_dir / f"{fname.removesuffix('.hdf5')}_traj{traj}.json.gz"
            if out_path.exists():
                print(f"  SKIP {out_path.name}", flush=True)
                continue
            rows = []
            prev = None
            t_net = 0.0
            t_start = time.perf_counter()
            for t0 in range(0, layout.n_frames, BLOCK):
                count = min(BLOCK, layout.n_frames - t0)
                tick = time.perf_counter()
                frames = read_frames(layout, traj, t0, count)
                t_net += time.perf_counter() - tick
                for k in range(count):
                    f64 = frames[k].astype(np.float64)
                    row = {
                        "t": t0 + k,
                        "variance_ddof1": spatial_sample_variance(f64, 3).tolist(),
                        "spatial_mean": f64.mean(axis=(0, 1, 2)).tolist(),
                    }
                    if prev is not None:
                        row["persistence_mse"] = spatial_mse(prev, f64, 3).tolist()
                    rows.append(row)
                    prev = f64
                del frames
            payload = {
                "file": fname,
                "trajectory": traj,
                "fields": FIELDS,
                "rows": rows,
                "net_seconds": t_net,
                "wall_seconds": time.perf_counter() - t_start,
            }
            with gzip.open(out_path, "wt", encoding="utf-8") as handle:
                json.dump(payload, handle)
            print(
                f"  wrote {out_path.name} net={t_net:.0f}s wall={payload['wall_seconds']:.0f}s",
                flush=True,
            )
            prev = None
    provenance["wall_seconds"] = time.perf_counter() - started
    (args.output_dir / "provenance.json").write_text(json.dumps(provenance, indent=1) + "\n")
    print(f"[audit] DONE {provenance['wall_seconds']:.0f}s", flush=True)


if __name__ == "__main__":
    main()
