"""normscreen CLI -- point it at a benchmark split, get a conditioning report.

    python3 -m normscreen data.h5 --auto
    python3 -m normscreen data.h5 --fields density velocity_x --spatial-dims 3
    python3 -m normscreen field.npy --spatial-dims 2 --json out.json
    python3 -m normscreen --demo
"""
from __future__ import annotations

import argparse
import json
import sys
from typing import Any

import numpy as np

from .screen import screen_fields


def _load_hdf5(path: str, wanted: list[str] | None, auto: bool,
               max_bytes: int) -> dict[str, np.ndarray]:
    try:
        import h5py
    except ImportError:
        sys.exit("reading HDF5 needs h5py: pip install h5py")
    out: dict[str, np.ndarray] = {}
    with h5py.File(path, "r") as f:
        found: list[tuple[str, Any]] = []

        def visit(name, obj):
            if isinstance(obj, h5py.Dataset) and obj.ndim >= 2:
                found.append((name, obj))
        f.visititems(visit)
        if not found:
            sys.exit(f"{path}: no datasets with ndim>=2 found")
        for name, dset in found:
            leaf = name.split("/")[-1]
            if wanted and leaf not in wanted and name not in wanted:
                continue
            if not wanted and not auto:
                continue
            if dset.nbytes > max_bytes:
                print(f"  [skip] {name}: {dset.nbytes/1e9:.1f} GB exceeds --max-gb "
                      f"(raise it, or pre-slice the file)", file=sys.stderr)
                continue
            out[leaf] = np.asarray(dset[...])
        if not out and not wanted:
            sys.exit("nothing selected; pass --auto or --fields")
        if not out:
            names = sorted({n.split('/')[-1] for n, _ in found})
            sys.exit(f"none of {wanted} found. available: {names}")
    return out


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        prog="normscreen",
        description="Conditioning screen for variance-normalized error metrics "
                    "(VRMSE / NRMSE / Nash-Sutcliffe-family scores).")
    p.add_argument("path", nargs="?", help="HDF5 (.h5/.hdf5) or NumPy (.npy/.npz) file")
    p.add_argument("--fields", nargs="*", default=None,
                   help="dataset names to screen (default: --auto for all)")
    p.add_argument("--auto", action="store_true", help="screen every array of ndim>=2")
    p.add_argument("--spatial-dims", type=int, default=2, dest="n_spatial",
                   help="number of trailing spatial axes (2 for 2D, 3 for 3D). default 2")
    p.add_argument("--eps", type=float, default=1e-7,
                   help="the stabilizing constant your metric uses. default 1e-7")
    p.add_argument("--ddof", type=int, default=1, help="variance ddof. default 1 (sample)")
    p.add_argument("--normalizer", choices=["variance", "rms"], default="variance",
                   help="what the metric divides by: 'variance' (Nash-Sutcliffe family, "
                        "e.g. The Well's VRMSE) or 'rms' (mean of squares, e.g. PDEBench's "
                        "nRMSE). default variance")
    p.add_argument("--component-axis", type=int, default=None, metavar="AXIS",
                   help="name the component axis explicitly. By default the screen infers "
                        "it: any field with more axes than one leading index plus the "
                        "spatial ones is split on its last axis and screened per component.")
    p.add_argument("--no-split-components", action="store_true",
                   help="disable per-component screening. NOT recommended: a vector field's "
                        "components are then averaged together and a degenerate component "
                        "can be hidden by a healthy one.")
    p.add_argument("--max-gb", type=float, default=4.0,
                   help="skip datasets larger than this, in GB. default 4")
    p.add_argument("--json", metavar="OUT", help="also write the full report as JSON")
    p.add_argument("--demo", action="store_true",
                   help="run on a synthetic quiescent-then-developed field and exit")
    a = p.parse_args(argv)

    if a.demo:
        rng = np.random.default_rng(0)
        # a field that starts at rest and develops structure -- the shape that makes a
        # variance-normalized score unreadable at early frames
        frames = []
        for t in range(40):
            amp = 1e-6 * (1.0 + t) ** 4
            frames.append(amp * rng.standard_normal((64, 64)))
        fields = {"quiescent_then_developed": np.stack(frames),
                  "always_healthy": rng.standard_normal((40, 64, 64))}
        rep = screen_fields(fields, eps=a.eps, n_spatial=2, ddof=a.ddof)
        print(rep.text())
        # same exit convention as a real run, so --demo also documents the CI contract
        return 1 if rep.worst_floor_share >= 0.90 else 0

    if not a.path:
        p.error("give a file, or --demo")

    if a.path.endswith((".npy", ".npz")):
        loaded = np.load(a.path, allow_pickle=False)
        fields = ({k: loaded[k] for k in loaded.files}
                  if hasattr(loaded, "files") else {"array": loaded})
        if a.fields:
            fields = {k: v for k, v in fields.items() if k in a.fields}
    else:
        fields = _load_hdf5(a.path, a.fields, a.auto or not a.fields,
                            int(a.max_gb * 1e9))

    rep = screen_fields(fields, eps=a.eps, n_spatial=a.n_spatial, ddof=a.ddof,
                        normalizer=a.normalizer,
                        component_axis=(None if a.no_split_components
                                        else (a.component_axis if a.component_axis is not None
                                              else "auto")))
    print(rep.text())
    if a.json:
        with open(a.json, "w", encoding="utf-8") as fh:
            json.dump(rep.to_dict(), fh, indent=1, sort_keys=True)
        print(f"\nwrote {a.json}")
    # exit 1 on a positive screen so the tool composes into CI. A zero denominator under
    # a floorless metric is also a positive: the score there is unbounded, not merely
    # saturated.
    degenerate = any(f.frames_zero_denominator for f in rep.fields)
    return 1 if (rep.worst_floor_share >= 0.90 or degenerate) else 0


if __name__ == "__main__":
    raise SystemExit(main())
