"""normscreen CLI -- point it at a benchmark split, get a conditioning report.

    python3 -m normscreen data.h5 --auto --no-split-components
    python3 -m normscreen data.h5 --fields density velocity_x --spatial-dims 3 --no-split-components
    python3 -m normscreen field.npy --spatial-dims 2 --json out.json --no-split-components
    python3 -m normscreen --demo
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from typing import Any

import numpy as np

from .screen import screen_fields


def _load_hdf5(path: str, wanted: list[str] | None, auto: bool,
               max_bytes: int, _skipped: list | None = None) -> dict[str, np.ndarray]:
    try:
        import h5py
    except ImportError:
        sys.exit("reading HDF5 needs h5py: pip install h5py")
    out: dict[str, np.ndarray] = {}
    _skipped = _skipped if _skipped is not None else []
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
                # A skipped field is a field the caller asked for and did NOT get screened.
                # An earlier version reported this on stderr only, so it was absent from the
                # report, the JSON and the exit code -- a second place that decided what got
                # screened, outside resolve_layout. Carry it through all three.
                _skipped.append(f"{name} ({dset.nbytes/1e9:.1f} GB exceeds --max-gb)")
                print(f"  [skip] {name}: {dset.nbytes/1e9:.1f} GB exceeds --max-gb "
                      f"(raise it, or pre-slice the file)", file=sys.stderr)
                continue
            # h5py names a ROOT-LEVEL dataset by its bare leaf, so name == leaf there.
            # An earlier version keyed collisions by `name`, which in that case is the key
            # already taken -- the second dataset silently OVERWROTE the first while the
            # note claimed "neither is dropped". Build a key that cannot collide, and make
            # the collision reach the report, the JSON and the exit code like any other
            # field the caller asked for and did not get screened as asked.
            key = leaf if leaf not in out else f"{leaf}@{name}"
            n = 1
            while key in out:
                n += 1
                key = f"{leaf}@{name}#{n}"
            if key != leaf:
                _skipped.append(f"leaf-name collision on {leaf!r}: {name!r} screened "
                                f"separately as {key!r}")
                print(f"  [note] two datasets share the leaf name {leaf!r}; screening this "
                      f"one separately as {key!r} so neither is dropped", file=sys.stderr)
            out[key] = np.asarray(dset[...])
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
                   help="name the component axis (e.g. -1 for channel-last, 1 for "
                        "channel-first). Fields are then split on it and screened per "
                        "component, which is what keeps a degenerate component from being "
                        "averaged away by a healthy one.")
    p.add_argument("--leading-axes", type=int, default=None, dest="n_leading", metavar="K",
                   help="how many LEADING (non-spatial, non-component) axes each field "
                        "carries, e.g. 2 for [trajectory, time, *spatial]. Declaring this "
                        "makes the layout determinate, so --component-axis can be applied "
                        "safely even when scalars and vectors share a file.")
    p.add_argument("--no-split-components", action="store_true",
                   help="screen every field whole. Use this when the extra axes are leading "
                        "indices (trajectory, time) rather than components. One of this flag "
                        "or --component-axis is REQUIRED when a field carries more axes than "
                        "one leading index plus --spatial-dims, because rank alone cannot "
                        "tell a multi-trajectory scalar from a vector and guessing wrong can "
                        "report a degenerate field as clean.")
    p.add_argument("--max-gb", type=float, default=4.0,
                   help="skip datasets larger than this, in GB. default 4")
    p.add_argument("--json", metavar="OUT", help="also write the full report as JSON")
    p.add_argument("--demo", action="store_true",
                   help="run on a synthetic quiescent-then-developed field and exit")
    a = p.parse_args(argv)

    if a.demo:
        # The demo builds its own 2-D synthetic split, so layout flags cannot be honoured.
        # Refuse them rather than run a 2-D screen while the caller believes otherwise.
        unsupported = [n for n, v in (("--spatial-dims", a.n_spatial != 2),
                                      ("--component-axis", a.component_axis is not None),
                                      ("--leading-axes", a.n_leading is not None),
                                      ("--no-split-components", a.no_split_components),
                                      ("--auto", a.auto), ("--fields", bool(a.fields)),
                                      ("--max-gb", a.max_gb != 4.0)) if v]
        if unsupported:
            p.error("--demo builds its own 2-D scalar fixture; these do not apply to it: "
                    + ", ".join(unsupported))
        rng = np.random.default_rng(0)
        # a field that starts at rest and develops structure -- the shape that makes a
        # variance-normalized score unreadable at early frames
        frames = []
        for t in range(40):
            amp = 1e-6 * (1.0 + t) ** 4
            frames.append(amp * rng.standard_normal((64, 64)))
        fields = {"quiescent_then_developed": np.stack(frames),
                  "always_healthy": rng.standard_normal((40, 64, 64))}
        rep = screen_fields(fields, eps=a.eps, n_spatial=2, ddof=a.ddof,
                            normalizer=a.normalizer, component_axis=None)
        print(rep.text())
        if a.json:
            with open(a.json, "w", encoding="utf-8") as fh:
                json.dump(rep.to_dict(), fh, indent=1, sort_keys=True)
            print(f"\nwrote {a.json}")
        # same exit convention as a real run, so --demo also documents the CI contract
        return _exit_status(rep)

    _skipped: list[str] = []
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
                            int(a.max_gb * 1e9), _skipped)

    try:
        rep = screen_fields(fields, eps=a.eps, n_spatial=a.n_spatial, ddof=a.ddof,
                            normalizer=a.normalizer,
                            n_leading=a.n_leading,
                            component_axis=(None if a.no_split_components
                                            else (a.component_axis
                                                  if a.component_axis is not None
                                                  else "auto")))
    except ValueError as e:
        # Layout ambiguity is a refusal, not a crash: exit non-zero so CI does not read an
        # unscreened split as a clean one, and tell the caller exactly which flag settles it.
        sys.exit(f"normscreen: {e}")
    if _skipped:
        rep.verdict += ("\n\nINCOMPLETE: " + str(len(_skipped)) + " requested field(s) were "
                        "not screened at all: " + "; ".join(_skipped))
    print(rep.text())
    if a.json:
        with open(a.json, "w", encoding="utf-8") as fh:
            json.dump(rep.to_dict(), fh, indent=1, sort_keys=True)
        print(f"\nwrote {a.json}")
    return _exit_status(rep)


def _exit_status(rep) -> int:
    """Exit 1 on anything a caller should not treat as a clean screen.

    That is three conditions, not one: a floor-determined denominator; a zero denominator
    under a floorless metric (unbounded rather than merely amplified by at most 1/sqrt(eps)); and a non-finite
    denominator, which is undefined rather than well conditioned and must not exit clean.
    """
    degenerate = any(f.frames_zero_denominator for f in rep.fields)
    # a floorless metric whose denominator spans orders of magnitude is not a clean screen
    floorless = any(f.band == "floorless-unbounded" for f in rep.fields)
    # A field the tool GUESSED was a scalar was not fully screened; that is not a clean run.
    incomplete = rep.verdict.startswith("INCOMPLETE") or "\nINCOMPLETE:" in rep.verdict
    undefined = any(f.frames_nonfinite_denominator for f in rep.fields)
    positive = (rep.worst_floor_share >= 0.90) if math.isfinite(rep.worst_floor_share) else True
    return 1 if (positive or degenerate or undefined or floorless or incomplete) else 0


if __name__ == "__main__":
    raise SystemExit(main())
