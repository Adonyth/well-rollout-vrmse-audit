#!/usr/bin/env python3
"""Worked example: normscreen on PDEBench, a benchmark with a different metric design.

PDEBench (Takamoto et al., NeurIPS 2022 D&B) is the other widely used physics-ML PDE
benchmark. Its headline normalized error divides by a DIFFERENT quantity than The Well's
VRMSE, and applies NO stabilizing constant at all -- from pdebench/models/metrics.py:

    nrm       = torch.sqrt(torch.mean(target.view([nb, nc, -1, nt]) ** 2, dim=2))
    err_nRMSE = torch.mean(err_mean / nrm, dim=0)

so the denominator is the RMS of the target (second moment about ZERO), not its variance
(second moment about the mean), and the division is bare. Two consequences:

  * RMS^2 = Var + mean^2, so a field oscillating slightly about a large offset has a
    healthy RMS and a degenerate variance. The two benchmarks' normalizers therefore
    disagree about which fields are scoreable -- neither choice is universally safer.
  * With no floor, the failure mode is UNBOUNDEDNESS rather than saturation. A floored
    metric caps at 1/sqrt(eps); a floorless one does not cap at all.

This example downloads one small public PDEBench file and screens it under both
conventions. It needs no model and no PDEBench install.

    python3 examples/pdebench_shocktube.py

Data: DaRUS doi:10.18419/darus-2986, file id 133150 (1D/CFD/Train/ShockTube/Sod6.hdf5),
~4.9 MB, CC-BY. Downloaded to /tmp on first run.
"""
import os
import sys
import urllib.request

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from normscreen.screen import screen_fields

URL = "https://darus.uni-stuttgart.de/api/access/datafile/133150"
DEST = "/tmp/pdebench_Sod6.hdf5"
FIELDS = ("Vx", "density", "pressure")


def fetch():
    if not os.path.exists(DEST):
        print(f"downloading {URL} -> {DEST}")
        urllib.request.urlretrieve(URL, DEST)
    return DEST


def main() -> int:
    try:
        import h5py
    except ImportError:
        sys.exit("this example needs h5py: pip install h5py")
    path = fetch()
    with h5py.File(path, "r") as f:
        fields = {k: np.asarray(f[k]) for k in FIELDS}
    print(f"\nfile: {os.path.basename(path)}   shapes: "
          + ", ".join(f"{k}{v.shape}" for k, v in fields.items()))

    print("\n" + "=" * 78)
    print("AS PDEBENCH SCORES IT  (normalizer = rms, no stabilizing constant)")
    print("=" * 78)
    rms = screen_fields(fields, eps=0.0, n_spatial=1, normalizer="rms")
    print(rms.text())

    print("\n" + "=" * 78)
    print("AS THE WELL WOULD SCORE IT  (normalizer = variance, eps = 1e-7)")
    print("=" * 78)
    var = screen_fields(fields, eps=1e-7, n_spatial=1, normalizer="variance")
    print(var.text())

    print("\n" + "=" * 78)
    print("WHAT THE SCREEN FOUND")
    print("=" * 78)
    vx = fields["Vx"]
    print(f"  Vx is identically zero: {bool(np.all(vx == 0))} "
          f"({int(np.count_nonzero(vx))} non-zero of {vx.size} entries)")
    print(f"  density time-invariant: {bool(np.allclose(fields['density'][0], fields['density'][-1]))}")
    print(f"  pressure time-invariant: {bool(np.allclose(fields['pressure'][0], fields['pressure'][-1]))}")
    print()
    print("  This particular file contains a static state: the velocity channel is exactly")
    print("  zero everywhere, and density and pressure are unchanged from the first frame to")
    print("  the last. Under PDEBench's floorless nRMSE the velocity denominator is therefore")
    print("  exactly zero on every frame, so that channel's score is undefined rather than")
    print("  large. The screen reports this in one pass, with no model and no training run.")
    print()
    print("  SCOPE: this is ONE file, and an anomalous one -- 4.9 MB against 12-25 GB for")
    print("  every other file in the same 1D/CFD collection. We make no claim about PDEBench")
    print("  as a whole; a data-only screen is exactly the cheap check that would have")
    print("  flagged it before it shipped.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
