"""Direct HTTP-Range reader for The Well RT test objects.

The RT HDF5 datasets are stored CONTIGUOUS (h5py .chunks is None), so exact
byte spans for any (trajectory, frame-range) can be fetched with plain ranged
GETs at full server throughput, bypassing h5py/fsspec request overhead.

Offsets are discovered once per file via h5py over fsspec (dataset
``id.get_offset()``) and each file's first frame is cross-validated against an
h5py read before trust.
"""

from __future__ import annotations

import concurrent.futures as cf
from dataclasses import dataclass

import numpy as np
import requests

FRAME_VOX = 128 * 128 * 128
DENSITY_FRAME_BYTES = FRAME_VOX * 4
VELOCITY_FRAME_BYTES = FRAME_VOX * 3 * 4


@dataclass
class FileLayout:
    url: str
    n_trajectories: int
    n_frames: int
    density_offset: int
    velocity_offset: int
    etag: str
    size: int


def discover_layout(url: str) -> FileLayout:
    import fsspec
    import h5py

    head = requests.head(url, timeout=30)
    head.raise_for_status()
    with fsspec.open(url, "rb", block_size=4 * 1024 * 1024, cache_type="readahead") as remote:
        with h5py.File(remote, "r") as h5:
            dens = h5["t0_fields/density"]
            velo = h5["t1_fields/velocity"]
            assert dens.chunks is None and velo.chunks is None, "layout not contiguous"
            assert dens.dtype == np.float32 and velo.dtype == np.float32
            n_traj, n_frames = int(dens.shape[0]), int(dens.shape[1])
            assert dens.shape == (n_traj, n_frames, 128, 128, 128)
            assert velo.shape == (n_traj, n_frames, 128, 128, 128, 3)
            layout = FileLayout(
                url=url,
                n_trajectories=n_traj,
                n_frames=n_frames,
                density_offset=dens.id.get_offset(),
                velocity_offset=velo.id.get_offset(),
                etag=head.headers.get("ETag", ""),
                size=int(head.headers["Content-Length"]),
            )
            # cross-validate first frame of trajectory 0 via both paths
            ref_d = np.asarray(dens[0, 0])
            ref_v = np.asarray(velo[0, 0, :, :, :64])  # partial to keep it light
    got = read_frames(layout, 0, 0, 1)
    assert np.array_equal(got[0, ..., 0], ref_d), "density range-read mismatch"
    assert np.array_equal(got[0, :, :, :64, 1:4], ref_v), "velocity range-read mismatch"
    return layout


def _get_with_retry(url: str, start: int, length: int, timeout: int) -> bytes:
    """Single ranged GET with retry/backoff for transient DNS/conn failures."""
    import time as _time

    last: Exception | None = None
    for attempt in range(6):
        try:
            r = requests.get(
                url,
                headers={"Range": f"bytes={start}-{start + length - 1}"},
                timeout=timeout,
            )
            r.raise_for_status()
            assert len(r.content) == length
            return r.content
        except (requests.exceptions.RequestException, AssertionError) as exc:
            last = exc
            _time.sleep(min(60, 2 ** (attempt + 1)))
    raise RuntimeError(f"ranged GET failed after retries: {last}")


def _ranged_get(url: str, start: int, length: int, n_workers: int = 4) -> bytes:
    """Fetch [start, start+length) with parallel sub-range GETs."""
    if length <= 32 * 1024 * 1024:
        return _get_with_retry(url, start, length, 300)
    piece = (length + n_workers - 1) // n_workers

    def fetch(k: int) -> bytes:
        s = start + k * piece
        ln = min(piece, start + length - s)
        return _get_with_retry(url, s, ln, 600)

    with cf.ThreadPoolExecutor(n_workers) as pool:
        parts = list(pool.map(fetch, range(n_workers)))
    return b"".join(parts)


def read_frames(layout: FileLayout, trajectory: int, t0: int, count: int) -> np.ndarray:
    """Return frames [count, 128,128,128, 4] float32 (density, vx, vy, vz)."""
    d_start = layout.density_offset + (
        trajectory * layout.n_frames + t0
    ) * DENSITY_FRAME_BYTES
    v_start = layout.velocity_offset + (
        trajectory * layout.n_frames + t0
    ) * VELOCITY_FRAME_BYTES
    with cf.ThreadPoolExecutor(2) as pool:
        f_d = pool.submit(_ranged_get, layout.url, d_start, count * DENSITY_FRAME_BYTES, 2)
        f_v = pool.submit(_ranged_get, layout.url, v_start, count * VELOCITY_FRAME_BYTES, 4)
        raw_d, raw_v = f_d.result(), f_v.result()
    dens = np.frombuffer(raw_d, dtype="<f4").reshape(count, 128, 128, 128)
    velo = np.frombuffer(raw_v, dtype="<f4").reshape(count, 128, 128, 128, 3)
    out = np.empty((count, 128, 128, 128, 4), dtype=np.float32)
    out[..., 0] = dens
    out[..., 1:4] = velo
    return out
