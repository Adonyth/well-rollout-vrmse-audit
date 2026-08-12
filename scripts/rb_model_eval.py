#!/usr/bin/env python3
"""Rayleigh-Benard checkpoint evaluation — the second checkpoint-level audit.

Mirrors rt_model_eval.py's protocol exactly (4-frame history, 30-step autoregressive
rollout from frame 4, sampled one-step starts, raw per-window MSE / target-variance /
pred-variance scalars written per row) but for the 2-D Rayleigh-Benard dataset and its
four public checkpoints. Frames are streamed from the public test split with h5py over
fsspec; nothing is written to disk but the small per-row JSON.

Field order follows WellDataset: t0_fields alphabetically, then t1_fields components ->
[buoyancy, pressure, velocity_x, velocity_y]; dim_in = 16 = 4 fields x 4 history steps.

Usage:
  python3 rb_model_eval.py --model UNetClassic --file rayleigh_benard_Rayleigh_1e6_Prandtl_1.hdf5 \
      --trajectories 0,1 --rollout-steps 30 --onestep-starts 0,2,5,12,29 \
      --device mps --output-dir ../results/rb_models
"""
from __future__ import annotations

import argparse
import gzip
import json
import os
import sys
import time
from typing import Any

import fsspec
import h5py
import numpy as np
import torch
import urllib.request
import yaml

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from independent_metrics import spatial_mse, spatial_sample_variance  # noqa: E402

import the_well.benchmark.models as model_zoo  # noqa: E402

HF = "https://huggingface.co/datasets/polymathic-ai/rayleigh_benard/resolve/main/"
FIELDS = ["buoyancy", "pressure", "velocity_x", "velocity_y"]
HISTORY = 4
N_SPATIAL = 2

MODEL_SPECS = {
    "FNO": ("FNO", "polymathic-ai/FNO-rayleigh_benard"),
    "TFNO": ("TFNO", "polymathic-ai/TFNO-rayleigh_benard"),
    "UNetClassic": ("UNetClassic", "polymathic-ai/UNetClassic-rayleigh_benard"),
    "UNetConvNext": ("UNetConvNext", "polymathic-ai/UNetConvNext-rayleigh_benard"),
}


def log(msg: str) -> None:
    print(msg, flush=True)


def load_stats() -> tuple[np.ndarray, np.ndarray]:
    with urllib.request.urlopen(HF + "stats.yaml", timeout=60) as r:
        stats = yaml.safe_load(r.read())
    means = np.asarray([stats["mean"]["buoyancy"], stats["mean"]["pressure"],
                        *stats["mean"]["velocity"]], dtype=np.float32)
    stds = np.asarray([stats["std"]["buoyancy"], stats["std"]["pressure"],
                       *stats["std"]["velocity"]], dtype=np.float32)
    stds = np.maximum(stds, 1e-4)   # ZScoreNormalization min_denom, as in the library
    return means, stds


def fetch_frames(url: str, traj: int, lo: int, hi: int) -> np.ndarray:
    """Frames [lo, hi] inclusive -> array [T, X, Y, F] float32."""
    fs = fsspec.filesystem("http")
    t0 = time.perf_counter()
    with fs.open(url, "rb", block_size=8 * 1024 * 1024) as fo:
        with h5py.File(fo, "r") as h5:
            buoy = h5["t0_fields/buoyancy"][traj, lo:hi + 1]
            pres = h5["t0_fields/pressure"][traj, lo:hi + 1]
            vel = h5["t1_fields/velocity"][traj, lo:hi + 1]
    out = np.stack([buoy, pres, vel[..., 0], vel[..., 1]], axis=-1).astype(np.float32)
    log(f"    fetched frames {lo}-{hi} traj {traj}: {out.shape} in {time.perf_counter()-t0:.0f}s")
    return out


def frames_to_channels_first(frames: np.ndarray) -> np.ndarray:
    t, x, y, f = frames.shape
    return frames.transpose(0, 3, 1, 2).reshape(1, t * f, x, y)


def forward(model: Any, window_norm: np.ndarray, device: torch.device) -> np.ndarray:
    inp = torch.nan_to_num(torch.from_numpy(frames_to_channels_first(window_norm))).to(device)
    with torch.inference_mode():
        pred = model(inp)
    out = pred.detach().to("cpu").numpy()[0]
    del inp, pred
    return out.transpose(1, 2, 0)      # [X,Y,F] normalized


def evaluate(model_name: str, model: Any, device: torch.device, url: str, fname: str,
             traj: int, means: np.ndarray, stds: np.ndarray,
             rollout_steps: int, onestep_starts: list[int]) -> list[dict[str, Any]]:
    hi = max(HISTORY + rollout_steps - 1, max(onestep_starts) + HISTORY if onestep_starts else 0)
    frames = fetch_frames(url, traj, 0, hi)
    rows: list[dict[str, Any]] = []

    # ---- autoregressive rollout ----
    t0 = time.perf_counter()
    state = (frames[:HISTORY] - means) / stds
    for step in range(1, rollout_steps + 1):
        target_t = HISTORY + step - 1
        pred_norm = forward(model, state, device)
        pred_raw = pred_norm.astype(np.float64) * stds.astype(np.float64) + means.astype(np.float64)
        target = frames[target_t].astype(np.float64)
        rows.append({
            "file": fname, "trajectory": traj, "mode": "rollout", "model": model_name,
            "t": target_t, "rollout_step": step,
            "mse": spatial_mse(pred_raw, target, N_SPATIAL).tolist(),
            "target_variance_ddof1": spatial_sample_variance(target, N_SPATIAL).tolist(),
            "pred_variance_ddof1": spatial_sample_variance(pred_raw, N_SPATIAL).tolist(),
        })
        state = np.concatenate([state[1:], pred_norm[None].astype(np.float32)], axis=0)
    log(f"    rollout {rollout_steps} steps in {time.perf_counter()-t0:.0f}s")

    # ---- one-step ----
    t0 = time.perf_counter()
    for s in onestep_starts:
        window = (frames[s:s + HISTORY] - means) / stds
        pred_norm = forward(model, window, device)
        pred_raw = pred_norm.astype(np.float64) * stds.astype(np.float64) + means.astype(np.float64)
        target = frames[s + HISTORY].astype(np.float64)
        rows.append({
            "file": fname, "trajectory": traj, "mode": "onestep", "model": model_name,
            "t": s + HISTORY, "input_start": s,
            "mse": spatial_mse(pred_raw, target, N_SPATIAL).tolist(),
            "target_variance_ddof1": spatial_sample_variance(target, N_SPATIAL).tolist(),
            "pred_variance_ddof1": spatial_sample_variance(pred_raw, N_SPATIAL).tolist(),
        })
    log(f"    one-step x{len(onestep_starts)} in {time.perf_counter()-t0:.0f}s")
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=list(MODEL_SPECS))
    ap.add_argument("--revision", required=True, help="pinned HF commit sha (enforced, not logged)")
    ap.add_argument("--file", required=True)
    ap.add_argument("--trajectories", default="0")
    ap.add_argument("--rollout-steps", type=int, default=30)
    ap.add_argument("--onestep-starts", default="0,2,5,12,29")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--output-dir", default="../results/rb_models")
    ap.add_argument("--cache-dir", default="../hf-cache")
    args = ap.parse_args()

    cls_name, repo = MODEL_SPECS[args.model]
    # enforce the pin before any weights are fetched (same discipline as rt_model_eval)
    with urllib.request.urlopen(f"https://huggingface.co/api/models/{repo}", timeout=60) as r:
        live_sha = json.load(r).get("sha")
    assert live_sha == args.revision, f"revision drift: live {live_sha} != pinned {args.revision}"

    device = torch.device(args.device)
    model = getattr(model_zoo, cls_name).from_pretrained(
        repo, revision=args.revision, cache_dir=args.cache_dir).to(device).eval()

    means, stds = load_stats()
    url = HF + "data/test/" + args.file
    starts = [int(x) for x in args.onestep_starts.split(",") if x != ""]
    trajs = [int(x) for x in args.trajectories.split(",")]

    os.makedirs(args.output_dir, exist_ok=True)
    all_rows: list[dict[str, Any]] = []
    for traj in trajs:
        log(f"  {args.model} {args.file} traj {traj}")
        all_rows += evaluate(args.model, model, device, url, args.file, traj,
                             means, stds, args.rollout_steps, starts)

    stem = args.file.replace(".hdf5", "")
    out = os.path.join(args.output_dir, f"{args.model}_{stem}.json.gz")
    json.dump({"rows": all_rows, "fields": FIELDS,
               "provenance": {"repo": repo, "revision": args.revision, "device": args.device,
                              "dataset_url": url, "history": HISTORY,
                              "captured_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}},
              open(out, "w"))
    log(f"wrote {out} ({len(all_rows)} rows)")


if __name__ == "__main__":
    main()
