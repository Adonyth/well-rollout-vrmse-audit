"""Pass B — checkpoint evaluation on the RT test split (one model per process).

Mirrors the the_well 1.2.0 published-evaluation semantics:
  * one-step: sliding windows (paper Table 2 protocol), raw physical space,
    z-score normalize in / denormalize out (ZScoreNormalization, stats.yaml,
    std clip 1e-4), DefaultChannelsFirstFormatter channel flattening + nan_to_num.
  * rollout: autoregressive from the trajectory start (paper Table 3 protocol),
    prediction feedback in normalized space (equivalent to Trainer.rollout_model's
    denormalize->append->renormalize round trip), targets from ground truth.

Stores per (trajectory, mode, time, field): raw-space MSE + target/prediction
spatial sample variance (ddof=1, float64). ALL VRMSE variants (#75 eps
semantics, #78 aggregation counterfactuals) are derived downstream in
aggregate_results.py from these scalars.

Dataloader identity: WellDataset sorts file paths lexicographically
(datasets.py line 323) and rollout_test_dataloader uses batch_size=1,
shuffle=False (datamodule.py lines 375-385) => batch order is
At_0625 t0, At_0625 t1, At_125 t0, ..., At_75 t1. The LAST batch is At_75
trajectory 1.
"""

from __future__ import annotations

import argparse
import gzip
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import requests
import torch
import yaml

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
# repro-harness note: original had a hardcoded absolute path to lane-3/;
# independent_metrics.py is now colocated in this scripts/ dir instead.
from independent_metrics import spatial_mse, spatial_sample_variance  # noqa: E402

from fast_reader import FileLayout, discover_layout, read_frames  # noqa: E402

BASE = (
    "https://sdsc-users.flatironinstitute.org/~polymathic/data/the_well/"
    "datasets/rayleigh_taylor_instability"
)
# Lexicographic (= WellDataset) order:
SORTED_FILES = [
    "rayleigh_taylor_instability_At_0625.hdf5",
    "rayleigh_taylor_instability_At_125.hdf5",
    "rayleigh_taylor_instability_At_25.hdf5",
    "rayleigh_taylor_instability_At_50.hdf5",
    "rayleigh_taylor_instability_At_75.hdf5",
]
FIELDS = ["density", "velocity_x", "velocity_y", "velocity_z"]
HISTORY = 4

MODEL_SPECS = {
    "FNO": ("FNO", "polymathic-ai/FNO-rayleigh_taylor_instability"),
    "TFNO": ("TFNO", "polymathic-ai/TFNO-rayleigh_taylor_instability"),
    "UNetClassic": ("UNetClassic", "polymathic-ai/UNetClassic-rayleigh_taylor_instability"),
    "UNetConvNext": ("UNetConvNext", "polymathic-ai/UNetConvNext-rayleigh_taylor_instability"),
}


def load_stats() -> tuple[np.ndarray, np.ndarray]:
    response = requests.get(f"{BASE}/stats.yaml", timeout=30)
    response.raise_for_status()
    stats = yaml.safe_load(response.text)
    means = np.asarray([stats["mean"]["density"], *stats["mean"]["velocity"]], dtype=np.float32)
    stds = np.asarray([stats["std"]["density"], *stats["std"]["velocity"]], dtype=np.float32)
    stds = np.maximum(stds, 1e-4)  # ZScoreNormalization min_denom (no-op for RT)
    return means, stds


def frames_to_channels_first(frames: np.ndarray) -> np.ndarray:
    t, x, y, z, f = frames.shape
    return frames.transpose(0, 4, 1, 2, 3).reshape(1, t * f, x, y, z)


def forward(model: Any, window_norm: np.ndarray, device: torch.device) -> np.ndarray:
    inp = torch.nan_to_num(
        torch.from_numpy(frames_to_channels_first(window_norm))
    ).to(device)
    with torch.inference_mode():
        pred = model(inp)
    out = pred.detach().to("cpu").numpy()[0]
    del inp, pred
    return out.transpose(1, 2, 3, 0)  # [X,Y,Z,F] normalized


def fetch_frames(layout: FileLayout, traj: int, needed: set[int], log) -> dict[int, np.ndarray]:
    """Fetch needed frames grouped into contiguous runs."""
    frames: dict[int, np.ndarray] = {}
    ordered = sorted(needed)
    runs: list[tuple[int, int]] = []
    start = prev = ordered[0]
    for t in ordered[1:]:
        if t == prev + 1:
            prev = t
            continue
        runs.append((start, prev - start + 1))
        start = prev = t
    runs.append((start, prev - start + 1))
    tick = time.perf_counter()
    for run_start, count in runs:
        got = read_frames(layout, traj, run_start, count)
        for k in range(count):
            frames[run_start + k] = got[k]
    log(f"    fetched {len(needed)} frames in {time.perf_counter() - tick:.0f}s ({len(runs)} runs)")
    return frames


def evaluate_trajectory(
    model_name: str,
    model: Any,
    device: torch.device,
    layout: FileLayout,
    fname: str,
    traj: int,
    means: np.ndarray,
    stds: np.ndarray,
    rollout_steps: int,
    onestep_starts: list[int],
    log,
) -> list[dict[str, Any]]:
    needed: set[int] = set(range(0, HISTORY + rollout_steps))
    for s in onestep_starts:
        needed |= set(range(s, s + HISTORY + 1))
    frames = fetch_frames(layout, traj, needed, log)

    rows: list[dict[str, Any]] = []
    t_model = time.perf_counter()

    # ---- rollout ----
    state = np.stack([frames[t] for t in range(HISTORY)])
    state = (state - means) / stds
    for step in range(1, rollout_steps + 1):
        target_t = HISTORY + step - 1
        pred_norm = forward(model, state, device)
        pred_raw = pred_norm.astype(np.float64) * stds.astype(np.float64) + means.astype(np.float64)
        target = frames[target_t].astype(np.float64)
        rows.append(
            {
                "file": fname,
                "trajectory": traj,
                "mode": "rollout",
                "model": model_name,
                "t": target_t,
                "rollout_step": step,
                "mse": spatial_mse(pred_raw, target, 3).tolist(),
                "target_variance_ddof1": spatial_sample_variance(target, 3).tolist(),
                "pred_variance_ddof1": spatial_sample_variance(pred_raw, 3).tolist(),
            }
        )
        state = np.concatenate([state[1:], pred_norm[None].astype(np.float32)], axis=0)
    log(f"    rollout {rollout_steps} steps done {time.perf_counter() - t_model:.0f}s")

    # ---- one-step ----
    t_model = time.perf_counter()
    for s in onestep_starts:
        window = np.stack([frames[t] for t in range(s, s + HISTORY)])
        window = (window - means) / stds
        pred_norm = forward(model, window, device)
        pred_raw = pred_norm.astype(np.float64) * stds.astype(np.float64) + means.astype(np.float64)
        target = frames[s + HISTORY].astype(np.float64)
        rows.append(
            {
                "file": fname,
                "trajectory": traj,
                "mode": "onestep",
                "model": model_name,
                "t": s + HISTORY,
                "input_start": s,
                "mse": spatial_mse(pred_raw, target, 3).tolist(),
                "target_variance_ddof1": spatial_sample_variance(target, 3).tolist(),
                "pred_variance_ddof1": spatial_sample_variance(pred_raw, 3).tolist(),
            }
        )
    log(f"    one-step x{len(onestep_starts)} done {time.perf_counter() - t_model:.0f}s")
    del frames
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=list(MODEL_SPECS))
    parser.add_argument("--device", required=True, choices=["mps", "cpu"])
    parser.add_argument(
        "--pairs",
        required=True,
        help="semicolon list of file:traj, file by sorted index, e.g. '4:1;2:0'",
    )
    parser.add_argument(
        "--revision",
        required=True,
        help=(
            "exact Hugging Face commit sha to pin the checkpoint to (required, "
            "not optional -- see MANIFEST.md's checkpoint table for the shas "
            "used to produce this package's frozen fixtures). The fetched "
            "revision is verified against this value before evaluation; a "
            "mismatch raises rather than silently evaluating a drifted checkpoint."
        ),
    )
    parser.add_argument("--rollout-steps", type=int, default=30)
    parser.add_argument("--onestep-starts", required=True, help="comma list")
    parser.add_argument("--output-dir", type=Path, default=HERE / "results" / "models")
    parser.add_argument("--cache-dir", type=Path, default=HERE / "hf-cache")
    parser.add_argument("--threads", type=int, default=10)
    args = parser.parse_args()

    torch.set_num_threads(args.threads)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    def log(msg: str) -> None:
        print(msg, flush=True)

    means, stds = load_stats()
    cls_name, model_id = MODEL_SPECS[args.model]
    from the_well.benchmark import models as model_zoo

    # Enforce the pinned checkpoint revision before evaluating anything: fetch
    # the revision-specific model info and require its sha to equal the
    # requested --revision, mirroring lane-3/shear_checkpoint_eval.py's
    # drift check. A silent "main"-branch fetch could evaluate a checkpoint
    # the paper's frozen fixtures were never computed from.
    hf_info = requests.get(
        f"https://huggingface.co/api/models/{model_id}/revision/{args.revision}",
        timeout=30,
    )
    hf_info.raise_for_status()
    hf_info = hf_info.json()
    if hf_info.get("sha") != args.revision:
        raise AssertionError(
            f"checkpoint revision drift: requested {args.revision}, "
            f"Hugging Face resolved {hf_info.get('sha')}"
        )

    model = getattr(model_zoo, cls_name).from_pretrained(
        model_id, revision=args.revision, cache_dir=args.cache_dir
    ).to(device)
    model.eval()
    starts = [int(x) for x in args.onestep_starts.split(",")]

    layouts: dict[str, FileLayout] = {}
    pairs = [
        (int(p.split(":")[0]), int(p.split(":")[1])) for p in args.pairs.split(";")
    ]
    prov = {
        "access_date": time.strftime("%Y-%m-%d"),
        "model": args.model,
        "model_id": model_id,
        "hf_sha": hf_info.get("sha"),
        "params": sum(p.numel() for p in model.parameters()),
        "device": str(device),
        "torch": torch.__version__,
        "threads": args.threads,
        "rollout_steps": args.rollout_steps,
        "onestep_starts": starts,
        "normalization_mean": means.tolist(),
        "normalization_std": stds.tolist(),
        "sorted_files": SORTED_FILES,
        "pairs": pairs,
        "files": {},
    }
    started = time.perf_counter()
    for fi, traj in pairs:
        fname = SORTED_FILES[fi]
        out_path = args.output_dir / f"{args.model}_{fname.removesuffix('.hdf5')}_traj{traj}.json.gz"
        if out_path.exists():
            log(f"  SKIP {out_path.name}")
            continue
        if fname not in layouts:
            layouts[fname] = discover_layout(f"{BASE}/data/test/{fname}")
        layout = layouts[fname]
        prov["files"][fname] = {
            "url": layout.url,
            "bytes": layout.size,
            "etag": layout.etag,
        }
        log(f"  {args.model} on {fname} traj {traj} (batch index {2 * fi + traj})")
        rows = evaluate_trajectory(
            args.model, model, device, layout, fname, traj, means, stds,
            args.rollout_steps, starts, log,
        )
        with gzip.open(out_path, "wt", encoding="utf-8") as handle:
            json.dump({"rows": rows, "fields": FIELDS}, handle)
        log(f"  wrote {out_path.name}")
        if device.type == "mps":
            torch.mps.empty_cache()

    prov["wall_seconds"] = time.perf_counter() - started
    prov_path = args.output_dir / f"provenance_{args.model}.json"
    existing = json.loads(prov_path.read_text()) if prov_path.exists() else {"runs": []}
    existing["runs"].append(prov)
    prov_path.write_text(json.dumps(existing, indent=1) + "\n")
    log(f"DONE {args.model} in {prov['wall_seconds']:.0f}s")


if __name__ == "__main__":
    main()
