"""Single-official-file diagnostic for the pinned public shear-flow FNO.

The 2.10 GB official first test object is read through HTTP Range requests, one
trajectory at a time.  No full HDF5 object is stored.  Results are deliberately
labelled a subset diagnostic: one registry file cannot reproduce the 28-file
paper test cell, and official score surfaces currently conflict.

repro-harness note: this is the source repo's `lanes/lane-3/shear_checkpoint_eval.py`,
colocated here unmodified (no hardcoded paths to remove) so the paper's
device-precision cross-check (`sec7_boundaries.tex` sec:shear-gap,
`float32_robustness` in `numbers.json`) has the same exact-command Tier-2
reproduction path as the Rayleigh-Taylor pipeline in `rt_audit_pass.py` /
`rt_model_eval.py`. See MANIFEST.md "Tier 2 continued" for the exact command
and pinned revisions. Not re-executed while assembling this harness (see
DIGEST.md open blockers).
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path
from typing import Any

import fsspec
import h5py
import numpy as np
import requests
import torch
import yaml

from independent_metrics import spatial_mse, spatial_sample_variance, vrmse
from the_well.benchmark.models import FNO


MODEL_ID = "polymathic-ai/FNO-shear_flow"
MODEL_REVISION = "9ff091f47bb547c3a91b6ee3804028e4f6609888"
DATASET_CARD_REVISION = "fc867f856f306905cf94c1f5df978cc518a2048c"
REMOTE_URL = (
    "https://sdsc-users.flatironinstitute.org/~polymathic/data/the_well/"
    "datasets/shear_flow/data/test/shear_flow_Reynolds_1e5_Schmidt_2e0.hdf5"
)
STATS_URL = (
    "https://sdsc-users.flatironinstitute.org/~polymathic/data/the_well/"
    "datasets/shear_flow/stats.yaml"
)
PAPER_TEST_VRMSE = 1.189
HF_DATASET_CARD_TEST_VRMSE = 0.1567
HF_MODEL_CARD_VALIDATION_VRMSE = 0.4450
HISTORY = 4
FIELD_ORDER = ["tracer", "pressure", "velocity_x", "velocity_y"]


def _select_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    return torch.device(requested)


def _load_stats() -> tuple[np.ndarray, np.ndarray]:
    response = requests.get(STATS_URL, timeout=30)
    response.raise_for_status()
    stats = yaml.safe_load(response.text)
    means = np.asarray(
        [
            stats["mean"]["tracer"],
            stats["mean"]["pressure"],
            *stats["mean"]["velocity"],
        ],
        dtype=np.float32,
    )
    stds = np.asarray(
        [
            stats["std"]["tracer"],
            stats["std"]["pressure"],
            *stats["std"]["velocity"],
        ],
        dtype=np.float32,
    )
    stds = np.maximum(stds, 1e-4)
    return means, stds


def _read_trajectory(h5: h5py.File, trajectory: int, stop: int) -> np.ndarray:
    frames = stop
    x_size = int(h5["dimensions/x"].shape[0])
    y_size = int(h5["dimensions/y"].shape[0])
    values = np.empty((frames, x_size, y_size, 4), dtype=np.float32)
    values[..., 0] = h5["t0_fields/tracer"][trajectory, :stop]
    values[..., 1] = h5["t0_fields/pressure"][trajectory, :stop]
    values[..., 2:4] = h5["t1_fields/velocity"][trajectory, :stop]
    return values


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def evaluate(
    output_dir: Path,
    cache_dir: Path,
    device_name: str,
    trajectory_limit: int,
    window_stride: int,
    max_windows_per_trajectory: int | None,
    batch_size: int,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)
    device = _select_device(device_name)
    means, stds = _load_stats()

    head = requests.head(REMOTE_URL, timeout=30)
    head.raise_for_status()
    model_info = requests.get(
        f"https://huggingface.co/api/models/{MODEL_ID}/revision/{MODEL_REVISION}?blobs=true",
        timeout=30,
    )
    model_info.raise_for_status()
    model_info_json = model_info.json()
    if model_info_json["sha"] != MODEL_REVISION:
        raise AssertionError("model revision drift")

    started = time.perf_counter()
    model = FNO.from_pretrained(
        MODEL_ID,
        revision=MODEL_REVISION,
        cache_dir=cache_dir,
    ).to(device)
    model.eval()
    trainable = sum(parameter.numel() for parameter in model.parameters())

    rows: list[dict[str, Any]] = []
    hdf5_metadata: dict[str, Any] = {}
    with fsspec.open(
        REMOTE_URL,
        "rb",
        block_size=8 * 1024 * 1024,
        cache_type="bytes",
    ) as remote:
        with h5py.File(remote, "r") as h5:
            hdf5_metadata = {
                key: value.tolist() if hasattr(value, "tolist") else value
                for key, value in h5.attrs.items()
            }
            t0_names = [str(value) for value in h5["t0_fields"].attrs["field_names"]]
            t1_names = [str(value) for value in h5["t1_fields"].attrs["field_names"]]
            if t0_names != ["tracer", "pressure"] or t1_names != ["velocity"]:
                raise AssertionError(
                    f"unexpected field order: t0={t0_names}, t1={t1_names}"
                )
            available_trajectories = int(h5.attrs["n_trajectories"])
            selected_trajectories = min(trajectory_limit, available_trajectories)
            for trajectory in range(selected_trajectories):
                total_frames = int(h5["dimensions/time"].shape[0])
                starts = list(range(0, total_frames - HISTORY, window_stride))
                if max_windows_per_trajectory is not None:
                    starts = starts[:max_windows_per_trajectory]
                stop = max(starts) + HISTORY + 1
                raw = _read_trajectory(h5, trajectory, stop)
                for batch_start in range(0, len(starts), batch_size):
                    batch_starts = starts[batch_start : batch_start + batch_size]
                    inputs = np.stack(
                        [raw[start : start + HISTORY] for start in batch_starts]
                    )
                    normalized = (inputs - means) / stds
                    formatted = normalized.transpose(0, 1, 4, 2, 3).reshape(
                        len(batch_starts), HISTORY * 4, raw.shape[1], raw.shape[2]
                    )
                    tensor = torch.from_numpy(formatted).to(device)
                    with torch.inference_mode():
                        prediction_normalized = model(tensor)
                    prediction_normalized_np = (
                        prediction_normalized.detach()
                        .to("cpu")
                        .numpy()
                        .transpose(0, 2, 3, 1)
                    )
                    prediction_raw = prediction_normalized_np * stds + means
                    targets_raw = np.stack(
                        [raw[start + HISTORY] for start in batch_starts]
                    )
                    targets_normalized = (targets_raw - means) / stds

                    raw_mse = spatial_mse(prediction_raw, targets_raw, 2)
                    raw_variance = spatial_sample_variance(targets_raw, 2)
                    raw_vrmse = vrmse(
                        prediction_raw, targets_raw, 2, eps=1e-7
                    )
                    normalized_vrmse = vrmse(
                        prediction_normalized_np,
                        targets_normalized,
                        2,
                        eps=1e-7,
                    )
                    for local_index, start in enumerate(batch_starts):
                        for field_index, field in enumerate(FIELD_ORDER):
                            rows.append(
                                {
                                    "trajectory": trajectory,
                                    "input_start": start,
                                    "target_step": start + HISTORY,
                                    "field": field,
                                    "raw_mse": float(raw_mse[local_index, field_index]),
                                    "raw_target_variance_ddof1": float(
                                        raw_variance[local_index, field_index]
                                    ),
                                    "independent_raw_vrmse": float(
                                        raw_vrmse[local_index, field_index]
                                    ),
                                    "independent_normalized_vrmse": float(
                                        normalized_vrmse[local_index, field_index]
                                    ),
                                }
                            )
                    del tensor, prediction_normalized
                del raw

    _write_csv(output_dir / "shear_per_window.csv", rows)
    raw_values = np.asarray([row["independent_raw_vrmse"] for row in rows])
    normalized_values = np.asarray(
        [row["independent_normalized_vrmse"] for row in rows]
    )
    n_samples = len(rows) // len(FIELD_ORDER)
    field_means = {
        field: float(
            np.mean(
                [
                    row["independent_raw_vrmse"]
                    for row in rows
                    if row["field"] == field
                ]
            )
        )
        for field in FIELD_ORDER
    }
    measured = float(raw_values.mean())
    summary = {
        "access_date": "2026-07-12",
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "model_trainable_parameter_numel": trainable,
        "dataset": "shear_flow",
        "dataset_card_revision": DATASET_CARD_REVISION,
        "remote_url": REMOTE_URL,
        "remote_bytes": int(head.headers["Content-Length"]),
        "remote_etag": head.headers.get("ETag"),
        "remote_last_modified": head.headers.get("Last-Modified"),
        "access_mode": "HTTP Range; full HDF5 not materialized",
        "hdf5_root_attributes": hdf5_metadata,
        "device": str(device),
        "torch_version": torch.__version__,
        "history_steps": HISTORY,
        "window_stride": window_stride,
        "trajectory_limit": trajectory_limit,
        "max_windows_per_trajectory": max_windows_per_trajectory,
        "n_evaluated_samples": n_samples,
        "n_field_scores": len(rows),
        "field_order": FIELD_ORDER,
        "normalization_mean": means.tolist(),
        "normalization_std": stds.tolist(),
        "primary_metric_space": (
            "raw physical space; matches the_well 1.2.0 Trainer denormalization "
            "before validation metrics"
        ),
        "independent_raw_vrmse_mean": measured,
        "sensitivity_normalized_space_vrmse_mean": float(normalized_values.mean()),
        "field_mean_raw_vrmse": field_means,
        "paper_table2_test_vrmse": PAPER_TEST_VRMSE,
        "diff_subset_minus_paper_table2": measured - PAPER_TEST_VRMSE,
        "hf_dataset_card_test_vrmse": HF_DATASET_CARD_TEST_VRMSE,
        "diff_subset_minus_hf_dataset_card": measured
        - HF_DATASET_CARD_TEST_VRMSE,
        "hf_model_card_validation_vrmse": HF_MODEL_CARD_VALIDATION_VRMSE,
        "diff_subset_minus_hf_model_card_validation": measured
        - HF_MODEL_CARD_VALIDATION_VRMSE,
        "comparability_verdict": "NOT_PAPER_COMPARABLE_SINGLE_FILE_DIAGNOSTIC",
        "comparability_limits": [
            "one Flatiron registry test file versus the paper's 28-file test split",
            "official test surfaces conflict: paper Table 2 is 1.189 versus HF dataset card 0.1567; model-card 0.4450 matches paper Table 5 validation, not test",
            "checkpoint does not bind an exact historical data/code revision",
            "issue #49 reports that some HF baselines reflect normalization or pre-release orientation states, without identifying this shear checkpoint specifically",
        ],
        "runtime_seconds": time.perf_counter() - started,
        "output": "shear_per_window.csv",
    }
    (output_dir / "shear_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return summary


def refresh_existing_summary(output_dir: Path) -> dict[str, Any]:
    """Apply source-label corrections without repeating expensive inference."""
    path = output_dir / "shear_summary.json"
    summary = json.loads(path.read_text(encoding="utf-8"))
    measured = float(summary["independent_raw_vrmse_mean"])
    summary["dataset_card_revision"] = summary.pop(
        "dataset_hf_revision", DATASET_CARD_REVISION
    )
    summary["primary_metric_space"] = (
        "raw physical space; matches the_well 1.2.0 Trainer denormalization "
        "before validation metrics"
    )
    if "independent_normalized_vrmse_mean" in summary:
        summary["sensitivity_normalized_space_vrmse_mean"] = summary.pop(
            "independent_normalized_vrmse_mean"
        )
    summary.pop("hf_model_card_test_vrmse", None)
    summary.pop("diff_subset_minus_hf_model_card_test", None)
    summary["hf_model_card_validation_vrmse"] = HF_MODEL_CARD_VALIDATION_VRMSE
    summary["diff_subset_minus_hf_model_card_validation"] = (
        measured - HF_MODEL_CARD_VALIDATION_VRMSE
    )
    summary["comparability_limits"] = [
        "one Flatiron registry test file versus the paper's 28-file test split",
        "official test surfaces conflict: paper Table 2 is 1.189 versus HF dataset card 0.1567; model-card 0.4450 matches paper Table 5 validation, not test",
        "checkpoint does not bind an exact historical data/code revision",
        "issue #49 reports that some HF baselines reflect normalization or pre-release orientation states, without identifying this shear checkpoint specifically",
    ]
    path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("results/shear"))
    parser.add_argument("--cache-dir", type=Path, default=Path("hf-cache"))
    parser.add_argument("--device", default="auto", choices=["auto", "mps", "cpu"])
    parser.add_argument("--trajectory-limit", type=int, default=1)
    parser.add_argument("--window-stride", type=int, default=1)
    parser.add_argument("--max-windows-per-trajectory", type=int)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--refresh-summary-only", action="store_true")
    args = parser.parse_args()
    if args.refresh_summary_only:
        print(
            json.dumps(
                refresh_existing_summary(args.output_dir), indent=2, sort_keys=True
            )
        )
        return
    summary = evaluate(
        args.output_dir,
        args.cache_dir,
        args.device,
        args.trajectory_limit,
        args.window_stride,
        args.max_windows_per_trajectory,
        args.batch_size,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
