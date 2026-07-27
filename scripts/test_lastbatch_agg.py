"""Integration reproduction for The Well issue #78.

The test executes ``Trainer.validation_loop`` with two equal-size synthetic
batches, captures the time series handed to ``plot_all_time_metrics``, and
compares it with the independently computed all-batch mean.
"""

from __future__ import annotations

import json
from importlib.metadata import version
from types import MethodType, SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch

import the_well.benchmark.trainer.training as training
from the_well.benchmark.trainer.training import Trainer


class VRMSE:
    """Synthetic long-time metric whose values are encoded in y_pred."""

    def __call__(self, y_pred, y_ref, metadata):
        del y_ref, metadata
        return y_pred


class EmptyModel(torch.nn.Module):
    def forward(self, value):  # pragma: no cover - rollout is replaced below
        return value


def reproduce() -> dict[str, float | list[float] | str]:
    metric = VRMSE()
    trainer = Trainer.__new__(Trainer)
    trainer.model = EmptyModel()
    trainer.dset_metadata = SimpleNamespace(dataset_name="synthetic")
    trainer.validation_suite = [metric]
    trainer.loss_fn = metric
    trainer.short_validation_length = 2
    trainer.device = torch.device("cpu")
    trainer.enable_amp = False
    trainer.amp_type = torch.float32
    trainer.is_distributed = False
    trainer.make_rollout_videos = False
    trainer.viz_folder = "."
    trainer.formatter = None
    trainer.num_time_intervals = 1

    def rollout_model(self, model, batch, formatter, train=False):
        del self, model, formatter, train
        return batch["prediction"], batch["target"]

    trainer.rollout_model = MethodType(rollout_model, trainer)

    batches = [
        {
            "prediction": torch.tensor([[[1.0], [3.0]]]),
            "target": torch.zeros((1, 2, 1)),
        },
        {
            "prediction": torch.tensor([[[9.0], [11.0]]]),
            "target": torch.zeros((1, 2, 1)),
        },
    ]
    captured: dict[str, torch.Tensor] = {}

    def capture_time_logs(time_logs, metadata, folder, epoch):
        del metadata, folder, epoch
        captured.update({key: value.clone() for key, value in time_logs.items()})

    with (
        patch.object(training, "flatten_field_names", return_value=["field"]),
        patch.object(training, "validation_plots", []),
        patch.object(training, "plot_all_time_metrics", capture_time_logs),
        patch.object(training.tqdm, "tqdm", side_effect=lambda values: values),
    ):
        scalar_average, _ = trainer.validation_loop(batches, full=True)

    key = "synthetic/full_VRMSE_rollout"
    observed = captured[key].numpy()
    expected_last = np.array([9.0, 11.0])
    expected_all = np.array([5.0, 7.0])
    np.testing.assert_allclose(observed, expected_last)
    assert not np.allclose(observed, expected_all)
    np.testing.assert_allclose(scalar_average, 6.0)

    return {
        "the_well_version": version("the_well"),
        "first_batch": [1.0, 3.0],
        "second_batch": [9.0, 11.0],
        "library_time_curve": observed.tolist(),
        "correct_all_batch_curve": expected_all.tolist(),
        "curve_mean_difference": float(observed.mean() - expected_all.mean()),
        "library_scalar_loss_all_batches": float(scalar_average),
        "verdict": "ISSUE_78_TIME_CURVE_LAST_BATCH_REPRODUCED",
    }


def test_long_time_curve_aggregates_all_batches() -> None:
    result = reproduce()
    assert result["verdict"] == "ISSUE_78_TIME_CURVE_LAST_BATCH_REPRODUCED"


if __name__ == "__main__":
    print(json.dumps(reproduce(), indent=2, sort_keys=True))
