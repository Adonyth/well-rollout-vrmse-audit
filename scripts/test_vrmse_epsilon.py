"""Executable reproduction for The Well issue #75.

Run directly to print the numeric contrast, or under pytest.  The test calls the
installed library for the audited value and uses ``independent_metrics`` for the
correct caller-epsilon implementation.
"""

from __future__ import annotations

import json
from importlib.metadata import version
from types import SimpleNamespace

import numpy as np
import torch

from independent_metrics import vrmse as independent_vrmse
from the_well.benchmark.metrics.spatial import VRMSE


REQUESTED_EPSILON = 1e-5
LIBRARY_FALLBACK_EPSILON = 1e-7


def reproduce() -> dict[str, float | str]:
    meta = SimpleNamespace(n_spatial_dims=1)
    target = torch.zeros((1, 4, 1), dtype=torch.float64)
    prediction = torch.full_like(target, 1e-2)

    library = VRMSE.eval(prediction, target, meta, eps=REQUESTED_EPSILON)
    correct = independent_vrmse(
        prediction.numpy(),
        target.numpy(),
        1,
        eps=REQUESTED_EPSILON,
    )
    fallback = independent_vrmse(
        prediction.numpy(),
        target.numpy(),
        1,
        eps=LIBRARY_FALLBACK_EPSILON,
    )

    library_value = float(library.item())
    correct_value = float(correct.item())
    fallback_value = float(fallback.item())

    np.testing.assert_allclose(library_value, fallback_value, rtol=1e-12)
    assert not np.isclose(library_value, correct_value, rtol=1e-6)
    assert library_value / correct_value > 9.9

    # Critical nuance: default callers are unaffected because the omitted value
    # and the callee fallback are both 1e-7.
    library_default = float(VRMSE.eval(prediction, target, meta).item())
    correct_default = float(
        independent_vrmse(
            prediction.numpy(),
            target.numpy(),
            1,
            eps=LIBRARY_FALLBACK_EPSILON,
        ).item()
    )
    np.testing.assert_allclose(library_default, correct_default, rtol=1e-12)

    return {
        "the_well_version": version("the_well"),
        "requested_epsilon": REQUESTED_EPSILON,
        "silently_used_epsilon": LIBRARY_FALLBACK_EPSILON,
        "library_vrmse": library_value,
        "correct_vrmse": correct_value,
        "amplification_ratio": library_value / correct_value,
        "library_default_vrmse": library_default,
        "correct_default_vrmse": correct_default,
        "verdict": "ISSUE_75_REPRODUCED_NONDEFAULT_EPSILON",
    }


def test_vrmse_epsilon_is_forwarded() -> None:
    result = reproduce()
    assert result["verdict"] == "ISSUE_75_REPRODUCED_NONDEFAULT_EPSILON"


if __name__ == "__main__":
    print(json.dumps(reproduce(), indent=2, sort_keys=True))
