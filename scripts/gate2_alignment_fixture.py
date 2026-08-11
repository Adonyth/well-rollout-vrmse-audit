"""Fixture test: my streaming mirror == real WellDataset/formatter/rollout_model.

Builds a tiny well-schema HDF5 (schema copied attribute-for-attribute from the
real RT test object), then asserts:

  A. WellDataset one-step sample (start s) == frames[s:s+4] -> target frame s+4.
  B. DefaultChannelsFirstFormatter channel order == frames_to_channels_first.
  C. Trainer.rollout_model autoregressive predictions == my mirror's rollout
     (identity normalization), elementwise.
  D. rollout_model with a Z-score dset_norm == my mirror with the same
     mean/std (normalized-feedback equivalence), elementwise within float32.

It needs no network and no checkpoint: the fixture is synthesized locally and the
"model" is a deterministic toy map, so what is compared is the DATA PATH and the
ROLLOUT BOOKKEEPING -- window alignment, channel order, autoregressive feedback --
rather than any model weights. That is precisely the multi-step path Gate 3 cannot
reach, since Gate 3 drives the library at n_steps_output=1.

Run:   python3 scripts/gate2_alignment_fixture.py [--json OUT]
Needs: the_well, torch, h5py, yaml (the Tier-2 stack in env-lock-full.txt).
"""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

import h5py
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
from rt_model_eval import frames_to_channels_first  # noqa: E402

GRID = 8
ROLLOUT = 31          # spans BOTH disputed windows (6-12 and 13-30)
T_TOTAL = 4 + ROLLOUT  # history + horizon
N_TRAJ = 2
HISTORY = 4


def build_fixture(path: str) -> np.ndarray:
    rng = np.random.default_rng(7)
    density = rng.normal(1.0, 0.3, (N_TRAJ, T_TOTAL, GRID, GRID, GRID)).astype(np.float32)
    velocity = rng.normal(0.0, 0.01, (N_TRAJ, T_TOTAL, GRID, GRID, GRID, 3)).astype(np.float32)
    with h5py.File(path, "w") as f:
        f.attrs.update(
            {
                "At": 0.75,
                "dataset_name": "rt_fixture",
                "grid_type": "cartesian",
                "n_spatial_dims": 3,
                "n_trajectories": N_TRAJ,
                "simulation_parameters": ["At"],
            }
        )
        dims = f.create_group("dimensions")
        dims.attrs["spatial_dims"] = ["x", "y", "z"]
        time = dims.create_dataset("time", data=np.arange(T_TOTAL, dtype=np.float32))
        time.attrs.update({"sample_varying": False, "time_varying": True})
        for d in "xyz":
            ds = dims.create_dataset(d, data=np.linspace(0, 1, GRID, dtype=np.float32))
            ds.attrs.update({"sample_varying": False, "time_varying": False})
        t0 = f.create_group("t0_fields")
        t0.attrs["field_names"] = ["density"]
        ds = t0.create_dataset("density", data=density)
        ds.attrs.update(
            {"dim_varying": [True, True, True], "sample_varying": True, "time_varying": True}
        )
        t1 = f.create_group("t1_fields")
        t1.attrs["field_names"] = ["velocity"]
        ds = t1.create_dataset("velocity", data=velocity)
        ds.attrs.update(
            {"dim_varying": [True, True, True], "sample_varying": True, "time_varying": True}
        )
        t2 = f.create_group("t2_fields")
        t2.attrs["field_names"] = []
        sc = f.create_group("scalars")
        sc.attrs["field_names"] = ["At"]
        at = sc.create_dataset("At", data=0.75)
        at.attrs.update({"sample_varying": False, "time_varying": False})
        bcs = f.create_group("boundary_conditions")
        for name, dim, kind in [
            ("x_periodic", "x", "PERIODIC"),
            ("y_periodic", "y", "PERIODIC"),
            ("z_wall", "z", "WALL"),
        ]:
            g = bcs.create_group(name)
            g.attrs.update(
                {
                    "associated_dims": [dim],
                    "associated_fields": [],
                    "bc_type": kind,
                    "sample_varying": False,
                    "time_varying": False,
                }
            )
            g.create_dataset("mask", data=np.ones(GRID, dtype=bool))
    stats = {
        "mean": {
            "density": float(density.mean()),
            "velocity": [float(velocity[..., i].mean()) for i in range(3)],
        },
        "std": {
            "density": float(density.std()),
            "velocity": [float(velocity[..., i].std()) for i in range(3)],
        },
        "mean_delta": {"density": 0.0, "velocity": [0.0, 0.0, 0.0]},
        "std_delta": {"density": 1.0, "velocity": [1.0, 1.0, 1.0]},
        "rms": {
            "density": float(np.sqrt((density**2).mean())),
            "velocity": [float(np.sqrt((velocity[..., i] ** 2).mean())) for i in range(3)],
        },
        "rms_delta": {"density": 1.0, "velocity": [1.0, 1.0, 1.0]},
    }
    import yaml

    Path(path).parent.joinpath("stats.yaml").write_text(yaml.safe_dump(stats))
    frames = np.concatenate([density[..., None], velocity], axis=-1)  # [N,T,X,Y,Z,4]
    return frames


class TinyModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        torch.manual_seed(3)
        self.conv = torch.nn.Conv3d(HISTORY * 4, 4, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


def my_rollout(
    model: torch.nn.Module,
    frames: np.ndarray,
    steps: int,
    means: np.ndarray,
    stds: np.ndarray,
) -> np.ndarray:
    """Mirror of rt_model_eval rollout (normalized-space feedback)."""
    state = (frames[:HISTORY] - means) / stds
    preds = []
    for _ in range(steps):
        inp = torch.nan_to_num(torch.from_numpy(frames_to_channels_first(state)))
        with torch.inference_mode():
            out = model(inp).numpy()[0].transpose(1, 2, 3, 0)
        preds.append(out * stds + means)
        state = np.concatenate([state[1:], out[None]], axis=0)
    return np.stack(preds)


def main() -> None:
    from torch.utils.data import DataLoader

    from the_well.benchmark.trainer.training import Trainer
    from the_well.data.data_formatter import DefaultChannelsFirstFormatter
    from the_well.data.datasets import WellDataset

    tmp = tempfile.mkdtemp(prefix="rtfix_")
    os.makedirs(f"{tmp}/data/train", exist_ok=True)
    frames = build_fixture(f"{tmp}/data/train/rt_fixture.hdf5")

    # --- A: one-step window alignment ---
    dset = WellDataset(
        path=f"{tmp}/data/train",
        n_steps_input=HISTORY,
        n_steps_output=1,
        use_normalization=False,
    )
    n_windows = T_TOTAL - HISTORY
    assert len(dset) == N_TRAJ * n_windows, len(dset)
    # EVERY window, and the difference is MEASURED. An earlier version sampled three
    # windows and then assigned a_max = 0.0 as a literal -- so the recorded "measurement"
    # was a constant the script typed rather than anything it observed.
    a_max = 0.0
    n_checked = 0
    for idx in range(len(dset)):
        traj, start = divmod(idx, n_windows)
        sample = dset[idx]
        got_in = sample["input_fields"].numpy()
        got_out = sample["output_fields"].numpy()
        want_in = frames[traj, start : start + HISTORY]
        want_out = frames[traj, start + HISTORY : start + HISTORY + 1]
        assert got_in.shape == want_in.shape and got_out.shape == want_out.shape, idx
        a_max = max(a_max, float(np.abs(got_in - want_in).max()),
                    float(np.abs(got_out - want_out).max()))
        n_checked += 1
    assert n_checked == len(dset), (n_checked, len(dset))
    assert a_max == 0.0, a_max
    print(f"PASS A: WellDataset one-step window alignment over all {n_checked} windows "
          f"(input s..s+3 -> target s+4), max abs diff {a_max:.3e}")

    # --- B: formatter channel order ---
    formatter = DefaultChannelsFirstFormatter(dset.metadata)
    sample = dset[0]
    batch = {
        "input_fields": sample["input_fields"][None],
        "output_fields": sample["output_fields"][None],
    }
    (inp,), _ = formatter.process_input(batch)
    mine = frames_to_channels_first(frames[0, :HISTORY])
    np.testing.assert_array_equal(inp.numpy(), mine)
    b_max = float(np.abs(inp.numpy() - mine).max())
    print(f"PASS B: formatter channel order == frames_to_channels_first, "
          f"max abs diff {b_max:.3e}")

    # --- C: rollout equivalence, identity normalization ---
    rollout_dset = WellDataset(
        path=f"{tmp}/data/train",
        n_steps_input=HISTORY,
        n_steps_output=1,
        use_normalization=False,
        full_trajectory_mode=True,
    )
    loader = DataLoader(rollout_dset, batch_size=1, shuffle=False)
    model = TinyModel()
    model.eval()
    trainer = object.__new__(Trainer)
    trainer.device = torch.device("cpu")
    trainer.max_rollout_steps = ROLLOUT
    trainer.is_delta = False
    trainer.formatter = formatter

    batch = next(iter(loader))
    with torch.inference_mode():
        y_pred, y_ref = trainer.rollout_model(model, dict(batch), formatter, train=False)
    ident = (np.zeros(4, np.float32), np.ones(4, np.float32))
    mine = my_rollout(model, frames[0], ROLLOUT, *ident)
    np.testing.assert_array_equal(
        y_ref.numpy()[0], frames[0, HISTORY : HISTORY + ROLLOUT]
    )
    np.testing.assert_allclose(y_pred.numpy()[0], mine, rtol=1e-6, atol=1e-7)
    c_max = float(np.abs(y_pred.numpy()[0] - mine).max())
    c_shape = list(y_pred.numpy()[0].shape)
    c_ref_shape = list(y_ref.numpy()[0].shape)
    c_per_step = [float(np.abs(y_pred.numpy()[0][t] - mine[t]).max())
                  for t in range(mine.shape[0])]
    print(f"PASS C: Trainer.rollout_model == mirror rollout (identity norm), "
          f"{ROLLOUT} steps, max abs diff {c_max:.3e}")

    # --- D: rollout equivalence under Z-score normalization ---
    # Library semantics: val/test datasets return RAW samples (datamodule
    # constructs them without use_normalization); the Trainer normalizes and
    # denormalizes around the model using dset_norm = train_dataset.norm.
    import yaml

    from the_well.data.normalization import ZScoreNormalization

    stats = yaml.safe_load((Path(tmp) / "data/train/stats.yaml").read_text())
    trainer.dset_norm = ZScoreNormalization(stats, ["density", "velocity"], [])
    mean_d = trainer.dset_norm.means["density"].numpy().reshape(-1)
    mean_v = trainer.dset_norm.means["velocity"].numpy().reshape(-1)
    std_d = trainer.dset_norm.stds["density"].numpy().reshape(-1)
    std_v = trainer.dset_norm.stds["velocity"].numpy().reshape(-1)
    means = np.concatenate([mean_d, mean_v]).astype(np.float32)
    stds = np.concatenate([std_d, std_v]).astype(np.float32)
    loader = DataLoader(rollout_dset, batch_size=1, shuffle=False)
    batch = next(iter(loader))
    with torch.inference_mode():
        y_pred, y_ref = trainer.rollout_model(model, dict(batch), formatter, train=False)
    mine = my_rollout(model, frames[0], ROLLOUT, means, stds)
    np.testing.assert_allclose(y_pred.numpy()[0], mine, rtol=2e-5, atol=1e-6)
    d_max = float(np.abs(y_pred.numpy()[0] - mine).max())
    print("PASS D: rollout under Z-score norm == mirror (normalized feedback), "
          f"max abs diff {d_max:.3e}")

    return {
        "gate": "gate2_alignment_fixture",
        "description": ("library WellDataset/formatter/Trainer.rollout_model vs this "
                        "audit's mirror on a synthesized well-schema fixture; no network, "
                        "no checkpoint, deterministic toy model"),
        "grid": GRID, "n_trajectories": N_TRAJ, "n_frames": T_TOTAL,
        "history": HISTORY, "rollout_steps": ROLLOUT,
        # MEASURED differences, not literal "exact" strings. An earlier version recorded
        # the word "exact" for A/B/C, which meant the verifier could only check that a
        # string had not changed -- it could not distinguish a real run from a fabricated
        # file. Shapes are recorded too, so the verifier can confirm the compared tensors
        # had the geometry the gate claims.
        "checks": {
            "A_window_alignment_max_abs_diff": a_max,
            "B_channel_order_max_abs_diff": b_max,
            "C_rollout_identity_norm_max_abs_diff": c_max,
            "D_rollout_zscore_norm_max_abs_diff": d_max,
        },
        "compared_shapes": {"rollout_pred": c_shape, "rollout_ref": c_ref_shape},
        "n_windows_checked": n_checked,
        # per-step maxima, so a claim about how the error behaves ACROSS the horizon is
        # backed by the series rather than by one aggregate
        "identity_per_step_max_abs_diff": c_per_step,
    }


if __name__ == "__main__":
    import argparse, json as _json
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", metavar="OUT", help="write the frozen result fixture")
    a = ap.parse_args()
    res = main()
    if a.json and res:
        with open(a.json, "w", encoding="utf-8") as fh:
            _json.dump(res, fh, indent=1, sort_keys=True)
        print(f"wrote {a.json}")
