#!/usr/bin/env python3
"""DEFINITIVE end-to-end validation of our RB evaluator against the library's own path.

Fixes the harness defect found in same_window_test.py: WellDataset(use_normalization=True)
is a NO-OP unless normalization_type is also passed (datasets.py:332 -> self.norm = None).
With ZScoreNormalization supplied, the library normalizes exactly as our evaluator does.

For each probed window this compares, on the SAME (file, trajectory, start):
  1. the model input tensor      (data loading + normalization + channel layout)
  2. the target tensor
  3. the model output tensor
  4. the final VRMSE

Agreement on all four validates our evaluator end to end. Disagreement anywhere voids the
RB results.

Also records which dataset class semantics apply: plain WellDataset returns ABSOLUTE next
states (is_delta False in the trainer); DeltaWellDataset (datasets.py:995) is the opt-in
residual variant. Our evaluator assumes absolute, matching plain WellDataset.
"""
import sys, os, json
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
HARNESS = os.path.dirname(HERE)
sys.path.insert(0, HERE)
from independent_metrics import spatial_mse, spatial_sample_variance

from the_well.data import WellDataset
from the_well.data.normalization import ZScoreNormalization
from the_well.data.data_formatter import DefaultChannelsFirstFormatter
from the_well.benchmark.metrics.spatial import VRMSE
import the_well.benchmark.models as model_zoo
import rb_model_eval as OURS

REPO = "polymathic-ai/UNetClassic-rayleigh_benard"
REV = "b72b1213dade391e4620476d1410c1bbe7103c1d"
IDXS = [0, 4899, 9799, 20000, 34299]
TOL = 1e-5


def vrmse_raw(pred_norm, tgt_norm, means, stds, ns):
    """VRMSE in raw physical units -- the convention the paper's reported cells use."""
    p = np.asarray(pred_norm, np.float64) * stds + means
    t = np.asarray(tgt_norm, np.float64) * stds + means
    return float(np.mean(np.sqrt(spatial_mse(p, t, ns) / (spatial_sample_variance(t, ns) + 1e-7))))


def reldiff(a, b):
    a = np.asarray(a, np.float64); b = np.asarray(b, np.float64)
    if a.shape != b.shape:
        return float("inf")
    return float(np.abs(a - b).max() / max(np.abs(a).max(), 1e-30))


def main():
    ds = WellDataset(
        well_base_path="hf://datasets/polymathic-ai/",
        well_dataset_name="rayleigh_benard", well_split_name="test",
        n_steps_input=4, n_steps_output=1,
        use_normalization=True,
        normalization_type=ZScoreNormalization,   # <-- the piece that was missing
    )
    assert ds.norm is not None, "normalization still not active -- abort"
    print(f"dataset len={len(ds)}  norm={type(ds.norm).__name__}  "
          f"core_fields={ds.core_field_names}")
    print(f"delta semantics: plain WellDataset -> absolute next state (trainer is_delta=False)\n")

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model = model_zoo.UNetClassic.from_pretrained(
        REPO, revision=REV, cache_dir=os.path.join(HARNESS, "hf-cache")
    ).to(device).eval()
    fmt = DefaultChannelsFirstFormatter(ds.metadata)
    means, stds = OURS.load_stats()

    results, worst = [], 0.0
    for idx in IDXS:
        _, file_idx, sample_idx, time_idx, dt = ds._load_one_sample(idx)
        fname = os.path.basename(str(ds.files_paths[file_idx]))

        sample = ds[idx]
        batch = {k: (v.unsqueeze(0) if torch.is_tensor(v) else v) for k, v in sample.items()}
        inputs, y_ref = fmt.process_input(batch)
        with torch.inference_mode():
            lib_out = model(*[x.to(device) for x in inputs]).cpu()
        lib_vrmse = VRMSE.eval(fmt.process_output_channel_last(lib_out),
                               y_ref, ds.metadata).mean().item()

        frames = OURS.fetch_frames(OURS.HF + "data/test/" + fname, int(sample_idx),
                                   int(time_idx), int(time_idx) + OURS.HISTORY)
        win = (frames[:OURS.HISTORY] - means) / stds
        our_in = torch.from_numpy(
            win.transpose(0, 3, 1, 2).reshape(1, OURS.HISTORY * len(OURS.FIELDS),
                                              win.shape[1], win.shape[2]).astype(np.float32))
        with torch.inference_mode():
            our_out = model(our_in.to(device)).cpu()
        tn = (frames[OURS.HISTORY] - means) / stds
        pn = our_out.numpy()[0].transpose(1, 2, 0)
        our_vrmse = float(np.mean(np.sqrt(
            spatial_mse(pn.astype(np.float64), tn.astype(np.float64), 2)
            / (spatial_sample_variance(tn.astype(np.float64), 2) + 1e-7))))

        lr = y_ref.numpy()
        while lr.ndim > 3:
            lr = lr[0]
        lo = lib_out.numpy()[0].transpose(1, 2, 0)
        # RAW-PHYSICAL units: the convention every cell this paper reports is computed in.
        our_raw = vrmse_raw(pn, tn, means, stds, 2)
        lib_raw = vrmse_raw(lo, lr, means, stds, 2)
        d_vraw = abs(lib_raw - our_raw) / max(abs(lib_raw), 1e-30)
        d_in = reldiff(inputs[0].numpy(), our_in.numpy())
        d_tg = reldiff(lr, tn)
        d_out = reldiff(lib_out.numpy(), our_out.numpy())
        d_v = abs(lib_vrmse - our_vrmse) / max(abs(lib_vrmse), 1e-30)
        worst = max(worst, d_in, d_tg, d_out, d_v, d_vraw)

        print(f"idx {idx:6d}  {fname.replace('rayleigh_benard_','')[:34]:34s} traj={sample_idx} t={time_idx}")
        print(f"    input {d_in:9.2e}   target {d_tg:9.2e}   output {d_out:9.2e}")
        print(f"    VRMSE  z-scored lib={lib_vrmse:.6f} ours={our_vrmse:.6f} rel={d_v:.2e} | "
              f"raw lib={lib_raw:.6f} ours={our_raw:.6f} rel={d_vraw:.2e}  "
              f"{'OK' if max(d_in,d_tg,d_out,d_v,d_vraw) < TOL else '*** MISMATCH ***'}\n")
        results.append({"idx": idx, "file": fname, "traj": int(sample_idx),
                        "start": int(time_idx), "lib_vrmse": lib_vrmse,
                        "our_vrmse": our_vrmse, "lib_vrmse_raw": lib_raw,
                        "our_vrmse_raw": our_raw, "d_input": d_in, "d_target": d_tg,
                        "d_output": d_out, "d_vrmse": d_v, "d_vrmse_raw": d_vraw})

    print("=== VERDICT ===")
    print(f"  windows: {len(results)}   worst relative discrepancy: {worst:.3e}   tol {TOL:g}")
    print(f"  -> RB evaluator {'VALIDATED end to end' if worst < TOL else 'DISAGREES -- RESULTS VOID'}")
    json.dump({"tol": TOL, "worst": worst, "validated": bool(worst < TOL),
               "windows": results},
              open(os.path.join(HARNESS, "fixtures/gate3/rb_evaluator_validation.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
