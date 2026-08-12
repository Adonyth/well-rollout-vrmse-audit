#!/usr/bin/env python3
"""End-to-end validation of the RT evaluator -- the one carrying the paper's core numbers.

Same construction as gate3_recheck_rb.py, applied to rayleigh_taylor_instability.
This is the check that matters most: the paper's floor sweep, class split and rollout cells
are all computed from rt_model_eval.py's mse/variance rows. If its forward path disagrees
with the library's own, those numbers are wrong.

Note it also cross-validates two independent data sources: the library streams the test split
from HuggingFace, our evaluator reads the SDSC mirror. Agreement therefore covers both the
protocol and the mirror's fidelity.
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
import rt_model_eval as OURS
from fast_reader import discover_layout, read_frames

REPO = "polymathic-ai/UNetClassic-rayleigh_taylor_instability"
N_PROBE = int(os.environ.get("N_PROBE", "4"))
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
        well_dataset_name="rayleigh_taylor_instability", well_split_name="test",
        n_steps_input=4, n_steps_output=1,
        use_normalization=True, normalization_type=ZScoreNormalization,
    )
    assert ds.norm is not None
    print(f"dataset len={len(ds)}  core_fields={ds.core_field_names}")

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    prov = json.load(open(os.path.join(
        HARNESS, "fixtures/models/provenance_UNetClassic.json")))
    revision = prov["runs"][0]["hf_sha"]          # the exact pin the paper's rows were produced with
    assert prov["runs"][0]["model_id"] == REPO, "provenance is for a different checkpoint"
    print(f"UNetClassic revision pin: {revision}")
    model = model_zoo.UNetClassic.from_pretrained(
        REPO, revision=revision, cache_dir=os.path.join(HARNESS, "hf-cache")
    ).to(device).eval()
    fmt = DefaultChannelsFirstFormatter(ds.metadata)
    means, stds = OURS.load_stats()

    idxs = np.linspace(0, len(ds) - 1, N_PROBE).astype(int)
    results, worst = [], 0.0
    layouts = {}
    for idx in idxs:
        idx = int(idx)
        _, file_idx, sample_idx, time_idx, dt = ds._load_one_sample(idx)
        fname = os.path.basename(str(ds.files_paths[file_idx]))

        sample = ds[idx]
        batch = {k: (v.unsqueeze(0) if torch.is_tensor(v) else v) for k, v in sample.items()}
        inputs, y_ref = fmt.process_input(batch)
        with torch.inference_mode():
            lib_out = model(*[x.to(device) for x in inputs]).cpu()
        lib_vrmse = VRMSE.eval(fmt.process_output_channel_last(lib_out),
                               y_ref, ds.metadata).mean().item()

        if fname not in layouts:
            layouts[fname] = discover_layout(f"{OURS.BASE}/data/test/{fname}")
        frames = read_frames(layouts[fname], int(sample_idx), int(time_idx), OURS.HISTORY + 1)
        win = (frames[:OURS.HISTORY] - means) / stds
        # the exact tensor our evaluator hands the model, built the way OURS.forward builds it
        our_in = torch.nan_to_num(torch.from_numpy(
            OURS.frames_to_channels_first(win.astype(np.float32))))
        our_out_np = OURS.forward(model, win.astype(np.float32), device)   # [X,Y,Z,F] normalized
        tn = (frames[OURS.HISTORY] - means) / stds
        our_vrmse = float(np.mean(np.sqrt(
            spatial_mse(our_out_np.astype(np.float64), tn.astype(np.float64), 3)
            / (spatial_sample_variance(tn.astype(np.float64), 3) + 1e-7))))

        lr = y_ref.numpy()
        while lr.ndim > 4:
            lr = lr[0]
        lo = lib_out.numpy()[0].transpose(1, 2, 3, 0)
        # RAW-PHYSICAL units: the convention every cell this paper reports is computed in
        # (the validation loop denormalizes predictions before scoring). VRMSE is NOT
        # invariant to per-field rescaling at fixed epsilon, so this leg is a different and
        # strictly more relevant check than the library's own Z-scored routine.
        our_raw = vrmse_raw(our_out_np, tn, means, stds, 3)
        lib_raw = vrmse_raw(lo, lr, means, stds, 3)
        d_vraw = abs(lib_raw - our_raw) / max(abs(lib_raw), 1e-30)
        d_in = reldiff(inputs[0].numpy(), our_in.numpy())
        d_tg = reldiff(lr, tn)
        d_out = reldiff(lo, our_out_np)
        d_v = abs(lib_vrmse - our_vrmse) / max(abs(lib_vrmse), 1e-30)
        worst = max(worst, d_in, d_tg, d_out, d_v, d_vraw)
        print(f"idx {idx:6d}  {fname[:44]:44s} traj={sample_idx} t={time_idx}")
        print(f"    input {d_in:9.2e}  target {d_tg:9.2e}   output {d_out:9.2e}")
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
    print(f"  -> RT evaluator {'VALIDATED end to end' if worst < TOL else 'DISAGREES -- PAPER NUMBERS VOID'}")
    json.dump({"tol": TOL, "worst": worst, "validated": bool(worst < TOL),
               "windows": results},
              open(os.path.join(HARNESS, "fixtures/gate3/rt_evaluator_validation.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
