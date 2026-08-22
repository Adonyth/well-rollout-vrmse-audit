#!/usr/bin/env python3
"""End-to-end validation of the RT evaluator -- the one carrying the paper's core numbers.

Same construction as the Rayleigh-Benard evaluator validation, applied here to TFNO on
rayleigh_taylor_instability.
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

LANE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__))))
from independent_metrics import spatial_mse, spatial_sample_variance

from the_well.data import WellDataset
from the_well.data.normalization import ZScoreNormalization
from the_well.data.data_formatter import DefaultChannelsFirstFormatter
from the_well.benchmark.metrics.spatial import VRMSE
import the_well.benchmark.models as model_zoo
import rt_model_eval as OURS
from fast_reader import discover_layout, read_frames

REPO = "polymathic-ai/TFNO-rayleigh_taylor_instability"
TFNO_REVISION = "5cccb8597ed98189b63493b3d303cd0d24622b2f"
N_PROBE = int(os.environ.get("N_PROBE", "4"))
TOL = 1e-5


def reldiff(a, b):
    a = np.asarray(a, np.float64); b = np.asarray(b, np.float64)
    if a.shape != b.shape:
        return float("inf")
    return float(np.abs(a - b).max() / max(np.abs(a).max(), 1e-30))


def _out_path() -> str:
    """Resolve the output path against the HARNESS, and create the directory.

    An earlier version wrote to <lane>/fixtures/models, which does not exist -- the
    fixtures tree lives under repro-harness/. The script therefore completed the entire
    download, inference and comparison and then died with FileNotFoundError on the last
    line, having thrown away the result. `check_producer_targets` could not see it: its
    pattern matches absolute, `../` and `.gate-work` literals, and this was a bare
    relative join.
    """
    d = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "fixtures", "models")
    os.makedirs(d, exist_ok=True)
    return os.path.join(d, "tfno_rt_crossvalidation.json")


def main():
    ds = WellDataset(
        well_base_path="hf://datasets/polymathic-ai/",
        well_dataset_name="rayleigh_taylor_instability", well_split_name="test",
        n_steps_input=4, n_steps_output=1,
        use_normalization=True, normalization_type=ZScoreNormalization,
    )
    assert ds.norm is not None
    print(f"dataset len={len(ds)}  core_fields={ds.core_field_names}")

    device = torch.device("cpu")   # TFNO OOMs this machine's MPS at 128^3
    # TFNO has no packaged provenance yet (this run is what produces it), so the pin is
    # explicit here and is the SAME sha the rollout run used. Verified against the hub
    # rather than trusted, mirroring rt_model_eval.py's own drift check.
    import requests as _rq
    revision = TFNO_REVISION
    _info = _rq.get(f"https://huggingface.co/api/models/{REPO}/revision/{revision}",
                    timeout=30); _info.raise_for_status()
    assert _info.json().get("sha") == revision, "TFNO checkpoint revision drift"
    print(f"TFNO revision pin (hub-verified): {revision}")
    model = model_zoo.TFNO.from_pretrained(
        REPO, revision=revision, cache_dir=os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "hf-cache")
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
        our_out_np = OURS.forward(model, win.astype(np.float32), device)   # [X,Y,Z,F] normalized
        tn = (frames[OURS.HISTORY] - means) / stds
        our_vrmse = float(np.mean(np.sqrt(
            spatial_mse(our_out_np.astype(np.float64), tn.astype(np.float64), 3)
            / (spatial_sample_variance(tn.astype(np.float64), 3) + 1e-7))))

        lr = y_ref.numpy()
        while lr.ndim > 4:
            lr = lr[0]
        lo = lib_out.numpy()[0].transpose(1, 2, 3, 0)
        d_tg = reldiff(lr, tn)
        d_out = reldiff(lo, our_out_np)
        d_v = abs(lib_vrmse - our_vrmse) / max(abs(lib_vrmse), 1e-30)
        worst = max(worst, d_tg, d_out, d_v)
        print(f"idx {idx:6d}  {fname[:44]:44s} traj={sample_idx} t={time_idx}")
        print(f"    target {d_tg:9.2e}   output {d_out:9.2e}")
        print(f"    VRMSE  library={lib_vrmse:.6f}  ours={our_vrmse:.6f}  rel={d_v:.2e}  "
              f"{'OK' if max(d_tg,d_out,d_v) < TOL else '*** MISMATCH ***'}\n")
        results.append({"idx": idx, "file": fname, "traj": int(sample_idx),
                        "start": int(time_idx), "lib_vrmse": lib_vrmse,
                        "our_vrmse": our_vrmse, "d_target": d_tg, "d_output": d_out,
                        "d_vrmse": d_v})

    print("=== VERDICT ===")
    print(f"  windows: {len(results)}   worst relative discrepancy: {worst:.3e}   tol {TOL:g}")
    ok = worst < TOL
    print(f"  -> RT evaluator {'VALIDATED end to end' if ok else 'DISAGREES -- PAPER NUMBERS VOID'}")
    json.dump({"tol": TOL, "worst": worst, "validated": bool(worst < TOL),
                              "windows": results},
              open(_out_path(), "w"), indent=1)
    # A producer that can announce "PAPER NUMBERS VOID" and exit 0 is fail-open: the
    # MANIFEST advertises this command as the check behind the manuscript's cited
    # agreement, and a reviewer found it returned success while printing that verdict.
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
