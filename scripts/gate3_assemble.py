#!/usr/bin/env python3
"""Assemble fixtures/gate3/library_equivalence.json from the two re-check outputs.

Gate 3 of the paper's appendix asserts that this audit's evaluator agrees with the library's
own WellDataset + ZScoreNormalization + DefaultChannelsFirstFormatter path, on the public
test split, against the pinned public checkpoints. gate3_recheck_rt.py and
gate3_recheck_rb.py produce the per-dataset window comparisons; this script folds them into
the single fixture that verify.py asserts and that extract_numbers.py reads.

Deterministic: sorted keys, fixed indent, no timestamps -- so re-running it on unchanged
inputs reproduces the packaged fixture byte for byte.
"""
import json, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
HARNESS = os.path.dirname(HERE)
GATE3 = os.path.join(HARNESS, "fixtures", "gate3")

# Per dataset: which evaluator produced our side, where each side's bytes came from, and
# which tensors the re-check compares. The RT evaluator reads the SDSC mirror while the
# library streams Hugging Face, so that row also witnesses mirror-vs-release equality.
META = {
    "rayleigh_taylor_instability": {
        "input": "rt_evaluator_validation.json",
        "checkpoint": "polymathic-ai/UNetClassic-rayleigh_taylor_instability",
        "revision": "3a13f12900bce9be214d70b411d1a112b10fb493",
        "our_source": "SDSC mirror",
        "library_source": "HuggingFace",
    },
    "rayleigh_benard": {
        "input": "rb_evaluator_validation.json",
        "checkpoint": "polymathic-ai/UNetClassic-rayleigh_benard",
        "revision": "b72b1213dade391e4620476d1410c1bbe7103c1d",
        "our_source": "HuggingFace",
        "library_source": "HuggingFace",
    },
}
TOLERANCE = 1e-5
# The floor both metric legs are evaluated at. Recorded explicitly: VRMSE is meaningless
# without its floor, and the paper quotes these window values in two unit conventions.
METRIC_EPS = 1e-7


TENSOR_KEYS = {"rel_diff_input": "model_input", "rel_diff_target": "target",
               "rel_diff_output": "model_output"}


def tensor_diffs(window):
    return [window[k] for k in TENSOR_KEYS if k in window]


def tensors_compared(windows):
    """DERIVED, never asserted: which tensors were actually compared on EVERY window.

    Hard-coding this list is what previously let the manuscript claim the model input was
    compared on windows where no input comparison was ever computed. Taking the intersection
    over windows means a leg missing anywhere disappears from the claim.
    """
    if not windows:
        return []
    common = set(TENSOR_KEYS)
    for w in windows:
        common &= {k for k in TENSOR_KEYS if k in w}
    return [TENSOR_KEYS[k] for k in TENSOR_KEYS if k in common]


def environment():
    import platform
    env = {"python": platform.python_version(), "platform": platform.platform()}
    for mod in ("the_well", "torch", "numpy"):
        try:
            import importlib.metadata as md
            env[mod] = md.version(mod)
        except Exception:
            env[mod] = "unavailable"
    return env


def main() -> int:
    # Fail closed. The whole point of recording an environment is that it is the one the gate
    # ran in; assembling under an interpreter without the library under audit would silently
    # stamp the fixture "the_well: unavailable" and look like a successful run.
    env = environment()
    if env.get("the_well") in (None, "unavailable"):
        print("refusing to assemble: 'the_well' is not importable in this interpreter, so the "
              "recorded environment would not be the one Gate 3 ran in. Re-run with the same "
              "interpreter used for gate3_recheck_*.py.", file=sys.stderr)
        return 1
    out = {
        "description": (
            "End-to-end equivalence of this audit's evaluator against the library's own "
            "WellDataset + ZScoreNormalization + DefaultChannelsFirstFormatter path, on the "
            "public test split with the pinned public checkpoints. Produced by "
            ".gate-work/validate_{rt,rb}_evaluator.py."),
        "tolerance": TOLERANCE,
        "metric_eps": METRIC_EPS,
        "environment_at_assembly": environment(),
        "datasets": {},
    }
    for dset, m in META.items():
        path = os.path.join(GATE3, m["input"])
        if not os.path.exists(path):
            print(f"missing {path} -- run gate3_recheck_*.py first", file=sys.stderr)
            return 1
        with open(path, encoding="utf-8") as f:
            d = json.load(f)
        windows = []
        for w in d["windows"]:
            row = {"file": w["file"], "trajectory": w["traj"], "start": w["start"],
                   "eps": METRIC_EPS,
                   "library_vrmse_zscored": w["lib_vrmse"], "our_vrmse_zscored": w["our_vrmse"],
                   "rel_diff_vrmse": w["d_vrmse"], "rel_diff_target": w["d_target"],
                   "rel_diff_output": w["d_output"]}
            if "d_input" in w:
                row["rel_diff_input"] = w["d_input"]
            # raw-physical units: the convention every reported cell uses
            if "lib_vrmse_raw" in w:
                row["library_vrmse_raw"] = w["lib_vrmse_raw"]
                row["our_vrmse_raw"] = w["our_vrmse_raw"]
                row["rel_diff_vrmse_raw"] = w["d_vrmse_raw"]
            windows.append(row)
        tens = [v for w in windows for v in tensor_diffs(w)]
        out["datasets"][dset] = {
            "checkpoint": m["checkpoint"], "revision": m["revision"],
            "our_source": m["our_source"], "library_source": m["library_source"],
            "tensors_compared": tensors_compared(windows),
            "n_windows": len(windows),
            "n_files": len({w["file"] for w in windows}),
            "worst_rel_diff_vrmse": max(w["rel_diff_vrmse"] for w in windows),
            "worst_rel_diff_vrmse_raw": (
                max(w["rel_diff_vrmse_raw"] for w in windows)
                if all("rel_diff_vrmse_raw" in w for w in windows) else None),
            "worst_rel_diff_tensor": max(tens),
            "all_tensors_bit_identical": all(v == 0.0 for v in tens),
            "windows": windows,
        }
    allv = [w["rel_diff_vrmse"] for d in out["datasets"].values() for w in d["windows"]]
    allt = [v for d in out["datasets"].values() for w in d["windows"] for v in tensor_diffs(w)]
    allvr = [w["rel_diff_vrmse_raw"] for d in out["datasets"].values()
             for w in d["windows"] if "rel_diff_vrmse_raw" in w]
    out["summary"] = {
        "n_windows_total": len(allv),
        "worst_rel_diff_vrmse": max(allv),
        "worst_rel_diff_vrmse_raw": max(allvr) if len(allvr) == len(allv) else None,
        "raw_units_leg_on_all_windows": len(allvr) == len(allv),
        "worst_rel_diff_tensor": max(allt),
        "all_tensors_bit_identical": all(v == 0.0 for v in allt),
    }
    dest = os.path.join(GATE3, "library_equivalence.json")
    with open(dest, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1, sort_keys=True, allow_nan=False)
    print(f"wrote {dest}: {out['summary']['n_windows_total']} windows, "
          f"worst tensor rel diff {out['summary']['worst_rel_diff_tensor']:g}, "
          f"worst VRMSE rel diff {out['summary']['worst_rel_diff_vrmse']:.3e}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
