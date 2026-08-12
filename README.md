# P3 repro-harness

Standalone reproduction package for the P3 "benchmark forensics" paper's
quantitative pipeline (The Well / `polymathic-ai` Rayleigh-Taylor benchmark,
issues #75 / #78, class-split census). Public code release, single-blind
DMLR review — full author attribution intended and preserved
(Jiaxuan Chen, github.com/Adonyth).

## One command

Install the Tier 1 dependency first (a single package, `numpy>=2.0`; a clean
shell without it will raise `ModuleNotFoundError: No module named 'numpy'`
on the next step):

```bash
pip install -r requirements.txt
```

Then run:

```bash
python3 verify.py
```

This recomputes `fixtures/summary.json` from the packaged raw fixtures (pure
numpy, no network, no GPU, ~1 second) and runs 142 value checks against the
frozen `paper/extracted/numbers.json` reference (or, for the figure-vs-table
and spatial-mean-baseline groups below, against a sibling key / a second
independent code path) to a relative tolerance of 1e-4 (about four significant figures):
for the pinned UNetClassic checkpoint, the one-step estimate under both the
library floor and the documented floor (eps=1e-5) together with its
well-/ill-conditioned split, both rollout windows under the library floor,
the documented floor, and the density-only reading, and the issue-#78
last-batch-vs-all-batch counterfactual at both windows (17); the
corresponding one-step, both-window rollout (library / documented-floor /
density-only), and issue-#78 window-13-30 counterfactual cells for
UNetConvNext (12); the one-step estimate under both floors and the
window-6-12 rollout cell under the library floor, the documented floor, and
density-only for FNO (5); every cell of the full ten-trajectory
denominator census in Table~census (50 = 10 trajectories x 5 printed
columns); a figure-vs-table consistency check (40 = 10 trajectories x 4
printed columns) asserting that the quantity Figure~census plots
(`n_frames_var_le_1e-N`, a frame count) equals its sibling key
`last_t_var_le_1e-N + 1` — a regression guard ensuring the figure plots the
frame count, not the off-by-one `last_t_var_le_1e-N` value directly, which
would disagree with the identically-labeled Table 1 cell in all 40 cells;
every printed cell of Table~fieldsplit
(16 = 2 models x 2 windows x 4 columns); and a cross-check (2) of
`scripts/spatial_mean_baseline.py`'s trivial mean-predictor baseline
(0.2699 at window 6-12, 0.3919 at 13-30 for UNetClassic under the
documented floor — the checkpoint's own eps5 score of 1.927/1.267 is
7.1x/3.2x *worse* than this baseline, so it is not comparable in quality to
predicting the field mean) against a second,
independent derivation of the same quantity in
`paper/extracted/numbers.json`. Exit code 0 = all match.

```
$ python3 verify.py
...
[map] ...   [gate3] ...   [coverage] ...   [gate2] ...
[3/3] 142 match, 0 mismatch.
[blocks] ...   [provenance] ...   [sweep] ...   [rb] ... x2
[chronology] ...   [rb] ...   [selfclaims] ...
PASS: repro-harness regenerates the frozen P3 values to a relative tolerance of 1e-4 (~4 sig figs).
```

One further stage, `[reference]`, prints between `[gate2]` and `[blocks]` **only inside
the source repository**, where it asserts that this package's frozen
`fixtures/numbers_reference.json` is leaf-for-leaf identical to the released
`paper/extracted/numbers.json`. That path is outside this package by design, so a
standalone clone does not print the line and does not check it: the frozen copy is the
comparison target here, and its agreement with the manuscript's copy is asserted where
both are present, not here. Everything else above runs in a clone.

This is what stands behind the paper's coverage-tier claim in
`sec7_boundaries.tex`: the wider Lane-3 chain (the two library-defect deltas
and the shear-flow device-precision cross-check) is documented with exact
source scripts and commands (see MANIFEST.md) but is not part of this
package's asserted exit code.

## Layout

```
repro-harness/
  README.md              <- you are here
  LICENSE                 <- BSD-3-Clause (code)
  MANIFEST.md             <- exact public data sources + exact commands (Tier 2)
  requirements.txt        <- Tier 1 deps (numpy) + pointer to Tier 2 deps
  env-lock-full.txt        <- full pinned pip freeze from the original Tier-2 run
  verify.py               <- one-command reproduction + check (run this)
  scripts/
    verify-time (pure numpy, no network)
      aggregate_results.py     <- raw scalars -> the 142 enumerated value checks
      independent_metrics.py   <- the_well-independent VRMSE/variance reference implementation
      io_contract.py           <- the one place a per-window row file's storage is decided
      rb_checkpoint_audit.py   <- re-derives every Rayleigh-Benard cell from the packaged rows
      assemble_map.py          <- assembles the 17-dataset conditioning map from per-dataset census outputs
      spatial_mean_baseline.py <- trivial spatial-mean-predictor VRMSE baseline
      gate3_assemble.py        <- assembles the Gate-3 fixture from the two per-dataset outputs
      check_self_claims.py     <- checks the package's prose/docstrings against the tree they describe
                                  (`--validate-guard` reintroduces fifteen real discrepancies, each must be caught)
      check_unsourced.py       <- numbers-must-trace-to-numbers.json checker (paper-writing discipline)
      test_lastbatch_agg.py    <- reproduces the issue-#78 last-batch aggregation defect
      test_vrmse_epsilon.py    <- reproduces the issue-#75 floor defect
    Tier-2 producers (network + torch + the_well; not run by verify.py)
      fast_reader.py           <- HTTP-range HDF5 reader for public Well test objects
      rt_audit_pass.py         <- Pass A: Rayleigh-Taylor denominator/census audit
      rt_model_eval.py         <- Pass B: Rayleigh-Taylor checkpoint eval (--revision required, drift-checked)
      rb_model_eval.py         <- the same for Rayleigh-Benard, writing via io_contract
      rb_census.py             <- the RB Prandtl-1 conditioning census
      well_denominator_census.py <- the benchmark-wide, data-only census over every Well dataset
      fetch_checkpoint_chronology.py <- checkpoint upload history from the model hub's commit API
      gate2_alignment_fixture.py <- Gate 2: multi-step alignment on a locally synthesized fixture
      gate3_recheck_rt.py / gate3_recheck_rb.py <- Gate 3: library-native re-check on real artifacts
      shear_checkpoint_eval.py <- device-precision cross-check on a second Well dataset
  instrument/              <- normscreen: the standalone conditioning screen (see below)
    normscreen/            <- the package (numpy only; h5py optional, for HDF5 input)
    test_against_paper.py  <- checks the screen against this repo's frozen census fixtures
    test_normalizers.py    <- pins the variance/RMS conventions, eps=0, saturation vs unboundedness
    examples/pdebench_shocktube.py <- worked example on a second, non-Well benchmark
  fixtures/
    audit/*.json.gz         <- packaged raw census scalars, all 5 public RT test objects (10 traj)
    audit/provenance.json   <- byte counts + ETags for the 5 public HTTP objects fetched
    models/*.json.gz        <- packaged raw per-window MSE/variance, 3 RT models
    models/provenance_*.json<- HF checkpoint identity (repo id + commit sha + param count)
    rb_models/*.json.gz     <- the Rayleigh-Benard depth pass (12 runs, 1728 rows) + its aggregate
    rb_spread/*.json.gz     <- the RB stratified pass / sampling control (20 runs)
    generalization/         <- the 17-dataset conditioning map, the RB detail census, the frozen
                               transcription of the published ">10" cells, and the source revisions
    gate2/, gate3/          <- the two frozen verification-gate fixtures
    provenance/             <- the checkpoint upload chronology
    lane3_report_numbers.json <- the wider chain's number set, carried for provenance only:
                               nothing in this package reads it (see MANIFEST.md)
    numbers_reference.json  <- frozen copy of paper/extracted/numbers.json (comparison target)
    summary.json            <- (written by verify.py / aggregate_results.py; not committed input)
```

## Two tiers

**Tier 1 (what `verify.py` runs today):** the raw per-frame/per-window
scalars (MSE, target/prediction variance — never raw field tensors) are
packaged in `fixtures/` (928,155 bytes in total, of which the RT per-frame/per-window scalars are 240,823; the rest is the Rayleigh-Benard checkpoint rows, the benchmark-wide census, and the Gate-2/Gate-3/provenance fixtures). `verify.py` re-derives the 142 enumerated value checks
from those scalars via `aggregate_results.py` and diffs against the
frozen paper numbers. This is a genuine recomputation, not a file diff: the
aggregation (eps variants, rollout window means, one-step interpolation,
well-conditioned-subset splits, issue-#78 counterfactual) is ~350 lines of
numpy executed fresh on every run.

**Tier 2 (documented, not executed by `verify.py`):** the raw fixtures
themselves were produced by streaming the public Well RT test-split HDF5
objects over HTTPS (`scripts/fast_reader.py`, exact-byte-offset ranged GETs)
and running the pinned `polymathic-ai/*` Hugging Face checkpoints --- three
Rayleigh-Taylor checkpoints (FNO, UNetClassic, UNetConvNext) via
`scripts/rt_model_eval.py`, four Rayleigh-Benard checkpoints via
`scripts/rb_model_eval.py`, plus a shear-flow FNO cross-check via
`scripts/shear_checkpoint_eval.py`. `MANIFEST.md` gives the exact commands, exact
`--pairs`/`--onestep-starts` arguments, and exact source URLs/HF commit
shas used to produce the packaged fixtures, so a reviewer can regenerate
them from scratch (network + ~40 GB test-split streaming + ~1.4 GB checkpoint
downloads + inference required — substantial time, dominated by the ~40 GB
transfer; not run inside this harness's one-command check).

## Why the fixtures are safe to package

They are the *outputs* of computing statistics (MSE, sample variance,
spatial means) over public CC-BY-4.0 data — small JSON scalars, never the
raw 128^3-voxel field tensors themselves (those remain fetched on demand
from the public Flatiron/SDSC mirror by Tier 2, never bundled).

## Environment

Tier 1 needs only `numpy` (see `requirements.txt`). Tier 2 needs the full
stack pinned in `env-lock-full.txt` (`torch==2.13.0`, `the_well==1.2.0`,
`h5py`, `fsspec`, `huggingface_hub`, ...) — install with
`pip install -r env-lock-full.txt` if you want to run the cold-start
commands in `MANIFEST.md`.

## Numbers discipline

Every number this harness's `verify.py` checks traces to a leaf of
`fixtures/numbers_reference.json`, which is itself pulled programmatically
(no hand transcription) from `fixtures/summary.json` +
Lane-3's `report_numbers.json` by the source repo's
`paper/extracted/extract_numbers.py`. `scripts/check_unsourced.py` is
included so a paper chapter draft can be checked the same way the source
repo checks it: `python3 scripts/check_unsourced.py <chapter.tex>
fixtures/numbers_reference.json` must print `UNSOURCED: 0`.

## The instrument (`instrument/normscreen`)

The paper's first lesson asks a benchmark to publish, beside each
variance-normalized score, how much of that score's denominator is the
stabilizing constant rather than the data. `instrument/` packages exactly that
check as a standalone tool so the lesson is cheap to act on. It needs no model
and no training run — point it at an HDF5 or NumPy split and it reports
`floor_share = eps/(Var+eps)` per field and per frame, under either the
variance convention (Nash–Sutcliffe family, as here) or the RMS convention
(as PDEBench's nRMSE). Exit status is non-zero on a positive screen, so it
composes into CI.

```bash
cd instrument
python3 -m normscreen --demo                          # synthetic, no download
python3 -m normscreen your_data.h5 --auto --spatial-dims 3 --eps 1e-7 --leading-axes 2 --component-axis -1
python3 test_against_paper.py                         # vs this repo's frozen census
python3 test_normalizers.py                           # convention/edge-case pins
```

`test_against_paper.py` re-applies the conditioning statistic to this repo's
**stored** variances (`fixtures/generalization/benchmark_map/MAP.json`, 17 rows;
`fixtures/rb_models/*.json.gz`, 1728 windows). For the map it compares against the
**stored** shares; the Rayleigh-Benard rows store no shares, so there it recomputes them
from the stored variances and compares against counts pinned in the test file. It is an internal-consistency check between the instrument and the
audit's stored inputs, not an independent re-derivation of the census — the
variance extraction itself is taken as given. Both fixture counts are pinned, and
a missing or truncated fixture tree fails the suite rather than reporting a pass
over nothing. See `instrument/README.md` for the API and the scope limits.

## License

Code (this repository) is licensed under BSD-3-Clause — see `LICENSE`.
The packaged fixtures are derived statistics (MSE, sample variance, spatial
means) over The Well's public Rayleigh-Taylor test-split data, which is
released under CC-BY-4.0; the derived fixtures in this repository are
likewise available under CC-BY-4.0. The upstream data and Hugging Face
model checkpoints keep whatever licenses their model repositories declare
— see MANIFEST.md.
