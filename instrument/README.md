# normscreen

**A conditioning screen for variance-normalized error metrics.** Point it at a benchmark
split; it tells you which fields and frames have a metric denominator carried by a
stabilizing constant rather than by the data. No model required.

```
pip install -e .
# declare the layout: how many LEADING axes, e.g. [trajectory, time, x, y, z] -> 2
normscreen your_data.h5 --auto --spatial-dims 3 --eps 1e-7 --leading-axes 2 --component-axis -1
# or screen every field whole:
normscreen your_data.h5 --auto --spatial-dims 3 --eps 1e-7 --no-split-components
# a floorless RMS-normalized metric (PDEBench-style):
normscreen your_data.h5 --auto --spatial-dims 1 --eps 0 --normalizer rms --no-split-components
```

**The screen never infers the component axis from rank, and neither should you.** A scalar
stored `[trajectory, time, y, x]` and a vector stored `[time, y, x, component]` have the same
rank; so do a two-leading-axis scalar and a one-leading-axis vector. Splitting the wrong one
takes the variance over the wrong axes and can report a degenerate field as clean — the exact
failure this tool exists to catch. So there are three safe ways to say what you have, and the
screen refuses rather than guess if you use none of them:

| you have | say |
|---|---|
| one layout for every field | `--leading-axes K --component-axis -1` |
| scalars and vectors in one file | `component_axis={"velocity": -1}` (Python API) |
| no component axes at all | `--no-split-components` |

A single `--component-axis` with no `--leading-axes` is **never** accepted when any field
could be split: rank-uniformity is not unambiguity, since a file of scalars stored
`[trajectory, time, y, x]` is uniform and still ambiguous. And `--leading-axes` on its own is
not enough either — a count of leading axes does not say *which* axis holds the components,
and assuming the last one splits a channel-first store along a spatial axis. Give both, or use
the per-field mapping, or screen whole.

## The problem it screens for

Scores of the form

```
S_f = sqrt( MSE_f / (Var_f + eps) )
```

divide each field's error by that field's own spatial variability. Up to a monotone
transform this is the **Nash–Sutcliffe efficiency**, in continuous use in hydrology since
Nash & Sutcliffe (1970) — and that field documented its pathologies decades ago. When
`Var_f` collapses, a fixed absolute error is amplified without bound, and scores computed
on data with different variability are not comparable to one another (Schaefli & Gupta,
2007; Knoben et al., 2019).

Machine-learning benchmarking has imported the estimator without importing the caveat.
The additive `eps` — there only to prevent division by zero — silently becomes the
dominant term wherever the target is quiescent. The reported number still moves with the
model's error, but its scale is then set by the constant rather than by the data. The numerical-analysis literature meets this regime with the
mixed tolerance `atol + rtol·|y|`. A fixed `eps` is an *analogue* of that construction, not an
instance of it: the solver scale is additive, `sqrt(eps) + sqrt(Var)`, while a floored
variance denominator is a root-sum-square, `sqrt(eps + Var)`. They differ by the cross term
and by a factor of `sqrt(2)` exactly at `Var = eps` — the regime this tool screens for.

`normscreen` reports the one quantity that makes this visible:

```
floor_share = eps / (Var + eps)
```

`0` means the data carries the denominator. `1` means the constant does.

## Two normalizer conventions, and why the choice matters

Benchmarks do not agree on what to divide by, and the choice decides which fields are
scoreable at all:

| benchmark | denominator | floor | failure mode where the signal vanishes |
|---|---|---|---|
| The Well (VRMSE) | `Var + eps` — second moment about the **mean** | `eps = 1e-7` | **saturates** at `1/sqrt(eps)` |
| PDEBench (nRMSE) | `sqrt(mean(x**2))` — second moment about **zero** | none | **unbounded** |

Because `RMS² = Var + mean²`, a field oscillating slightly about a large offset has a
healthy RMS and a degenerate variance. The same array can be `well-conditioned` under one
convention and `floor-determined` under the other — `normscreen --normalizer {variance,rms}`
reports whichever your metric actually uses.

## What it reports

| band | floor share | reading |
|---|---|---|
| well-conditioned | < 0.01 | the score means what it appears to mean |
| floor-sensitive | 0.01 – 0.50 | the constant is material |
| floor-dominated | 0.50 – 0.90 | the constant supplies most of the denominator |
| floor-determined | ≥ 0.90 | the score still tracks the error, but its scale is set by `eps` |

Plus, per field: min/median/max variance, the fraction of frames that are
floor-determined (share >= 0.90) and the fraction at or below the floor (Var <= eps),
and an `eps` sweep giving the factor by which any fixed error's score moves between
candidate floors.

Exit status is `1` on anything you should not read as a clean screen: a floor-determined
denominator; a zero denominator under a floorless metric (unbounded, not merely saturated);
a non-finite denominator (undefined — NaN/inf input, or too few spatial points to form a
variance); and an **incomplete** screen, where the tool had to guess that a field's extra axis
was a leading index rather than components. An asserted whole-screen (`--no-split-components`,
or a per-field `None`) is not incomplete — you told it what the layout is — but the report
still says which fields it did not screen per component. So it composes into CI without
certifying data it could not actually score, and without silently certifying data it only
partly screened.

## What it does *not* claim

A floor-dominated denominator is **necessary but not sufficient** for a misleading score.
Whether a reading is actually inflated also depends on the model's error, which no
data-only pass can see. The screen identifies cells worth checking — it does not
adjudicate them. It also bounds only the frames you feed it.

## Validation

`test_against_paper.py` checks the screen against an audit's frozen census of a 17-dataset
physics-ML benchmark. It re-applies the conditioning statistic to that audit's **stored**
variances and compares with its **stored** shares; it does not re-extract the variances from
the source arrays, so it is an internal-consistency check rather than an independent
reproduction of the census:

```
17 rows reproduced exactly at the stored precision, 0 disagree
1728 per-window variances: worst floor share 0.999908, floor-determined (share>=0.90) in 39.8% of windows
PASS
```

**What this does not establish.** `normscreen.floor_share` and the census assembler evaluate
the same expression, `eps/(Var+eps)`, and the fixture stores the result at six significant
figures. Comparing them is a regression check that binds the two together — it would catch a
change in either — but it is not independent corroboration of the statistic, and it takes the
variance extraction as given. An earlier version of this file quoted a "worst-case relative
agreement of 3.715e-6"; that number was the `%.6g` rounding of the single tightest row and would
have been `0.0` had the fixture stored full precision. It is withdrawn.

Both fixture counts (17 rows, 1728 windows) are pinned, so a missing or partially fetched
fixture tree fails the suite instead of passing over nothing.

`test_normalizers.py` pins the two-convention behaviour, including the `RMS² = Var + mean²`
identity, the `eps = 0` case (floor share is 0, never NaN), and saturation-versus-
unboundedness.

## Worked example on a second benchmark

`examples/pdebench_shocktube.py` downloads one small public PDEBench file (DaRUS
doi:10.18419/darus-2986, ~4.9 MB, CC-BY) and screens it under both conventions — a
different benchmark, a different file layout, a different metric design, no model.

On that file the screen reports a velocity channel whose denominator is exactly zero on
every frame; direct inspection confirms the channel is identically zero and that density
and pressure are unchanged from first frame to last, i.e. the file holds a static state.
Under PDEBench's floorless nRMSE that channel's score is undefined rather than large.
**Scope:** one file, and an anomalous one — 4.9 MB against 12–25 GB for every other file in
the same collection. No claim is made about PDEBench as a whole; the point is that a
data-only screen costs seconds and would have caught it.

## API

```python
from normscreen import screen_fields, floor_share

report = screen_fields({"density": rho, "velocity_x": vx}, eps=1e-7, n_spatial=3,
                       component_axis=None)   # or component_axis={"velocity": -1}
print(report.text())
report.to_dict()                 # JSON-serializable
floor_share(var=1e-11, eps=1e-7) # 0.99989
```

## References

- Nash, J.E. & Sutcliffe, J.V. (1970). River flow forecasting through conceptual models
  part I — A discussion of principles. *Journal of Hydrology* 10(3), 282–290.
- Schaefli, B. & Gupta, H.V. (2007). Do Nash values have value? *Hydrological Processes*
  21(15), 2075–2080.
- Knoben, W.J.M., Freer, J.E. & Woods, R.A. (2019). Technical note: Inherent benchmark or
  not? Comparing Nash–Sutcliffe and Kling–Gupta efficiency scores. *HESS* 23, 4323–4331.
- Post, M. (2018). A Call for Clarity in Reporting BLEU Scores. *WMT*.
- Hairer, E., Nørsett, S.P. & Wanner, G. (1993). *Solving Ordinary Differential Equations I*,
  §II.4 (automatic step size control).

## License

BSD-3-Clause, matching the audit harness this instrument was extracted from.
