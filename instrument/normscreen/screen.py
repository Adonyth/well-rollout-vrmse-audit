"""normscreen -- conditioning screen for variance-normalized error metrics.

WHAT THIS IS FOR
----------------
Scores of the family

    S_f = sqrt( MSE_f / (Var_f + eps) )

are variance-normalized: each field's error is divided by that field's own spatial
variability. Up to a monotone transform this is the Nash-Sutcliffe efficiency, which
hydrology has used since 1970 and whose pathologies that field documented long ago:
when Var_f collapses, a fixed absolute error is amplified without bound, and scores
computed on different data are not comparable to each other.

Machine-learning benchmarks have imported the estimator without importing the caveat.
The additive constant `eps` -- present to stop a division by zero -- silently becomes
the dominant term in the denominator wherever the target is quiescent. The reported number
still moves with the model's error, but its SCALE is then set by the constant rather than by
the data, so it no longer sits on the scale the score is read on.

This tool answers one question about YOUR benchmark, using no model at all:

    for each field and each frame, what share of the metric's denominator is supplied
    by the stabilizing constant rather than by the data?

        floor_share = eps / (Var + eps)

A field at floor_share ~ 0 is well conditioned: the score means what it appears to
mean. A field approaching 1 is floor-dominated: the score still tracks the error, but its
scale is set by `eps` rather than by the data, so it is
not comparable to a score computed where the denominator is healthy, and should not be
aggregated with one.

It is a SCREEN, not a verdict. A floor-dominated denominator is necessary but not
sufficient for a misleading score -- whether the reading is actually inflated also
depends on the model's error, which no data-only pass can see. The screen tells you
which cells to check, not which cells are wrong.

USAGE
-----
    normscreen data.h5 --fields density velocity_x velocity_y --no-split-components
    normscreen data.h5 --auto --eps 1e-7 --json report.json --no-split-components
    normscreen field.npy --spatial-dims 3 --leading-axes 1 --component-axis -1

Accepts HDF5, NumPy .npy/.npz, or any array via the Python API. No dependency on any
particular benchmark's layout.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field as _dc_field
from typing import Any, Sequence

import numpy as np

__all__ = ["FieldReport", "ScreenReport", "screen_array", "screen_fields",
           "split_components",
           "floor_share", "eps_sweep", "CONDITION_BANDS", "NORMALIZERS",
           "spatial_rms2"]

# Bands are reporting conventions, not physics. They exist so that two people reading
# the same report reach the same words. The boundaries are round numbers chosen to be
# memorable; the continuous floor_share is always reported alongside them.
CONDITION_BANDS: tuple[tuple[float, str], ...] = (
    (0.01, "well-conditioned"),      # the floor supplies under 1% of the denominator
    (0.50, "floor-sensitive"),       # the floor is material but not dominant
    (0.90, "floor-dominated"),       # the floor supplies most of the denominator
    (1.01, "floor-determined"),      # the score still tracks the error, but only up to
                                     # eps: its scale is set by the constant, not the data
)


def band_for(share: float) -> str:
    """Name the conditioning band a floor share falls in."""
    if not math.isfinite(share):
        return "undefined"
    for upper, name in CONDITION_BANDS:
        if share < upper:
            return name
    return CONDITION_BANDS[-1][1]


def floor_share(var: np.ndarray | float, eps: float) -> np.ndarray | float:
    """Share of the metric denominator supplied by the stabilizing constant.

    floor_share = eps / (Var + eps).  0 means the data carries the denominator;
    1 means the constant does.

    `eps` must be non-negative. A negative stabilizer is not a floor: it shrinks the
    denominator, so it *amplifies* rather than bounds, and it produces a negative
    share that reads as a healthier-than-healthy screen. An earlier version accepted
    it and reported "the denominator is carried by the data everywhere here" for
    eps = -1e-7. It is rejected.

    With eps == 0 the metric has no stabilizing constant, so the constant's share is
    0 wherever the denominator is positive -- and at Var == 0 the expression is 0/0.
    That case is not "half floored": it is a metric dividing by exactly zero, which is
    a different and worse failure than saturation. We return 0.0 there (the constant
    genuinely supplies nothing) and surface the condition through the unbounded count
    in eps_sweep and through the eps==0 verdict, rather than propagating a NaN that
    would silently poison a max() or a band lookup.
    """
    if eps < 0:
        raise ValueError(f"eps must be non-negative; got {eps!r}. A negative "
                         f"stabilizer shrinks the denominator instead of flooring it.")
    var = np.asarray(var, dtype=np.float64)
    if eps == 0:
        return np.zeros_like(var) if var.ndim else 0.0
    return eps / (var + eps)


NORMALIZERS = {
    # name -> (callable(array, n_spatial, ddof) -> per-frame denominator BEFORE the floor,
    #          human description)
    "variance": ("second moment about the mean (Var); The Well's VRMSE, Nash-Sutcliffe family"),
    "rms": ("second moment about zero (mean of squares); PDEBench's nRMSE"),
}


def spatial_rms2(a: np.ndarray, n_spatial: int, ddof: int = 0) -> np.ndarray:
    """Per-frame mean of squares over the trailing `n_spatial` axes.

    This is the square of the quantity PDEBench's nRMSE divides by
    (nrm = sqrt(mean(target**2))). It differs from the variance by the mean's
    contribution: RMS^2 = Var + mean^2. A field oscillating slightly about a large
    offset therefore has a healthy RMS and a degenerate variance -- which is exactly
    why two benchmarks can make opposite normalization choices and see opposite
    conditioning. `ddof` is accepted for signature symmetry and ignored.
    """
    a = np.asarray(a, dtype=np.float64)
    if n_spatial < 1 or n_spatial > a.ndim:
        raise ValueError(f"n_spatial={n_spatial} incompatible with array of ndim={a.ndim}")
    axes = tuple(range(a.ndim - n_spatial, a.ndim))
    return np.mean(a ** 2, axis=axes)


def spatial_variance(a: np.ndarray, n_spatial: int, ddof: int = 1) -> np.ndarray:
    """Per-frame, per-field variance over the trailing `n_spatial` axes.

    Input is [..., *spatial]; output keeps the leading axes. ddof=1 matches the sample
    convention most evaluation code uses; pass ddof=0 to match a population convention.
    """
    a = np.asarray(a, dtype=np.float64)
    if n_spatial < 1 or n_spatial > a.ndim:
        raise ValueError(f"n_spatial={n_spatial} incompatible with array of ndim={a.ndim}")
    axes = tuple(range(a.ndim - n_spatial, a.ndim))
    return a.var(axis=axes, ddof=ddof)


def eps_sweep(var: np.ndarray, mse: np.ndarray | None, epsilons: Sequence[float]) -> dict:
    """How much a score would move across candidate floors.

    With `mse` supplied the sweep is exact for those predictions. Without it, the sweep
    is reported as the denominator-only factor sqrt((Var+eps_max)/(Var+eps_min)), i.e.
    the factor by which ANY fixed error's score changes between the two floors -- which
    is the quantity that does not depend on having a model.
    """
    var = np.asarray(var, dtype=np.float64)
    out: dict[str, Any] = {"epsilons": list(epsilons)}
    if mse is not None:
        mse = np.asarray(mse, dtype=np.float64)
        out["score_at_eps"] = {
            repr(e): float(np.mean(np.sqrt(mse / (var + e)))) for e in epsilons
        }
        vals = [out["score_at_eps"][repr(e)] for e in epsilons]
        finite = [v for v in vals if math.isfinite(v) and v > 0]
        out["span_factor"] = (max(finite) / min(finite)) if len(finite) > 1 else float("nan")
    else:
        lo, hi = min(epsilons), max(epsilons)
        denom_lo = var + lo
        # At the definitional limit (eps=0) a frame with exactly zero variance makes the
        # ratio infinite -- which is the true answer, not an error: with no floor and no
        # signal the score is unbounded. Report it as such instead of emitting a warning.
        with np.errstate(divide="ignore", invalid="ignore"):
            factor = np.sqrt((var + hi) / denom_lo)
        n_unbounded = int(np.sum(~np.isfinite(factor)))
        finite = factor[np.isfinite(factor)]
        out["denominator_only_span_factor"] = {
            "min": float(np.min(finite)) if finite.size else float("nan"),
            "median": float(np.median(finite)) if finite.size else float("nan"),
            "max": float(np.max(finite)) if finite.size else float("nan"),
            "n_unbounded_at_smallest_eps": n_unbounded,
        }
        out["note"] = ("no predictions supplied; this is the factor by which any fixed "
                       "error's score changes between the smallest and largest floor")
    return out


@dataclass
class FieldReport:
    name: str
    normalizer: str
    n_frames: int
    min_variance: float
    median_variance: float
    max_variance: float
    floor_share_at_min_variance: float
    median_floor_share: float
    # Counts the >=0.90 band, which CONDITION_BANDS names "floor-determined". An earlier
    # version called these fields *_floor_dominated, which named the 0.50-0.90 band and so
    # contradicted the band table they were printed beside.
    frames_floor_determined: int
    fraction_floor_determined: float
    band: str
    frames_at_or_below_floor: int = 0
    fraction_at_or_below_floor: float = 0.0
    frames_zero_denominator: int = 0
    frames_nonfinite_denominator: int = 0
    sweep: dict = _dc_field(default_factory=dict)

    def to_dict(self) -> dict:
        d = self.__dict__.copy()
        return d


@dataclass
class ScreenReport:
    eps: float
    n_spatial: int
    ddof: int
    normalizer: str
    fields: list[FieldReport]
    least_conditioned_field: str
    worst_floor_share: float
    verdict: str

    def to_dict(self) -> dict:
        return {
            "eps": self.eps, "n_spatial": self.n_spatial, "ddof": self.ddof,
            "normalizer": self.normalizer,
            "least_conditioned_field": self.least_conditioned_field,
            "worst_floor_share": self.worst_floor_share, "verdict": self.verdict,
            "fields": [f.to_dict() for f in self.fields],
        }

    def text(self) -> str:
        w = max((len(f.name) for f in self.fields), default=5)
        lines = [
            f"normscreen: normalizer={self.normalizer}, eps={self.eps:g}, "
            f"{self.n_spatial} spatial dims, ddof={self.ddof}",
            "",
            f"{'field'.ljust(w)}  {'min Var':>11}  {'floor share':>11}  "
            f"{'%<=floor':>9}  {'%>=0.9':>7}  band",
            f"{'-'*w}  {'-'*11}  {'-'*11}  {'-'*9}  {'-'*7}  {'-'*18}",
        ]
        for f in self.fields:
            lines.append(
                f"{f.name.ljust(w)}  {f.min_variance:11.3e}  "
                f"{f.floor_share_at_min_variance:11.4f}  "
                f"{100*f.fraction_at_or_below_floor:8.1f}%  "
                f"{100*f.fraction_floor_determined:6.1f}%  {f.band}"
            )
        lines += ["", f"least-conditioned field: {self.least_conditioned_field} "
                      f"(floor share {self.worst_floor_share:.4f})", "", self.verdict]
        return "\n".join(lines)


def _denominator(arr, n_spatial, ddof, normalizer):
    if normalizer == "variance":
        return spatial_variance(arr, n_spatial, ddof=ddof)
    if normalizer == "rms":
        return spatial_rms2(arr, n_spatial)
    raise ValueError(f"unknown normalizer {normalizer!r}; choose from {sorted(NORMALIZERS)}")


def _floorless_amplification(var: np.ndarray) -> float:
    """1/sqrt(min Var): how much a floorless score multiplies a fixed error, absolutely.

    The relative span cannot see a report whose fields are ALL uniformly quiescent -- every
    span is ~1 and the cross-field ratio is ~1 too, yet the score is amplified by orders of
    magnitude relative to any normally-conditioned data. That was the exact regression this
    band exists to prevent, still reachable through the documented floorless mode.
    """
    v = np.asarray(var, dtype=np.float64)
    v = v[np.isfinite(v) & (v > 0)]
    if v.size == 0:
        return float("inf")
    return float(1.0 / np.sqrt(v.min()))


def _floorless_span(var: np.ndarray) -> float:
    """Ratio of largest to smallest positive denominator, or inf if any is zero.

    With no stabilizing constant this ratio IS the conditioning: the score varies by its
    square root across frames from the denominator alone, independently of the model. A
    denominator span of 1e6 is a thousandfold swing in the score produced by the data rather
    than by the model, which is the point past which frames are not comparable to each other.
    """
    v = np.asarray(var, dtype=np.float64)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return float("nan")
    if np.any(v <= 0):
        return float("inf")
    return float(v.max() / v.min())


def screen_array(name: str, arr: np.ndarray, *, eps: float = 1e-7, n_spatial: int = 2,
                 ddof: int = 1, mse: np.ndarray | None = None,
                 epsilons: Sequence[float] | None = None,
                 normalizer: str = "variance") -> FieldReport:
    """Screen one field's array of shape [..., *spatial].

    `normalizer` selects what the metric divides by: "variance" (Nash-Sutcliffe family,
    The Well's VRMSE) or "rms" (PDEBench's nRMSE). With eps=0 -- which is what a metric
    that has no stabilizing constant does -- the floor share is 0 everywhere and the
    reported risk is unboundedness instead of saturation.
    """
    var = _denominator(arr, n_spatial, ddof, normalizer).ravel()
    shares = np.asarray(floor_share(var, eps)).ravel()
    # A non-finite denominator makes every threshold comparison below false, which would
    # otherwise read as "no frame is floor-dominated" and exit clean. Undefined is not
    # well-conditioned: count these and let the report carry them into the verdict.
    n_nonfinite = int(np.sum(~np.isfinite(var)))
    dominated = int(np.sum(shares[np.isfinite(shares)] >= 0.90))
    # The threshold statistic a benchmark should publish is the fraction of scored elements
    # whose denominator is at or below the floor, i.e. Var <= eps, i.e. floor share >= 0.5.
    # That is NOT the same as the 0.90 band (Var <= eps/9); reporting only the latter would
    # return 0.0 for a field sitting exactly at the floor.
    at_or_below_floor = int(np.sum(var <= eps)) if eps > 0 else int(np.sum(var <= 0))
    finite_shares = shares[np.isfinite(shares)]
    worst = float(np.max(finite_shares)) if finite_shares.size else float("nan")
    n_zero_denom = int(np.sum(var <= 0))
    eps_list = list(epsilons) if epsilons else [0.0, 1e-9, eps, 1e-5]
    return FieldReport(
        name=name,
        normalizer=normalizer,
        n_frames=int(var.size),
        min_variance=float(np.min(var)),
        median_variance=float(np.median(var)),
        max_variance=float(np.max(var)),
        floor_share_at_min_variance=worst,
        median_floor_share=float(np.median(shares)),
        frames_floor_determined=dominated,
        fraction_floor_determined=float(dominated / var.size) if var.size else float("nan"),
        frames_at_or_below_floor=at_or_below_floor,
        fraction_at_or_below_floor=(float(at_or_below_floor / var.size)
                                    if var.size else float("nan")),
        # With eps == 0 the floor share is 0 everywhere by construction, so the band table
        # cannot say anything: an earlier version reported a variance of 1e-26 -- the worst
        # conditioning this tool can encounter, where a floorless score is ~1e13x the error --
        # as "well-conditioned" and exited clean. Without a floor the meaningful statement is
        # the denominator's DYNAMIC RANGE within the field: a span of 1e12 in the denominator
        # is a span of 1e6 in the score, produced by the data alone.
        band=("undefined-denominator" if n_nonfinite
              else "degenerate-denominator" if (eps == 0 and n_zero_denom)
              else ("floorless-unbounded"
                    if (_floorless_span(var) >= 1e6 or _floorless_amplification(var) >= 1e6)
                    else "no-floor")
              if eps == 0 else band_for(worst)),
        frames_zero_denominator=n_zero_denom,
        frames_nonfinite_denominator=n_nonfinite,
        sweep=eps_sweep(var, mse, eps_list),
    )


def split_components(name: str, arr: np.ndarray, component_axis: int) -> dict[str, np.ndarray]:
    """Split a vector/tensor field into per-component arrays.

    Benchmarks commonly store a vector field with a trailing component axis, e.g.
    [..., x, y, z, c]. Screening such an array as if the component axis were spatial
    averages a healthy component together with degenerate ones and can report the whole
    field WELL-CONDITIONED when two of its three components sit at 1e-22. Conditioning is
    a per-scored-element property, and these components are scored separately, so they
    must be screened separately.
    """
    arr = np.asarray(arr)
    n = arr.shape[component_axis]
    return {f"{name}.{k}": np.take(arr, k, axis=component_axis) for k in range(n)}


def resolve_layout(shapes: dict, n_spatial: int, component_axis, n_leading):
    """Decide, for every field, exactly one of: split on axis k / screen whole / refuse.

    THIS IS THE ONLY PLACE THAT DECIDES. Earlier versions made this call in three separate
    branches -- inferred, declared-by-count, and per-field mapping -- each with its own rule
    and only one of them disclosing what it had assumed. Every silent false clean this tool
    has shipped came from a branch that reached "screen whole" without saying so. So the
    decision is centralised here and every whole-screening of a field that COULD have carried
    components is recorded in `notes`, whatever route produced it.

    Returns (decisions, notes) where decisions maps name -> int axis or None.
    Raises ValueError to refuse.
    """
    decisions: dict[str, int | None] = {}
    notes: list[str] = []
    # A field can only carry components if it has an axis beyond the spatial block.
    could_split = {k: sh for k, sh in shapes.items() if len(sh) > n_spatial}

    def _check_axis(k, sh, ax):
        nd = len(sh)
        if not (-nd <= ax < nd):
            raise ValueError(f"component axis {ax} is out of range for {k}{list(sh)}")
        if nd - 1 < n_spatial:
            raise ValueError(
                f"splitting {k}{list(sh)} on axis {ax} would leave {nd - 1} axes, fewer than "
                f"the {n_spatial} spatial ones the screen needs.")
        return ax

    if isinstance(component_axis, dict):
        unknown = sorted(set(component_axis) - set(shapes))
        if unknown:
            raise ValueError(f"component_axis names fields that were not supplied: {unknown}")
        # A mapping must speak for every field that could carry components -- including
        # rank n_spatial+1, which an earlier version exempted, letting a channel-last store
        # with no leading axis be omitted and screened whole in silence.
        missing = sorted(k for k in could_split if k not in component_axis)
        if missing:
            raise ValueError(
                f"component_axis does not say what to do with {missing}, and their rank "
                f"leaves it ambiguous. Name an axis for each, or map it to None to screen "
                f"that field whole.")
        for k, sh in shapes.items():
            ax = component_axis.get(k)
            decisions[k] = _check_axis(k, sh, ax) if ax is not None else None
            if ax is None and k in could_split:
                notes.append(("asserted", f"{k}{list(sh)} screened whole at your request"))
        return decisions, notes

    if n_leading is not None:
        if n_leading < 0:
            raise ValueError(f"n_leading must be >= 0, got {n_leading}")
        if component_axis in ("auto", None):
            raise ValueError(
                f"n_leading={n_leading} says how many leading axes the fields carry but not "
                f"WHICH axis holds the components, and assuming the last one is a guess: on a "
                f"channel-first store it would split along a spatial axis. Name the axis too "
                f"(component_axis=-1 channel-last, component_axis={n_leading} channel-first), "
                f"or screen every field whole (component_axis=None, without n_leading).")
        ax = int(component_axis)
        for k, sh in shapes.items():
            nd = len(sh)
            if nd == n_leading + n_spatial + 1:
                decisions[k] = _check_axis(k, sh, ax)
            elif nd == n_leading + n_spatial:
                decisions[k] = None
                # Determined by the declaration, but still a judgement the caller should see:
                # this rank is ALSO what a vector looks like when n_leading is one too high.
                # GRADED "assumed", not "asserted". The caller declared a GLOBAL leading-axis
                # count; the tool inferred from rank that THIS field is a scalar. The note
                # below concedes it may be wrong -- a tool that says it might be wrong is
                # assuming, not being told -- so the run is incomplete and must not exit 0.
                notes.append(("assumed",
                    f"{k}{list(sh)} read as a scalar under n_leading={n_leading} and screened "
                    f"whole; the caller declared the leading-axis count, not this field's "
                    f"kind, so if it is really a vector n_leading is one too high and its "
                    f"components are not being screened. Name it per field to settle this"))
            else:
                raise ValueError(
                    f"{k}{list(sh)} has rank {nd}, neither n_leading+n_spatial="
                    f"{n_leading + n_spatial} (a scalar) nor one more (a vector) under the "
                    f"declared layout. Correct the layout or screen this field separately.")
        return decisions, notes

    if component_axis is None:
        for k, sh in shapes.items():
            decisions[k] = None
            if k in could_split:
                notes.append(("asserted",
                    f"{k}{list(sh)} screened whole; any component axis it carries is averaged "
                    f"into the field, so per-component degeneracy is not visible"))
        return decisions, notes

    if component_axis == "auto":
        # Genuinely undecidable from rank alone: more axes than one leading index plus the
        # spatial ones could be extra leading axes OR a component axis.
        undecidable = {k: sh for k, sh in shapes.items() if len(sh) > n_spatial + 1}
        if undecidable:
            listing = "; ".join(f"{k}{list(sh)}" for k, sh in sorted(undecidable.items()))
            raise ValueError(
                f"ambiguous layout with n_spatial={n_spatial}: {listing}. Rank cannot tell a "
                f"scalar with several leading axes from a vector with a component axis, and "
                f"splitting the wrong one can report a degenerate field as clean, so it is "
                f"not guessed. Resolve it: name the components per field "
                f"(component_axis={{'velocity': -1}}), declare the layout (n_leading=K "
                f"together with component_axis), or screen every field whole "
                f"(component_axis=None, --no-split-components).")
        for k, sh in shapes.items():
            decisions[k] = None
            if k in could_split:
                # NOT asserted by the caller -- the tool guessed the common reading. That
                # makes the screen INCOMPLETE, not clean: a hidden component axis would put
                # a degenerate component out of reach entirely.
                notes.append(("assumed",
                    f"{k}{list(sh)} screened whole -- the one axis beyond the {n_spatial} "
                    f"spatial ones was READ AS a leading index, not asserted to be one. If it "
                    f"holds components, their degeneracy is invisible here; re-run naming it "
                    f"(component_axis=-1) or assert the reading (component_axis=None)"))
        return decisions, notes

    # a bare integer with no layout declaration
    if not could_split:
        # The caller named an axis but no field has an axis to spend on components. Screening
        # whole would silently do something other than what was asked.
        raise ValueError(
            f"component_axis={component_axis} was given, but no field has an axis beyond the "
            f"{n_spatial} spatial ones to hold components: "
            + "; ".join(f"{k}{list(sh)}" for k, sh in sorted(shapes.items()))
            + ". Drop the flag, or correct n_spatial.")
    if could_split:
        listing = "; ".join(f"{k}{list(sh)}" for k, sh in sorted(could_split.items()))
        raise ValueError(
            f"component_axis={component_axis} names an axis but not a layout, and rank "
            f"cannot supply one: {listing}. Declare the layout (n_leading=K), name the "
            f"components per field, or screen every field whole.")
    return {k: None for k in shapes}, notes


def screen_fields(fields: dict[str, np.ndarray], *, eps: float = 1e-7, n_spatial: int = 2,
                  ddof: int = 1, normalizer: str = "variance",
                  component_axis: int | str | None | dict = "auto",
                  n_leading: int | None = None) -> ScreenReport:
    """Screen a whole benchmark split: {field_name: array[..., *spatial]}.

    Layout resolution is delegated entirely to `resolve_layout`, which is the single place
    that decides what is split, what is screened whole, and what is refused -- and which
    records every whole-screening of a possibly-componented field so the report can say so.
    """
    if not fields:
        raise ValueError("no fields given")
    arrs = {k: np.asarray(v) for k, v in fields.items()}
    decisions, _notes = resolve_layout({k: v.shape for k, v in arrs.items()},
                                       n_spatial, component_axis, n_leading)
    expanded: dict[str, np.ndarray] = {}
    for k, v in arrs.items():
        ax = decisions[k]
        if ax is None:
            expanded[k] = v
        else:
            _parts = split_components(k, v, ax)
            _clash = sorted(set(_parts) & set(expanded))
            if _clash:
                raise ValueError(
                    f"splitting {k} produces names that collide with fields already present: "
                    f"{_clash}. One field would silently replace another; rename the inputs.")
            expanded.update(_parts)
    _want = sum(1 if decisions[k] is None else v.shape[decisions[k]]
                for k, v in arrs.items())
    if len(expanded) != _want:
        raise ValueError(f"internal: {len(expanded)} screened fields, expected {_want}")
    fields = expanded
    _incomplete = [t for sev, t in _notes if sev == "assumed"]
    _assumed_note = (("INCOMPLETE: " if _incomplete else "NOTE: ")
                     + " | ".join(t for _, t in _notes)) if _notes else ""
    reports = [screen_array(k, v, eps=eps, n_spatial=n_spatial, ddof=ddof,
                            normalizer=normalizer)
               for k, v in fields.items()]
    if eps == 0:
        # Per-field span cannot see a field that is uniformly quiescent: it is flat, so its
        # own span is ~1, yet its floorless score is orders of magnitude off every other
        # field's. Without a floor the comparability statement is necessarily ACROSS fields.
        finite_mins = [r.min_variance for r in reports if math.isfinite(r.min_variance) and r.min_variance > 0]
        finite_maxs = [r.max_variance for r in reports if math.isfinite(r.max_variance) and r.max_variance > 0]
        if finite_mins and finite_maxs and max(finite_maxs) / min(finite_mins) >= 1e6:
            for r in reports:
                if r.band == "no-floor":
                    r.band = "floorless-unbounded"
    worst_r = max(reports, key=lambda r: (math.isfinite(r.floor_share_at_min_variance)
                                          and r.floor_share_at_min_variance) or -1.0)
    ws = worst_r.floor_share_at_min_variance
    # An undefined denominator is not a clean result. Every threshold test below is false
    # against a NaN, so without this branch corrupt input reports SCREEN NEGATIVE and exits
    # zero -- the tool would certify data it could not actually score.
    n_undef = sum(r.frames_nonfinite_denominator for r in reports)
    if n_undef:
        bad = [r.name for r in reports if r.frames_nonfinite_denominator]
        verdict = (
            f"SCREEN UNDEFINED. {n_undef} frame(s) across {len(bad)} field(s) "
            f"({', '.join(bad[:4])}{', ...' if len(bad) > 4 else ''}) have a non-finite "
            f"denominator, so no conditioning statement can be made about them. This is not "
            f"a clean screen: check the input for NaN/inf, or for frames with too few "
            f"spatial points to form a variance."
        )
    elif ws >= 0.90:
        verdict = (
            f"SCREEN POSITIVE. The least-conditioned field ({worst_r.name}) has a denominator "
            f"that is {100*ws:.2f}% stabilizing constant at its quietest frame. A score "
            f"aggregated over this field is, there, still monotone in the model's error --- "
            f"the ordering survives --- but its SCALE is set by eps rather than by the data, "
            f"so it does not sit on the scale a variance-normalized score is read on and is "
            f"not comparable to a score computed where the denominator is healthy. "
            f"Report the floor share beside the score, or restrict the metric to "
            f"well-conditioned frames. This screen uses no model: it identifies cells to "
            f"check, not cells that are wrong."
        )
    elif ws >= 0.50:
        verdict = (
            f"SCREEN BORDERLINE. {worst_r.name} reaches a floor share of {ws:.4f}. The "
            f"constant is material at the quiet end but does not dominate. Worth reporting "
            f"the share; unlikely on its own to manufacture a headline."
        )
    else:
        verdict = (
            f"SCREEN NEGATIVE over the frames supplied. The worst floor share is {ws:.4g} "
            f"({worst_r.name}), so the denominator is carried by the data everywhere here. "
            f"This bounds only what was screened, not the whole split."
        )
    # eps==0 has its own headline, but an UNDEFINED denominator outranks it: with NaN in
    # the data no conditioning statement can be made at any floor. Guard, or the
    # documented floorless path (--eps 0 --normalizer rms) silently loses the diagnostic.
    if eps == 0 and not n_undef:
        n_zero = sum(1 for r in reports if r.min_variance <= 0)
        verdict = (
            f"NO FLOOR IN USE (eps=0). With no stabilizing constant the score is unbounded "
            f"wherever the denominator reaches zero rather than saturating at a constant. "
            f"Smallest denominator seen: {min(r.min_variance for r in reports):.3e} on "
            f"{min(reports, key=lambda r: r.min_variance).name}"
            + (f"; {n_zero} field(s) reach exactly zero." if n_zero else
               "; no field reaches exactly zero over the frames supplied.")
        )
    return ScreenReport(eps=eps, n_spatial=n_spatial, ddof=ddof, normalizer=normalizer,
                        fields=reports,
                        least_conditioned_field=worst_r.name, worst_floor_share=ws,
                        verdict=(verdict + ("\n\n" + _assumed_note if _assumed_note else "")))
