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
the dominant term in the denominator wherever the target is quiescent, and the reported
number then measures the constant rather than the model.

This tool answers one question about YOUR benchmark, using no model at all:

    for each field and each frame, what share of the metric's denominator is supplied
    by the stabilizing constant rather than by the data?

        floor_share = eps / (Var + eps)

A field at floor_share ~ 0 is well conditioned: the score means what it appears to
mean. A field approaching 1 is floor-dominated: the score is a function of `eps`, is
not comparable to a score computed where the denominator is healthy, and should not be
aggregated with one.

It is a SCREEN, not a verdict. A floor-dominated denominator is necessary but not
sufficient for a misleading score -- whether the reading is actually inflated also
depends on the model's error, which no data-only pass can see. The screen tells you
which cells to check, not which cells are wrong.

USAGE
-----
    normscreen data.h5 --fields density velocity_x velocity_y
    normscreen data.h5 --auto --eps 1e-7 --json report.json
    normscreen --npy field.npy --spatial-dims 3

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
    (1.01, "floor-determined"),      # the score is essentially a function of eps
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

    With eps == 0 the metric has no stabilizing constant, so the constant's share is
    0 wherever the denominator is positive -- and at Var == 0 the expression is 0/0.
    That case is not "half floored": it is a metric dividing by exactly zero, which is
    a different and worse failure than saturation. We return 0.0 there (the constant
    genuinely supplies nothing) and surface the condition through the unbounded count
    in eps_sweep and through the eps==0 verdict, rather than propagating a NaN that
    would silently poison a max() or a band lookup.
    """
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
    frames_floor_dominated: int
    fraction_floor_dominated: float
    band: str
    frames_at_or_below_floor: int = 0
    fraction_at_or_below_floor: float = 0.0
    frames_zero_denominator: int = 0
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
                f"{100*f.fraction_floor_dominated:6.1f}%  {f.band}"
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
    dominated = int(np.sum(shares >= 0.90))
    # The threshold statistic a benchmark should publish is the fraction of scored elements
    # whose denominator is at or below the floor, i.e. Var <= eps, i.e. floor share >= 0.5.
    # That is NOT the same as the 0.90 band (Var <= eps/9); reporting only the latter would
    # return 0.0 for a field sitting exactly at the floor.
    at_or_below_floor = int(np.sum(var <= eps)) if eps > 0 else int(np.sum(var <= 0))
    worst = float(np.max(shares))
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
        frames_floor_dominated=dominated,
        fraction_floor_dominated=float(dominated / var.size) if var.size else float("nan"),
        frames_at_or_below_floor=at_or_below_floor,
        fraction_at_or_below_floor=(float(at_or_below_floor / var.size)
                                    if var.size else float("nan")),
        band=("degenerate-denominator" if (eps == 0 and n_zero_denom)
              else band_for(worst)),
        frames_zero_denominator=n_zero_denom,
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


def screen_fields(fields: dict[str, np.ndarray], *, eps: float = 1e-7, n_spatial: int = 2,
                  ddof: int = 1, normalizer: str = "variance",
                  component_axis: int | str | None = "auto") -> ScreenReport:
    """Screen a whole benchmark split: {field_name: array[..., *spatial]}.

    `component_axis` controls per-component screening and DEFAULTS TO "auto": any field
    carrying more axes than one leading index plus `n_spatial` is assumed to store
    components on its last axis and is split before screening. This is the safe default,
    because the failure it prevents is silent: averaging a healthy component together
    with degenerate ones reports the whole field well-conditioned. Pass an explicit int
    to name the axis, or None to disable splitting entirely.
    """
    if not fields:
        raise ValueError("no fields given")
    if component_axis == "auto":
        # split only where there is genuinely an extra axis beyond [leading..., *spatial]
        component_axis = -1 if any(
            np.asarray(v).ndim > n_spatial + 1 for v in fields.values()) else None
    if component_axis is not None:
        expanded: dict[str, np.ndarray] = {}
        for k, v in fields.items():
            v = np.asarray(v)
            # only split when there IS an extra axis beyond the spatial ones
            if v.ndim > n_spatial + 1 or (component_axis == -1 and v.ndim > n_spatial):
                expanded.update(split_components(k, v, component_axis))
            else:
                expanded[k] = v
        fields = expanded
    reports = [screen_array(k, v, eps=eps, n_spatial=n_spatial, ddof=ddof,
                            normalizer=normalizer)
               for k, v in fields.items()]
    worst_r = max(reports, key=lambda r: r.floor_share_at_min_variance)
    ws = worst_r.floor_share_at_min_variance
    if ws >= 0.90:
        verdict = (
            f"SCREEN POSITIVE. The least-conditioned field ({worst_r.name}) has a denominator "
            f"that is {100*ws:.2f}% stabilizing constant at its quietest frame. A score "
            f"aggregated over this field is, there, largely a function of eps rather than of "
            f"the model, and is not comparable to a score computed where the denominator is "
            f"healthy. Report the floor share beside the score, or restrict the metric to "
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
    if eps == 0:
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
                        verdict=verdict)
