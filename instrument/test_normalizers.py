"""Convention and regression checks for normscreen.

These are SCRIPTS, not pytest cases: each check returns a failure COUNT and the
module exits non-zero if any is non-zero. The functions are deliberately NOT named
test_* -- pytest treats a non-None return as a pass, so under pytest a fully failing
suite reported "passed". Run them directly:

    python3 test_normalizers.py
    python3 test_against_paper.py
"""
#!/usr/bin/env python3
"""Regression tests for the two normalizer conventions.

These encode the finding that motivated normalizer support: two peer physics-ML
benchmarks divide by different quantities, and the choice decides whether a quiescent
field is scoreable at all.

  The Well   VRMSE  : divide by (Var + eps),  eps = 1e-7   -> a fixed error's amplification saturates at 1/sqrt(eps)
  PDEBench   nRMSE  : divide by sqrt(mean(x^2)), no eps    -> unbounded at zero signal

Run:  python3 test_normalizers.py
"""
import sys, os
import pathlib
import numpy as np
from normscreen.__main__ import _exit_status
from normscreen import band_for

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from normscreen.screen import floor_share, screen_fields, spatial_rms2, spatial_variance

fails = []


def check(name, cond, detail=""):
    print(f"  [{'OK  ' if cond else 'FAIL'}] {name}" + (f"  {detail}" if detail else ""))
    if not cond:
        fails.append(name)


print("=== RMS vs variance: the same field, opposite verdicts ===")
rng = np.random.default_rng(0)
# oscillation about a large offset: RMS ~ 1, variance ~ 1e-12
f = {"offset": 1.0 + 1e-6 * rng.standard_normal((10, 64, 64))}
rv = screen_fields(f, eps=1e-7, n_spatial=2, normalizer="variance").fields[0]
rr = screen_fields(f, eps=1e-7, n_spatial=2, normalizer="rms").fields[0]
check("variance normalizer sees it as floor-determined", rv.band == "floor-determined",
      f"share={rv.floor_share_at_min_variance:.6f}")
check("rms normalizer sees it as well-conditioned", rr.band == "well-conditioned",
      f"share={rr.floor_share_at_min_variance:.6f}")
check("RMS^2 = Var + mean^2 identity holds",
      np.allclose(spatial_rms2(f["offset"], 2),
                  spatial_variance(f["offset"], 2, ddof=0) + f["offset"].mean(axis=(1, 2))**2))

print("\n=== eps = 0 (a metric with no stabilizing constant) ===")
z = {"identically_zero": np.zeros((5, 32, 32)), "healthy": rng.standard_normal((5, 32, 32))}
r0 = screen_fields(z, eps=0.0, n_spatial=2, normalizer="rms")
zf = [x for x in r0.fields if x.name == "identically_zero"][0]
check("floor share is 0 at eps=0, never NaN", np.isfinite(zf.floor_share_at_min_variance)
      and zf.floor_share_at_min_variance == 0.0)
check("zero-denominator frames counted", zf.frames_zero_denominator == 5,
      f"n={zf.frames_zero_denominator}")
check("band reports degeneracy, not a floor band", zf.band == "degenerate-denominator")
check("verdict names the unboundedness", "unbounded" in r0.verdict)
check("scalar floor_share at eps=0 is 0.0", floor_share(0.0, 0.0) == 0.0)

print("\n=== Lesson-1 threshold statistic (regression: caught by the Codex gate leg) ===")
# The statistic a benchmark should publish is the fraction with Var <= eps (share >= 0.5).
# An earlier build reported only the >=0.90 band (Var <= eps/9) and therefore returned 0.0
# for a field sitting exactly at the floor -- silently answering a different question.
x = rng.standard_normal((6, 32, 32))
x *= (1e-7 / x.var(axis=(1, 2), ddof=1).mean()) ** 0.5      # drive Var to ~eps
at_eps = screen_fields({"at_eps": x}, eps=1e-7, n_spatial=2).fields[0]
check("floor share is ~0.5 when Var ~ eps", 0.4 < at_eps.floor_share_at_min_variance < 0.6,
      f"{at_eps.floor_share_at_min_variance:.4f}")
check("Lesson-1 fraction (Var<=eps) is non-zero there",
      at_eps.fraction_at_or_below_floor > 0.0,
      f"{at_eps.fraction_at_or_below_floor:.3f}")
check("the 0.90 band is a STRICTER, separate statistic",
      at_eps.fraction_floor_determined < at_eps.fraction_at_or_below_floor,
      f"{at_eps.fraction_floor_determined:.3f} < {at_eps.fraction_at_or_below_floor:.3f}")
# and a mutation test: swapping the threshold back to 0.90 must change the answer
check("the two thresholds are not interchangeable",
      at_eps.fraction_at_or_below_floor != at_eps.fraction_floor_determined)

print("\n=== saturation vs unboundedness ===")
# What a floor bounds is the AMPLIFICATION of a fixed error, not the score. An earlier
# version of this block fixed MSE = 1 and called sqrt(MSE/eps) "the worst achievable
# score"; at MSE = 1 the amplification factor and the score coincide numerically, which
# is exactly why the error was invisible here. Both properties are now pinned, including
# the one the mistaken version asserted the negation of.
for _mse in (1e-4, 1.0, 1e4):
    _amp = float(np.sqrt(_mse / (0.0 + 1e-7)) / np.sqrt(_mse))
    check(f"floored: amplification of a fixed error saturates at 1/sqrt(eps) (MSE={_mse:g})",
          abs(_amp - 1 / np.sqrt(1e-7)) < 1e-6, f"{_amp:.1f}")
_big = float(np.sqrt(1e4 / (0.0 + 1e-7)))
check("floored: the SCORE itself is NOT bounded by 1/sqrt(eps) -- it grows with the error",
      _big > 1 / np.sqrt(1e-7), f"{_big:.4g} > {1 / np.sqrt(1e-7):.4g}")
with np.errstate(divide="ignore"):
    unbounded = float(np.sqrt(np.array([1.0]) / (0.0 + 0.0))[0])
check("floorless metric is genuinely infinite", not np.isfinite(unbounded))

# The module-level checks above tally into `fails`; the regression guards defined below
# add to it in __main__, so both sets run in one invocation and one exit status.


# ---------------------------------------------------------------------------
# Regression guards. Each of these failed in a released version of this file.
# ---------------------------------------------------------------------------

def _check(label, cond, detail=""):
    print(f"  [{'OK  ' if cond else 'FAIL'}] {label}  {detail}")
    return 0 if cond else 1


def check_layout_ambiguity_is_refused_not_guessed():
    """A scalar [traj, time, y, x] must not be split along a SPATIAL axis.

    The released default took the last axis whenever rank exceeded one leading index
    plus the spatial ones. For density[2,4,16,16] with n_spatial=2 that split the field
    into 16 pseudo-components along x, and because screen_array reads the TRAILING axes
    as spatial it then measured variance over the wrong axes -- which can turn a
    degenerate field into a clean report. Rank cannot distinguish this layout from a
    vector [time, y, x, component], so the screen must refuse rather than pick.
    """
    bad = 0
    rng = np.random.default_rng(0)
    scalar = rng.standard_normal((2, 4, 16, 16)) * 1e-11
    try:
        screen_fields({"density": scalar}, eps=1e-7, n_spatial=2)
        bad += _check("ambiguous layout refused", False, "no error raised")
    except ValueError as e:
        bad += _check("ambiguous layout refused", "ambiguous layout" in str(e))
        bad += _check("refusal names the remedies",
                      "component_axis" in str(e) and "n_leading" in str(e))
    whole = screen_fields({"density": scalar}, eps=1e-7, n_spatial=2, component_axis=None)
    bad += _check("explicit whole keeps one field with all frames",
                  len(whole.fields) == 1 and whole.fields[0].n_frames == 8,
                  f"{len(whole.fields)} field(s), {whole.fields[0].n_frames} frames")
    vec = rng.standard_normal((4, 16, 16, 3))
    split = screen_fields({"v": vec}, eps=1e-7, n_spatial=2, component_axis=-1,
                          n_leading=1)
    bad += _check("declared channel-last splits into 3", len(split.fields) == 3)
    first = screen_fields({"v": np.moveaxis(vec, -1, 0)}, eps=1e-7, n_spatial=2,
                          component_axis=0, n_leading=1)
    bad += _check("declared channel-first is honoured too", len(first.fields) == 3)
    return bad


def check_nonfinite_denominator_is_not_a_clean_screen():
    """NaN input used to report SCREEN NEGATIVE and exit 0.

    Every threshold comparison against a NaN is false, so a corrupt split read as
    'no frame is floor-dominated'. Undefined is not well-conditioned.
    """
    bad = 0
    rng = np.random.default_rng(1)
    a = rng.standard_normal((6, 16, 16))
    a[2, 0, 0] = np.nan
    rep = screen_fields({"corrupt": a}, eps=1e-7, n_spatial=2, component_axis=None)
    bad += _check("band is undefined-denominator",
                  rep.fields[0].band == "undefined-denominator", rep.fields[0].band)
    bad += _check("non-finite frames counted",
                  rep.fields[0].frames_nonfinite_denominator == 1)
    bad += _check("verdict is SCREEN UNDEFINED", rep.verdict.startswith("SCREEN UNDEFINED"))
    bad += _check("exit status is non-zero", _exit_status(rep) == 1)
    return bad


def check_band_field_names_match_the_band_table():
    """The counted band is >=0.90, which CONDITION_BANDS calls floor-determined.

    The fields were previously named *_floor_dominated, which names the 0.50-0.90 band,
    so the JSON contradicted the band table printed beside it.
    """
    bad = 0
    rng = np.random.default_rng(2)
    a = rng.standard_normal((4, 8, 8)) * 1e-12
    rep = screen_fields({"q": a}, eps=1e-7, n_spatial=2, component_axis=None)
    f = rep.fields[0]
    bad += _check("field is named *_floor_determined", hasattr(f, "frames_floor_determined"))
    bad += _check("old contradictory name is gone", not hasattr(f, "frames_floor_dominated"))
    bad += _check("band_for(>=0.90) says floor-determined",
                  band_for(0.95) == "floor-determined", band_for(0.95))
    bad += _check("band_for(0.7) says floor-dominated",
                  band_for(0.70) == "floor-dominated", band_for(0.70))
    return bad


def check_no_layout_can_silently_downgrade_a_degenerate_field():
    """Every degenerate component must be visible, or the run must not read as clean.

    SEVEN prior versions of this check passed while the defect was live, each for a
    different reason: wrong case; a fixture degenerate on every axis; the dangerous mode
    always refused; a max over FIELDS masking a vector; the only assertion gated on eps>0;
    a max over COMPONENTS with the last one never degenerate; and finally two eps=0
    assertions that were unsatisfiable by construction. Every one of them printed a
    reassuring summary line.

    So this asserts PER DERIVED COMPONENT, sweeps which component is degenerate (each
    singly, so no max can mask), asserts at eps=0 on properties that are actually reachable
    there, and requires that a run which did not screen a component either finds it or does
    not exit 0. `validate_guard_fires` below proves each assertion can fail.
    """
    bad = 0
    rng = np.random.default_rng(11)

    def deg_vec(lead, sp, axis_last, bad_idx):
        shape = lead + sp + (3,) if axis_last else lead + (3,) + sp
        a = np.empty(shape, dtype=np.float64)
        for c in range(3):
            base = rng.standard_normal(lead + sp)
            if c in bad_idx:
                idx = (np.arange(int(np.prod(lead))).reshape(lead + (1,) * len(sp))
                       if lead else 0.0)
                base = 1e-13 * base + idx
            if axis_last:
                a[..., c] = base
            else:
                a[(slice(None),) * len(lead) + (c,)] = base
        return a

    executed = refused = 0
    for n_spatial in (2, 3):
        for n_lead in (0, 1, 2):
            for axis_last in (True, False):
                # (0,1,2) matters: with any healthy component present the eps=0 cross-field
                # rescue fires, so the eps=0 leg was satisfied by a max over fields
                # rather than by the component under assertion.
                for bad_idx in ((0,), (1,), (2,), (0, 1), (0, 1, 2)):
                    sp = (8,) * n_spatial
                    lead = (2, 5)[:n_lead]
                    vec = deg_vec(lead, sp, axis_last, bad_idx)
                    cax = -1 if axis_last else n_lead
                    fields = {"v": vec}
                    for mode, kw in (
                        ("auto", dict(component_axis="auto")),
                        ("named", dict(component_axis={"v": cax})),
                        ("declared", dict(component_axis=cax, n_leading=n_lead)),
                        # a MISDECLARED count: the caller is one too high, so the tool reads
                        # the vector as a scalar. Previously graded "asserted" and exited 0.
                        ("misdeclared", dict(component_axis=cax, n_leading=n_lead + 1)),
                        ("whole", dict(component_axis=None)),
                    ):
                        for eps in (1e-7, 0.0):
                            try:
                                r = screen_fields(fields, eps=eps, n_spatial=n_spatial, **kw)
                            except ValueError:
                                refused += 1
                                continue
                            executed += 1
                            names = {f.name for f in r.fields}
                            split = names != {"v"}
                            if split:
                                # every degenerate component must be individually flagged
                                for c in bad_idx:
                                    f = next((x for x in r.fields if x.name == f"v.{c}"), None)
                                    if f is None:
                                        bad += _check(f"nsp={n_spatial} lead={n_lead} {mode} "
                                                      f"eps={eps:g} bad={bad_idx}", False,
                                                      f"component v.{c} missing from report")
                                        continue
                                    flagged = (f.floor_share_at_min_variance >= 0.99
                                               if eps > 0 else
                                               f.band in ("floorless-unbounded",
                                                          "degenerate-denominator"))
                                    if not flagged:
                                        bad += _check(f"nsp={n_spatial} lead={n_lead} {mode} "
                                                      f"eps={eps:g} bad={bad_idx}", False,
                                                      f"v.{c} NOT FLAGGED "
                                                      f"(share={f.floor_share_at_min_variance:.3g}, "
                                                      f"band={f.band})")
                            else:
                                # Not split. The contract depends on WHO decided:
                                #  - the caller asserted it ("whole"/"declared"): the tool
                                #    makes no component claim, but MUST disclose that it is
                                #    not making one;
                                #  - the tool guessed it ("auto"): the screen is incomplete
                                #    and must not exit 0.
                                # Assert the GRADING, not merely that something was said.
                                # An earlier version accepted "NOTE:" OR "INCOMPLETE:" and
                                # gated the exit check to mode=="auto" -- so the misdeclared
                                # leg ran 120 times and asserted nothing that distinguished
                                # the fix from the defect. Reverting "assumed"->"asserted"
                                # left every packaged suite green. The two gradings mean
                                # different things and are now asserted separately.
                                incomplete = "INCOMPLETE:" in r.verdict
                                noted = "NOTE:" in r.verdict
                                tag = (f"nsp={n_spatial} lead={n_lead} {mode} "
                                       f"eps={eps:g} bad={bad_idx}")
                                if mode in ("auto", "misdeclared"):
                                    # the tool INFERRED the field was a scalar
                                    if not incomplete:
                                        bad += _check(tag, False,
                                                      "inferred the layout but did not "
                                                      "report INCOMPLETE")
                                    if _exit_status(r) == 0:
                                        bad += _check(tag, False,
                                                      "inferred the layout and still exited 0")
                                else:
                                    # the caller ASSERTED it: clean run, but it must say
                                    # which fields it did not screen per component
                                    if not (noted or incomplete):
                                        bad += _check(tag, False,
                                                      "whole-screened a vector WITHOUT "
                                                      "disclosing it")
    total = executed + refused
    print(f"  [OK  ] swept {total} configurations: {executed} executed, {refused} refused, "
          f"every degenerate component flagged or the run marked not-clean")
    bad += _check("the sweep EXECUTED a majority of configurations",
                  executed >= total // 2, f"{executed}/{total}")
    return bad


def validate_guard_fires():
    """Prove the sweep's assertions CAN fail. A guard never seen to fire is not a guard.

    This is the step whose absence let seven broken versions ship: each printed a summary
    line that looked like coverage. Here every assertion class is driven by a deliberately
    broken screen and must produce a non-zero count.
    """
    import normscreen.screen as S
    bad = 0
    real_split = S.split_components

    def run_with(patch_split=None):
        if patch_split:
            S.split_components = patch_split
        try:
            import io, contextlib
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                n = check_no_layout_can_silently_downgrade_a_degenerate_field()
            return n
        finally:
            S.split_components = real_split

    # (1) a split that drops the last component entirely
    def drop_last(name, arr, axis):
        d = real_split(name, arr, axis)
        return {k: v for k, v in list(d.items())[:-1]}
    n1 = run_with(patch_split=drop_last)
    bad += _check("guard fires when a component is dropped", n1 > 0, f"{n1} failures")

    # Per-LEG validation. An aggregate count > 0 is satisfied by the eps>0 leg alone, which
    # is how the eps=0 leg stayed vacuous through two revisions.
    import normscreen.screen as S2
    real_band_helper = S2._floorless_amplification
    try:
        S2._floorless_amplification = lambda v: 1.0       # kill the eps=0 absolute band
        import io, contextlib
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            n_eps0 = check_no_layout_can_silently_downgrade_a_degenerate_field()
        bad += _check("guard fires on the eps=0 leg specifically", n_eps0 > 0,
                      f"{n_eps0} failures")
    finally:
        S2._floorless_amplification = real_band_helper

    # (2) a split that returns the SAME healthy slice for every component
    def alias_healthy(name, arr, axis):
        d = real_split(name, arr, axis)
        keys = list(d)
        good = max(d.values(), key=lambda x: float(np.var(x)))
        return {k: good for k in keys}
    # Flip the grading the (a) fix introduced. If the guard cannot see this, the
    # misdeclared/declared legs assert nothing -- which is exactly how it shipped.
    import normscreen.screen as S3
    _src = pathlib.Path(S3.__file__).read_text()
    try:
        _mut = _src.replace('notes.append(("assumed",\n                    f"{k}{list(sh)} read as a scalar',
                            'notes.append(("asserted",\n                    f"{k}{list(sh)} read as a scalar')
        if _mut != _src:
            pathlib.Path(S3.__file__).write_text(_mut)
            import importlib
            importlib.reload(S3)
            import normscreen as _ns
            importlib.reload(_ns)
            import io, contextlib
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                n_grade = check_no_layout_can_silently_downgrade_a_degenerate_field()
            bad += _check("guard fires when the inferred grading is flipped to 'asserted'",
                          n_grade > 0, f"{n_grade} failures")
        else:
            bad += _check("grading mutation applied", False, "anchor not found")
    finally:
        pathlib.Path(S3.__file__).write_text(_src)
        import importlib
        importlib.reload(S3)
        import normscreen as _ns2
        importlib.reload(_ns2)

    n2 = run_with(patch_split=alias_healthy)
    bad += _check("guard fires when components are aliased to a healthy one", n2 > 0,
                  f"{n2} failures")
    return bad


def check_per_field_and_declared_layout_still_split_vectors():
    """The safe modes must still do the useful thing, or the fix is just a refusal."""
    bad = 0
    rng = np.random.default_rng(12)
    vec = rng.standard_normal((2, 5, 8, 8, 3))
    r = screen_fields({"v": vec}, eps=1e-7, n_spatial=2, component_axis={"v": -1})
    bad += _check("per-field dict splits the vector", len(r.fields) == 3)
    r2 = screen_fields({"v": vec}, eps=1e-7, n_spatial=2, component_axis=-1, n_leading=2)
    bad += _check("declared n_leading splits the vector", len(r2.fields) == 3)
    r3 = screen_fields({"v": rng.standard_normal((5, 8, 8, 3))}, eps=1e-7, n_spatial=2,
                       component_axis=-1, n_leading=1)
    bad += _check("declared single-layout file still splits", len(r3.fields) == 3)
    try:
        screen_fields({"v": rng.standard_normal((5, 8, 8, 3))}, eps=1e-7, n_spatial=2,
                      component_axis=-1)
        bad += _check("a bare global axis is refused without a layout", False, "no error")
    except ValueError as e:
        bad += _check("a bare global axis is refused without a layout",
                      "names an axis but not a layout" in str(e))
    try:
        screen_fields({"s": rng.standard_normal((2, 5, 8, 8)),
                       "v": vec}, eps=1e-7, n_spatial=2, component_axis=-1)
        bad += _check("mixed ranks under one global axis are refused", False, "no error")
    except ValueError as e:
        bad += _check("mixed ranks under one global axis are refused",
                      "names an axis but not a layout" in str(e))
    try:
        screen_fields({"flat": rng.standard_normal((6, 3))}, eps=1e-7, n_spatial=2,
                      component_axis=-1)
        bad += _check("splitting below the spatial rank is refused", False, "no error")
    except ValueError as e:
        bad += _check("splitting below the spatial rank is refused",
                      "no field has an axis beyond" in str(e) or "fewer than" in str(e))
    return bad


print("\n=== regression guards (layout / non-finite / band naming) ===")


if __name__ == "__main__":
    n = (len(fails)
         + check_layout_ambiguity_is_refused_not_guessed()
         + check_nonfinite_denominator_is_not_a_clean_screen()
         + check_band_field_names_match_the_band_table()
         + check_no_layout_can_silently_downgrade_a_degenerate_field()
         + validate_guard_fires()
         + check_per_field_and_declared_layout_still_split_vectors())
    print("\nPASS" if not n else f"\nFAIL: {n} check(s)" + (f" (module-level: {fails})" if fails else ""))
    raise SystemExit(1 if n else 0)
