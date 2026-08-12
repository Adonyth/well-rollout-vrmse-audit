#!/usr/bin/env python3
"""Check the claims this package makes *about itself*.

Every other stage of `verify.py` checks a number against the data that produced
it. Nothing checked the prose: the README, the MANIFEST, the instrument's README,
the docstrings, and the comments inside the harness all describe what this package
binds, packages, verifies and produces -- and those descriptions drifted
repeatedly while the numbers stayed correct. A file was renamed and three
docstrings kept naming the old one; a dependency floor was raised in the code and
not in `requirements.txt`; a size was measured on two subtrees and quoted as the
total; a producer string naming scripts that were never shipped was stamped into a
packaged fixture.

None of those is a wrong result. Each is the package asserting something untrue
about itself, which is the exact failure mode the paper is about, so it is worth a
machine check rather than a reviewer's attention.

Ten families are checkable without judgement:

  paths     -- every repo-relative path named in the prose or in a docstring resolves,
               at the location it names when it names one
  counts    -- the fixture counts the prose pins, and the enumerated check total,
               recomputed from the tree and from verify.py's own constants
  stages    -- the README transcript lists, in print order, every stage that gates the
               exit code, and says so where a stage is conditional
  cli       -- every invocation the package advertises uses flags its parser accepts
               and declares a layout where the instrument requires one
  numbers   -- the worked numeric examples evaluate to the values they claim
  io        -- the row-file writer and reader round-trip and are single-sourced
  markers   -- a ledger claiming its withdrawals are marked inline has them marked
  producers -- no shipped script defaults to writing outside the package
  commands  -- no documented command carries an abbreviated or placeholder value
  deps      -- the declared dependency floor admits the APIs the scripts call

**What this does not do.** It is one-directional: it checks that what the package
says is true, never that everything the package ships is said. It cannot judge whether
a description is *apt*. The dependency rule knows about one API, not the dependency
graph. The count rules bind the quantities named below, not every number in the prose.
Reviewers found ten discrepancies of this class by hand before it existed and six more
after its first version, so read a pass as "these families are clean", not as
"the package is honest about itself".

Run standalone (`python3 scripts/check_self_claims.py`) or via `verify.py`.
"""

from __future__ import annotations

import gzip
import json
import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent

# Prose surfaces. Docstrings/comments in .py are swept separately (below), because
# their path-like tokens are not backticked.
PROSE = ["README.md", "MANIFEST.md", "instrument/README.md"]
CODE = sorted(
    str(p.relative_to(ROOT))
    for p in list(ROOT.glob("scripts/*.py"))
    + list(ROOT.glob("instrument/*.py"))
    + list(ROOT.glob("instrument/normscreen/*.py"))
    + list(ROOT.glob("instrument/examples/*.py"))
    + [ROOT / "verify.py"]
)

# A backticked token is treated as a path claim when it looks like one: it has a
# known suffix, or a directory separator, and no spaces or shell metacharacters.
_SUFFIXES = (".py", ".json", ".json.gz", ".md", ".txt", ".sty", ".tex", ".toml", ".cfg")
_PATHY = re.compile(r"`([A-Za-z0-9_./{},*~-]+)`")

# Brace/star globs are claims about a *set*; require at least one match.
_BRACE = re.compile(r"\{([^{}]*)\}")


def _expand(tok: str) -> list[str]:
    """Expand a single level of {a,b} alternation. Not recursive -- none is used."""
    m = _BRACE.search(tok)
    if not m:
        return [tok]
    return [tok[: m.start()] + alt + tok[m.end():] for alt in m.group(1).split(",")]


def _is_path_claim(tok: str) -> bool:
    """A token is a path claim only if it ends in a source/data suffix.

    Directory tokens are excluded: several are run-time outputs that a clean
    checkout legitimately does not have yet. Ratio-like tokens (``6.72/2.84``)
    survive the tokenizer and are excluded by the suffix requirement.
    """
    if " " in tok or tok.startswith(("http", "-", "$")):
        return False
    if not tok.endswith(_SUFFIXES):
        return False
    stem = tok.rsplit("/", 1)[-1]
    for suf in _SUFFIXES:
        if stem.endswith(suf):
            stem = stem[: -len(suf)]
            break
    # A bare suffix is the tail of a glob the tokenizer split on `*`; a stem with
    # several hyphens is a hyphenated English phrase, not a filename.
    return bool(stem) and stem.count("-") <= 2


# Files named by these surfaces that belong to something *else* -- the library
# under audit, the paper repo, the private working lane, or the reader's own data.
# Naming them is not a claim about this package's contents.
EXTERNAL = {
    # the_well, the library being audited
    "datasets.py", "datamodule.py", "training.py", "normalization.py",
    # the paper repo / private lane, deliberately not shipped here
    "numbers.json", "report_numbers.json", "extract_numbers.py", "make_figures.py",
    "DIGEST.md", "rt_denominator_audit.py",
    # written at run time, absent from a clean checkout by design
    "summary.json",
    # the reader's own inputs, in usage examples
    "your_data.h5", "chapter.tex",
    # one level up in the submission tree, swept by check_withdrawal_markers
    "RT-QUANT.md",
    # another project's files, named in the worked example's provenance note
    "metrics.py",
    # files the *reader* creates by running an advertised command
    "out.json", "report.json",
}
# .tex chapters live in the paper repo; this package ships none.
_EXTERNAL_SUFFIX = (".tex",)


def _exists(tok: str) -> bool:
    p = ROOT / tok
    if p.exists():
        return True
    if "*" in tok:
        head, _, tail = tok.rpartition("/")
        try:
            return any((ROOT / head).glob(tail)) if head else any(ROOT.glob(tok))
        except (ValueError, OSError):
            return False
    return False


def _resolve(tok: str, surface: str = "") -> bool:
    """Resolve a named file.

    A bare basename (``aggregate_results.py``) is a claim that the file is *in the
    package*, and is satisfied anywhere in the tree -- prose names scripts that way
    far more often than by full path. A token that carries a directory component
    (``scripts/aggregate_results.py``) is a claim about *that* location and must
    resolve there. An earlier version fell back to the basename in both cases, so a
    real filename under a directory it does not live in -- the verifier placed inside
    the fixtures tree, the manifest inside the scripts tree -- passed as a valid claim.
    """
    if _exists(tok):
        return True
    if "/" in tok:
        # A path may be written relative to the package root, relative to the
        # directory of the file that names it (the instrument's README says
        # the worked example by its path under that README's own directory), or
        # relative to the submission tree one
        # level up, with the harness directory as the leading component. All three
        # are the same claim; none of them licenses a wrong directory.
        if tok.startswith("repro-harness/") and _exists(tok[len("repro-harness/"):]):
            return True
        here = (ROOT / surface).parent if surface else ROOT
        while True:
            if (here / tok).exists():
                return True
            if here == ROOT or ROOT not in here.parents:
                break
            here = here.parent
        return False
    return any(ROOT.rglob(tok))


def check_paths() -> list[str]:
    """Every repo-relative path the package names about itself must exist."""
    bad: list[str] = []
    for rel in PROSE + CODE:
        f = ROOT / rel
        if not f.exists():
            bad.append(f"{rel}: surface itself is missing")
            continue
        text = f.read_text(encoding="utf-8", errors="replace")
        if rel.endswith(".py"):
            # Only docstrings and comments -- code strings are the program, not a claim.
            lines = []
            for ln in text.splitlines():
                st = ln.strip()
                if st.startswith("#"):
                    lines.append(st)
            docs = re.findall(r'"""(.*?)"""', text, re.S)
            text = "\n".join(lines + docs)
            toks = re.findall(r"[A-Za-z0-9_./{}-]+", text)
        else:
            toks = _PATHY.findall(text)
        for tok in toks:
            tok = tok.rstrip(".,;:")
            if not _is_path_claim(tok):
                continue
            for cand in _expand(tok):
                base = cand.rsplit("/", 1)[-1]
                if base in EXTERNAL or cand.endswith(_EXTERNAL_SUFFIX):
                    continue
                if cand.startswith(("paper/", "the_well/", "~", "hf-cache/", "lanes/", "lane-3/")):
                    continue
                if not _resolve(cand, rel):
                    bad.append(f"{rel}: names `{cand}`, which is nowhere in the package")
    return bad


def _g2_steps() -> int:
    """The Gate-2 horizon, read from the fixture rather than from any prose."""
    return int(json.loads((ROOT / "fixtures/gate2/alignment_fixture.json")
                          .read_text(encoding="utf-8"))["rollout_steps"])


def _derive_check_total() -> int | None:
    """Recompute the enumerated check total from verify.py's own constants."""
    import importlib.util

    spec = importlib.util.spec_from_file_location("_p3_verify_selfclaims", ROOT / "verify.py")
    if spec is None or spec.loader is None:
        return None
    mod = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(mod)
    except Exception:
        return None
    try:
        return (len(mod.CHECKS)
                + len(mod.FIELDSPLIT_CHECKS)
                + len(mod.AUDIT_TRAJ_KEYS) * 5
                + len(mod.AUDIT_TRAJ_KEYS) * 4
                + len(mod.SPATIAL_MEAN_BASELINE_WINDOWS))
    except AttributeError:
        return None


def check_counts() -> list[str]:
    """Quantities the prose pins must equal what the tree actually holds."""
    bad: list[str] = []
    inst = (ROOT / "instrument" / "README.md").read_text(encoding="utf-8")
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    man = (ROOT / "MANIFEST.md").read_text(encoding="utf-8")

    n_map = len(json.loads((ROOT / "fixtures/generalization/benchmark_map/MAP.json").read_text()))
    n_win = 0
    for p in sorted((ROOT / "fixtures/rb_models").glob("*.json.gz")):
        with gzip.open(p, "rt") as fh:
            n_win += len(json.load(fh)["rows"])

    # Both counts are pinned in `instrument/test_against_paper.py` as well, so the
    # prose, the test and the tree are held to one number rather than to each other.
    vpy = (ROOT / "verify.py").read_text(encoding="utf-8")
    for surface, text in (("instrument/README.md", inst), ("README.md", readme),
                          ("MANIFEST.md", man), ("verify.py", vpy)):
        for pattern, actual, what in (
            (r"MAP\.json`?,? (\d+) rows", n_map, "rows in MAP.json"),
            (r"(\d+) rows reproduced", n_map, "rows in MAP.json"),
            (r"\((\d+) rows, [\d,]+ windows\)", n_map, "rows in MAP.json"),
            (r"json\.gz`?,? (\d+) windows", n_win, "packaged RB windows"),
            (r"(\d+) per-window variances", n_win, "packaged RB windows"),
            (r"\(\d+ rows, ([\d,]+) windows\)", n_win, "packaged RB windows"),
            (r"\(12 runs, ([\d,]+) rows\)", n_win, "packaged RB rows"),

        ):
            for c in re.findall(pattern, text):
                if int(c) != actual:
                    bad.append(f"{surface}: claims {c} {what}; the tree holds {actual}")

    # The Gate-2 horizon, wherever a surface states it. Scoped by context: the
    # Rayleigh-Benard protocol also says "30-step rollout" and is a different number.
    g2 = _g2_steps()
    for surface, text in (("README.md", readme), ("MANIFEST.md", man),
                          ("verify.py", (ROOT / "verify.py").read_text(encoding="utf-8"))):
        for m in re.finditer(r"(\d+)-step[\s\n#]*(?:autoregressive )?rollout", text):
            ctx = text[max(0, m.start() - 260):m.end() + 260]
            if "rollout_model" not in ctx and "Gate 2" not in ctx and "gate2" not in ctx:
                continue
            if int(m.group(1)) != g2:
                bad.append(f"{surface}: states a {m.group(1)}-step Gate-2 rollout; "
                           f"the fixture records {g2}")

    # The headline check count, derived from verify.py's own constants the same way
    # extract_numbers.py derives it -- so the prose is bound to the code, not merely
    # to the other prose surface. Changing 142 to 999 on both surfaces used to pass.
    n_claimed: set[int] = set()
    for text in (readme, man):
        for pat in (r"(\d{2,4}) (?:enumerated )?value checks",
                    r"the (\d{2,4}) Tier-1 checks",
                    r"asserts\*? (\d{2,4}) value checks",
                    r"each of the (\d{2,4}) enumerated",
                    r"among the (\d{2,4}) checks"):
            n_claimed |= {int(m) for m in re.findall(pat, text)}
    if n_claimed:
        derived = _derive_check_total()
        if derived is None:
            bad.append("verify.py: cannot import its check constants to bind the prose count")
        else:
            for c in sorted(n_claimed):
                if c != derived:
                    bad.append(f"prose claims {c} value checks; verify.py's own constants "
                               f"produce {derived}")
    return bad


def _conditional_stages() -> set[str]:
    """Stage tags `verify.py` prints only when a path outside this package exists.

    A standalone clone never sees these, so requiring them in the sample transcript
    would make the transcript wrong for the reader who actually has the package.
    Instead they must be *documented* as conditional in prose.
    """
    vp = (ROOT / "verify.py").read_text(encoding="utf-8")
    out: set[str] = set()
    for m in re.finditer(r"if\s+([A-Za-z_]\w*)\.exists\(\):", vp):
        head = vp[max(0, m.start() - 500):m.start()]
        if "HERE.parent" not in head:
            continue
        tail = vp[m.end():m.end() + 1500]
        tags = re.findall(r'print\(f?"\[([a-z][a-z0-9]*)\]', tail)
        if tags:
            out.add(tags[0])
    return out


def check_own_counts() -> list[str]:
    """The guard's statements about its OWN size must equal its own tables.

    It shipped "Four families" over ten, "the five below" over eight, and "eleven
    real past discrepancies" on two prose surfaces over thirteen entries -- four
    false self-descriptions inside the artifact whose sole purpose is catching false
    self-descriptions, on surfaces it sweeps, all passing.
    """
    _WORDS = {"three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8,
              "nine": 9, "ten": 10, "eleven": 11, "twelve": 12, "thirteen": 13,
              "fourteen": 14, "fifteen": 15, "sixteen": 16}
    src = (ROOT / "scripts" / "check_self_claims.py").read_text(encoding="utf-8")
    doc = src.split('"""')[1] if '"""' in src else ""
    n_fam = len(re.findall(r"^  [a-z]+ *-- ", doc, re.M))
    n_mut = len(_MUTATIONS)
    bad = []

    m = re.search(r"^([A-Za-z]+) families are checkable", doc, re.M)
    if not m:
        bad.append("check_self_claims.py: docstring no longer states its family count")
    elif _WORDS.get(m.group(1).lower()) != n_fam:
        bad.append(f"check_self_claims.py: docstring says {m.group(1)!r} families; it lists {n_fam}")

    for rel in ("README.md", "MANIFEST.md"):
        text = (ROOT / rel).read_text(encoding="utf-8")
        for w in re.findall(r"--validate-guard`? reintroduces ([a-z]+) real", text):
            if _WORDS.get(w) != n_mut:
                bad.append(f"{rel}: says --validate-guard reintroduces {w!r} discrepancies; "
                           f"the table holds {n_mut}")
    # and the manuscript, one level up, when present
    tex = ROOT.parent / "paper" / "tex" / "sec7_boundaries.tex"
    if tex.exists():
        t = tex.read_text(encoding="utf-8")
        for w in re.findall(r"holds ([a-z]+) rule families", t):
            if _WORDS.get(w) != n_fam:
                bad.append(f"sec7_boundaries.tex: says {w!r} rule families; the guard has {n_fam}")
        for w in re.findall(r"reintroduces ([a-z]+) real past discrepancies", t):
            if _WORDS.get(w) != n_mut:
                bad.append(f"sec7_boundaries.tex: says {w!r} reintroduced discrepancies; "
                           f"the table holds {n_mut}")
    return bad


def check_stage_tags() -> list[str]:
    """The README transcript must list every stage tag `verify.py` can print.

    A stage that gates the exit code but appears in no transcript reads, to a
    reviewer deciding what this package actually checks, as a stage that is not
    there. This was already wrong once: eleven stage lines had been added since
    the transcript was last written.
    """
    vp = (ROOT / "verify.py").read_text(encoding="utf-8")
    printed = set(re.findall(r'print\(f?"\[([a-z][a-z0-9]*)\]', vp))
    shown = set(re.findall(r"\[([a-z][a-z0-9]*)\] \.\.\.", (ROOT / "README.md").read_text(encoding="utf-8")))
    bad = []
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    cond = _conditional_stages()
    for tag in sorted(printed - shown):
        if tag in cond:
            # Allowed out of the transcript, but only if the prose says it is conditional.
            if not re.search(rf"`?\[{tag}\]`?[^.]{{0,400}}?\bonly\b", readme, re.S):
                bad.append(f"README.md: the [{tag}] stage prints only when a path outside "
                           f"this package exists, and the README does not say so")
            continue
        bad.append(f"README.md: verify.py prints a [{tag}] stage the transcript omits")
    for tag in sorted(shown - printed):
        bad.append(f"README.md: transcript shows a [{tag}] stage verify.py never prints")
    return bad


def check_cli_examples() -> list[str]:
    """Every CLI invocation the package advertises must be runnable as written.

    Two defects of this kind shipped in the released instrument: a documented
    `--npy` flag that does not exist (the path is positional and `.npy` is
    detected by suffix), and a headline example that declares none of the three
    layouts the instrument itself calls mandatory, so on the layout its own
    `--spatial-dims 3` implies it exits non-zero with `ambiguous layout`.

    Checked mechanically (the worked example under the instrument's examples
    directory is swept the same way):
      - every long flag in an advertised `normscreen` line exists in its parser
      - no advertised line both passes `--spatial-dims` and omits all three
        layout declarations, which the instrument documents as refused
    """
    import argparse

    bad: list[str] = []
    sys.path.insert(0, str(ROOT / "instrument"))
    try:
        from normscreen.__main__ import build_parser  # type: ignore
    except Exception:
        try:
            from normscreen import __main__ as _m  # type: ignore
            build_parser = getattr(_m, "build_parser", None)
        except Exception:
            build_parser = None
    known: set[str] = set()
    if build_parser is not None:
        try:
            p = build_parser()
            for a in p._actions:
                known.update(o for o in a.option_strings if o.startswith("--"))
        except Exception:
            known = set()
    if not known:
        # Fall back to reading the option strings out of the source, so a parser
        # that cannot be imported does not silently disable the check.
        src = (ROOT / "instrument" / "normscreen" / "__main__.py").read_text(encoding="utf-8")
        known = set(re.findall(r'add_argument\(\s*"(--[a-z0-9-]+)"', src))
    if not known:
        return ["check_cli_examples: could not determine normscreen's flags; check disabled"]

    LAYOUT_DECLS = ("--leading-axes", "--no-split-components", "--component-axis")
    surfaces = [("README.md", ROOT / "README.md"),
                ("instrument/README.md", ROOT / "instrument" / "README.md"),
                ("instrument/normscreen/__main__.py", ROOT / "instrument" / "normscreen" / "__main__.py"),
                ("instrument/normscreen/screen.py", ROOT / "instrument" / "normscreen" / "screen.py")]
    for label, path in surfaces:
        if not path.exists():
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            st = line.strip().lstrip("#").strip()
            if "normscreen" not in st or st.startswith(("*", "|", "-")):
                continue
            if not (st.startswith("normscreen") or st.startswith("python3 -m normscreen")
                    or st.startswith("python -m normscreen")):
                continue
            flags = re.findall(r"(--[a-z0-9-]+)", st)
            for f in flags:
                if f not in known:
                    bad.append(f"{label}: advertises `{f}` in `{st}`, which the parser does not accept")
            if "--spatial-dims" in flags and not any(d in flags for d in LAYOUT_DECLS):
                if "--demo" not in flags:
                    bad.append(f"{label}: advertises `{st}`, which declares no layout and so "
                               f"cannot produce a clean screen on the rank the flag implies")
    return bad


def check_saturation_claim() -> list[str]:
    """No shipped surface may say a floored score is capped.

    `sqrt(MSE/(Var+eps))` grows without limit in the model's error; what a floor
    bounds is the amplification of a FIXED error. The claim survived a round in the
    worked example after being corrected in the README and the test, because nothing
    checked for it -- and there it asserted the negation of the package's own test.
    """
    bad: list[str] = []
    pat = re.compile(r"(?:score|metric)[^.\n]{0,60}\bcaps?\b(?![a-z])|caps at\s*`?1/sqrt", re.I)
    for path in sorted(list(ROOT.rglob("*.py")) + list(ROOT.rglob("*.md"))):
        if "__pycache__" in str(path) or path.name == "check_self_claims.py":
            continue
        for i, line in enumerate(path.read_text(encoding="utf-8", errors="replace").splitlines(), 1):
            if pat.search(line) and "amplification" not in line.lower():
                bad.append(f"{path.relative_to(ROOT)}:{i}: says a floored score is capped; "
                           f"the score is unbounded in the error, the amplification is not")
    return bad


def check_numeric_examples() -> list[str]:
    """Numeric examples in prose and docstrings must evaluate to what they claim.

    `floor_share(var=1e-11, eps=1e-7) # 0.99989` shipped in the released
    instrument's README; the value is 0.99990.
    """
    bad: list[str] = []
    sys.path.insert(0, str(ROOT / "instrument"))
    try:
        from normscreen import floor_share  # type: ignore
    except Exception as exc:
        return [f"check_numeric_examples: cannot import normscreen ({exc}); check disabled"]
    pat = re.compile(r"floor_share\(\s*var\s*=\s*([0-9.e+-]+)\s*,\s*eps\s*=\s*([0-9.e+-]+)\s*\)"
                     r"\s*#\s*([0-9.]+)")
    for rel in ("instrument/README.md", "README.md", "instrument/normscreen/screen.py"):
        p = ROOT / rel
        if not p.exists():
            continue
        for var, eps, claimed in pat.findall(p.read_text(encoding="utf-8")):
            actual = float(floor_share(var=float(var), eps=float(eps)))
            if abs(actual - float(claimed)) > 0.5 * 10 ** -(len(claimed.split(".")[-1])):
                bad.append(f"{rel}: claims floor_share(var={var}, eps={eps}) # {claimed}; "
                           f"it is {actual:.{len(claimed.split('.')[-1])}f}")
    return bad


def check_transcript_order() -> list[str]:
    """The README transcript must match the order stages actually print, and must
    not present a conditional stage as unconditional.

    `[reference]` prints fifth, not first, and only when a path *outside* this
    standalone package exists -- so a reviewer running the released harness
    alone never sees the line the transcript leads with. A set comparison passed
    this; an order comparison does not.
    """
    vp = (ROOT / "verify.py").read_text(encoding="utf-8")
    printed = re.findall(r'print\(f?"\[([a-z][a-z0-9]*)\]', vp)
    shown = re.findall(r"\[([a-z][a-z0-9]*)\] \.\.\.", (ROOT / "README.md").read_text(encoding="utf-8"))
    bad = []
    # Order, comparing first appearances so the "x3" shorthand stays legal.
    def firsts(seq):
        out = []
        for t in seq:
            if t not in out:
                out.append(t)
        return out
    cond = _conditional_stages()
    printed = [t for t in printed if t not in cond]
    if firsts(printed) != firsts(shown):
        bad.append(f"README.md: transcript order {firsts(shown)} is not the print order {firsts(printed)}")
    return bad


def check_io_contract() -> list[str]:
    """The row-file writer and reader must agree, and must be single-sourced.

    They were written twice and drifted: the writer named its output `.json.gz`
    and wrote it uncompressed, so the documented regeneration path produced
    files the aggregator raised `BadGzipFile` on.
    """
    import tempfile

    bad: list[str] = []
    sys.path.insert(0, str(ROOT / "scripts"))
    try:
        from io_contract import read_rows, write_rows  # type: ignore
    except Exception as exc:
        return [f"scripts/io_contract.py: cannot import ({exc})"]
    payload = {"rows": [{"a": 1.5}], "fields": ["a"], "provenance": {"x": "y"}}
    with tempfile.TemporaryDirectory() as td:
        path = str(pathlib.Path(td) / "sub" / "dir" / "probe.json.gz")
        write_rows(path, payload)
        if read_rows(path) != payload:
            bad.append("scripts/io_contract.py: write_rows/read_rows do not round-trip")
        with gzip.open(path, "rt", encoding="utf-8") as fh:
            json.load(fh)  # raises BadGzipFile if the writer stopped compressing
    for rel in ("scripts/rb_model_eval.py", "scripts/rb_checkpoint_audit.py",
                "scripts/rt_model_eval.py"):
        src = (ROOT / rel).read_text(encoding="utf-8")
        if "io_contract" not in src:
            bad.append(f"{rel}: does not use scripts/io_contract.py, so the row-file "
                       f"contract is stated in more than one place again")
    return bad


# The claims RT-QUANT.md's banner withdraws, as fragments that actually occur in the
# body. Deriving these from the banner text alone does not work: the banner states the
# provenance claim in contracted form while the body interpolates a repository glob into
# the middle of it, so a literal-substring match silently never fires -- which is how
# four unmarked withdrawals survived a version of this check that tried to be clever.
# Each entry is (fragment-in-body, fragment-the-banner-must-still-declare).
_WITHDRAWN = [
    ("are the paper-run models", "paper-run models"),
    ("10/10 trajectories", "wherever they appear"),
    ("class reproduced", "reproduces / is reproduced"),
    ("predicting-the-mean", "predicting the mean"),
    ("deepest near-zero-variance", "deepest near-zero-variance"),
    ("entirely denominator", "attribution of the *published*"),
]
_MARKER = "WITHDRAWN"


def check_withdrawal_markers() -> list[str]:
    """A ledger claiming its withdrawals are marked inline must have them marked.

    `RT-QUANT.md` is the working ledger the appendix cites as the evidentiary record,
    and its banner asserts each withdrawn claim is marked "wherever it appears". Four
    occurrences were not, including the provenance identity the manuscript exists to
    deny. Checked in both directions: every fragment must still be declared by the
    banner, and every occurrence in the body must carry the marker.

    **Limit, stated because a reviewer defeated the rule with it.** This matches literal
    fragments. A withdrawn claim restated in different words passes. The rule keeps a
    ledger honest against edits and regressions; it cannot certify that no paraphrase of
    a withdrawn claim survives, and nothing here should be read as claiming it does.
    """
    p = ROOT.parent / "RT-QUANT.md"
    if not p.exists():
        return []          # not shipped inside the harness; swept from the submission tree
    text = p.read_text(encoding="utf-8")
    head, sep, body = text.partition("\n## ")
    if not sep:
        return ["RT-QUANT.md: no section heading found; cannot separate banner from body"]
    bad = []
    for frag, declared in _WITHDRAWN:
        if declared.lower() not in head.lower():
            bad.append(f"RT-QUANT.md: the banner no longer declares {declared!r} withdrawn, "
                       f"but this checker still holds {frag!r} as a withdrawn claim")
        for i, line in enumerate(body.splitlines(), start=head.count("\n") + 2):
            if frag.lower() in line.lower() and _MARKER not in line:
                bad.append(f"RT-QUANT.md:{i}: banner declares {frag!r} withdrawn wherever it "
                           f"appears; this occurrence carries no marker")
    return bad


def check_producer_targets() -> list[str]:
    """No shipped script may default to writing outside the package.

    Three did, all into a `.gate-work` working directory that is not shipped and in
    two cases was never created: the RB census died with `FileNotFoundError` after a
    full network run, and the benchmark-wide census wrote where the assembler never
    looked, so the MANIFEST's "re-runnable from the two scripts" did not compose.
    A default output path outside the package is not reproducible by a reader.
    """
    bad: list[str] = []
    for path in sorted(ROOT.glob("scripts/*.py")):
        if path.name == "check_self_claims.py":
            continue  # its mutation table names the defect it checks for
        src = path.read_text(encoding="utf-8")
        for m in re.finditer(r'["\']((?:/|~|\.\./|\.gate-work)[^"\']*)["\']', src):
            cand = m.group(1)
            line = src[:m.start()].count("\n") + 1
            ctx = src.splitlines()[line - 1]
            if ctx.lstrip().startswith("#"):
                continue          # naming the old path in a comment is not a default
            if not re.search(r"\b(out|output|dest|path|dir|_OUT)\w*\s*=|environ\.get|makedirs|open\(", ctx):
                continue          # not a write target
            if cand.startswith("../") and (path.parent.parent / cand[3:]).exists():
                continue          # inside the submission tree, e.g. ../hf-cache for downloads
            bad.append(f"scripts/{path.name}:{line}: defaults to writing {cand!r}, "
                       f"which is outside the package")
    return bad


# Fixtures whose producer the MANIFEST names. Each entry is checked in BOTH
# directions: the MANIFEST must attribute the fixture to that script, and the
# script's own default output must be that fixture. Re-attributing a fixture to a
# different *shipped* script used to pass, because the named file existed.
_PRODUCERS = {
    "fixtures/generalization/rayleigh_benard_census.json": ("rb_census.py", "_OUT"),
    "fixtures/generalization/benchmark_map": ("well_denominator_census.py", "_OUT_DIR"),
}


def check_producer_relations() -> list[str]:
    """A producer relation the MANIFEST asserts must be one the script performs."""
    import importlib.util
    import types

    bad: list[str] = []
    man = (ROOT / "MANIFEST.md").read_text(encoding="utf-8")
    for fixture, (script, const) in sorted(_PRODUCERS.items()):
        if f"scripts/{script}" not in man:
            bad.append(f"MANIFEST.md: no longer names scripts/{script} as a producer")
            continue
        # the MANIFEST must not attribute this fixture to some *other* shipped script
        for m in re.finditer(re.escape(pathlib.Path(fixture).name) + r"[^|\n]{0,160}?`scripts/([a-z0-9_]+\.py)`", man):
            if m.group(1) != script:
                bad.append(f"MANIFEST.md: attributes {fixture} to scripts/{m.group(1)}; "
                           f"the script that writes it is scripts/{script}")
        spec = importlib.util.spec_from_file_location(f"_p3_prod_{script}", ROOT / "scripts" / script)
        if spec is None or spec.loader is None:
            continue
        mod = importlib.util.module_from_spec(spec)
        for dep in ("fsspec", "h5py", "torch", "yaml", "requests"):
            sys.modules.setdefault(dep, types.ModuleType(dep))
        try:
            spec.loader.exec_module(mod)
        except Exception as exc:
            bad.append(f"scripts/{script}: cannot import to check its output path ({exc})")
            continue
        actual = getattr(mod, const, None)
        if actual is None:
            bad.append(f"scripts/{script}: no {const}; cannot bind its output to the MANIFEST")
        elif pathlib.Path(str(actual)).resolve() != (ROOT / fixture).resolve():
            bad.append(f"scripts/{script}: writes {actual!r}, but the MANIFEST attributes "
                       f"{fixture} to it")
    return bad


def check_quoted_sizes() -> list[str]:
    """Byte totals quoted in the prose must be the tree's actual byte totals."""
    bad: list[str] = []
    total = sum(f.stat().st_size for f in (ROOT / "fixtures").rglob("*")
                if f.is_file() and f.name != "summary.json")
    for rel in ("README.md", "MANIFEST.md"):
        text = (ROOT / rel).read_text(encoding="utf-8")
        for m in re.findall(r"([\d,]{6,}) bytes in total", text):
            if int(m.replace(",", "")) != total:
                bad.append(f"{rel}: quotes {m} bytes for the fixtures tree; it is {total:,}")
    return bad


def check_command_placeholders() -> list[str]:
    """A documented command must be runnable as written, not a sketch.

    A revision sha abbreviated to `7d351350eaf6...` inside a reproduction command is
    not a command; it is a note that someone still has to look the value up. The full
    values are in the packaged fixtures' provenance blocks, so there is no reason for
    a placeholder to survive.
    """
    bad: list[str] = []
    for rel in ("MANIFEST.md", "README.md"):
        text = (ROOT / rel).read_text(encoding="utf-8")
        for block in re.findall(r"```(?:bash|sh)?\n(.*?)```", text, re.S):
            for i, line in enumerate(block.splitlines(), start=1):
                if re.search(r"[0-9a-f]{6,}\.\.\.", line) or "<YOUR" in line or "TODO" in line:
                    bad.append(f"{rel}: a documented command carries a placeholder: {line.strip()[:78]!r}")
    return bad


def check_deps() -> list[str]:
    """The declared floor must admit the APIs the scripts actually call."""
    bad: list[str] = []
    req = (ROOT / "requirements.txt").read_text(encoding="utf-8")
    m = re.search(r"numpy\s*>=\s*(\d+)", req)
    if not m:
        return ["requirements.txt: no numpy floor declared"]
    floor = int(m.group(1))
    # APIs added in NumPy 2.0 that a 1.x install would not have.
    np2_only = {"np.trapezoid": 2}
    for p in sorted(ROOT.glob("scripts/*.py")) + [ROOT / "verify.py"]:
        if p.name == "check_self_claims.py":
            continue  # this file names the API in its own rule table

        src = p.read_text(encoding="utf-8")
        for api, need in np2_only.items():
            if api in src and floor < need:
                bad.append(
                    f"{p.relative_to(ROOT)}: calls {api} (NumPy {need}.0+) "
                    f"but requirements.txt declares numpy>={floor}"
                )
    return bad


def run() -> tuple[int, list[str]]:
    findings = (check_paths() + check_counts() + check_own_counts() + check_stage_tags()
                + check_transcript_order() + check_cli_examples()
                + check_numeric_examples() + check_saturation_claim()
                + check_io_contract()
                + check_withdrawal_markers() + check_producer_targets()
                + check_producer_relations() + check_quoted_sizes()
                + check_command_placeholders()
                + check_deps())
    n = len(PROSE) + len(CODE)
    return n, findings


# Each entry is one discrepancy this guard was written after finding by hand.
# `--validate-guard` copies the package, reintroduces the defect, and requires a
# non-zero exit -- a guard that cannot fail is not evidence of anything.
_MUTATIONS = [
    ("a renamed script whose docstrings keep the old name",
     "scripts/gate3_recheck_rt.py", "gate3_recheck_rb.py", "validate_rb_evaluator.py"),
    ("a dependency floor below the APIs the scripts call",
     "requirements.txt", "numpy>=2.0", "numpy>=1.26"),
    ("a packaged-window count off by one",
     "instrument/README.md", "1728 per-window", "1727 per-window"),
    ("a census row count off by one",
     "instrument/README.md", "17 rows reproduced", "16 rows reproduced"),
    ("a producer string naming a script that was never shipped",
     "MANIFEST.md", "scripts/gate3_assemble.py", "scripts/gate3_build_fixture.py"),
    # Added after two independent reviewers defeated the first version of this guard
    # with mutations it was not written to catch; every entry below the first five is one
# that a reviewer used to defeat an earlier version of this guard.
    ("the enumerated check total changed on every prose surface at once",
     "README.md", "142 value checks", "143 value checks"),
    ("a real filename claimed under a directory it does not live in",
     "MANIFEST.md", "`scripts/aggregate_results.py`", "`fixtures/aggregate_results.py`"),
    ("an advertised flag the parser does not accept",
     "instrument/README.md", "--no-split-components", "--whole-fields"),
    ("a worked numeric example that no longer evaluates to its claim",
     "instrument/README.md", "# 0.99990", "# 0.99889"),
    ("the row-file writer bypassing the single-sourced contract",
     "scripts/rb_model_eval.py", "from io_contract import write_rows  # noqa: E402", "import json as _unused"),
    ("a stage comment contradicting the fixture it introduces",
     "verify.py", "over a 31-step", "over a 5-step"),
    ("a documented command left with an abbreviated identifier",
     "MANIFEST.md", "7d351350eaf68caba1eac1e545cd6661251dd1fd", "7d351350eaf6..."),
    ("an unmarked restatement of a withdrawn claim in the packaged ledger",
     "../RT-QUANT.md", "are the paper-run models \u27e6WITHDRAWN", "are the paper-run models \u2014 note"),
    ("a shipped surface claiming a floored score is capped",
     "instrument/examples/pdebench_shocktube.py", "bounds the AMPLIFICATION of a fixed error at",
     "caps at"),
    ("a shipped script defaulting to a write outside the package",
     "scripts/rb_census.py", 'os.environ.get("RB_CENSUS_OUT", os.path.join(',
     'os.environ.get("RB_CENSUS_OUT", ".gate-work/out.json") or os.path.join('),
]


def validate_guard_fires() -> int:
    import shutil
    import subprocess
    import tempfile

    missed = []
    for label, rel, old, new in _MUTATIONS:
        with tempfile.TemporaryDirectory() as td:
            # Copy the SUBMISSION TREE, not just the package. The markers rule reads
            # RT-QUANT.md from one level up, so copying ROOT alone left that rule
            # silently disabled in every mutation run while the validator still
            # reported a full pass -- the one family with a failure history was the
            # one family --validate-guard could not fail on.
            root = pathlib.Path(td) / "lane"
            ignore = shutil.ignore_patterns("__pycache__", "*.pyc", ".git", "hf-cache",
                                            ".gate-work", "results", "tmp", "*.pdf")
            shutil.copytree(ROOT.parent, root, ignore=ignore)
            d = root / ROOT.name
            f = (root / rel[len("../"):]) if rel.startswith("../") else (d / rel)
            src = f.read_text(encoding="utf-8")
            if old not in src:
                missed.append(f"{label}: anchor {old!r} no longer in {rel}")
                continue
            f.write_text(src.replace(old, new, 1), encoding="utf-8")
            r = subprocess.run([sys.executable, "scripts/check_self_claims.py"],
                               cwd=d, capture_output=True, text=True)
            if r.returncode == 0:
                missed.append(f"{label}: reintroduced in {rel}, guard still passed")
            else:
                print(f"  caught: {label}")
    if missed:
        print("FAIL: the guard does not catch what it was written for:")
        for m in missed:
            print(f"  - {m}")
        return 1
    print(f"[selfclaims] guard validated: {len(_MUTATIONS)}/{len(_MUTATIONS)} "
          f"reintroduced discrepancies caught")
    return 0


def main() -> int:
    if "--validate-guard" in sys.argv:
        return validate_guard_fires()
    n, findings = run()
    if findings:
        print(f"FAIL: {len(findings)} self-description(s) do not match the package:")
        for f in findings:
            print(f"  - {f}")
        return 1
    print(f"[selfclaims] OK over {n} surfaces: ten rule families found nothing. Each binds "
          f"a NAMED set, not a universal -- the counts checked are the ones enumerated in "
          f"the module docstring, the producer rule the relations listed there, the marker "
          f"rule literal fragments (a paraphrase of a withdrawn claim passes). It is also "
          f"one-directional: it checks that what the package says is true, never that "
          f"everything the package ships is said.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
