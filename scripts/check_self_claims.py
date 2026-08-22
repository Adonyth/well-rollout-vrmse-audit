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
Reviewers repeatedly found discrepancies of this class by hand, both before this
check existed and after its first version, and defeated several of its rules with
mutations it was not written to catch. Read a pass as "the rules below found
nothing", not as "the package is honest about itself".

Run standalone (`python3 scripts/check_self_claims.py`) or via `verify.py`.
"""

from __future__ import annotations

import gzip
import json
import pathlib
import re
import shlex
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
_SUFFIXES = (".py", ".json", ".json.gz", ".md", ".txt", ".sty", ".tex", ".toml", ".cfg",
             ".h5", ".hdf5", ".yaml", ".yml", ".npy", ".npz", ".csv", ".sh", ".lock")
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
    # A bare suffix is the tail of a glob the tokenizer split on `*`. A stem with
    # several hyphens is usually a hyphenated English phrase rather than a filename --
    # but only in unbackticked prose. Inside backticks it is a claim, and discarding it
    # let `fixtures/a-b-c-d.json` pass.
    return bool(stem)


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
    # the reader's own inputs in usage examples
    "data.h5", "field.npy", "x.h5", "split.h5",
    # fetched from the public mirror at run time, never shipped
    "stats.yaml",
}
# .tex chapters live in the paper repo; this package ships none.
# Raw field tensors are never package contents: the package ships derived statistics
# only, and says so. A .hdf5/.h5 path in the prose names a remote object on the public
# mirror or a file the reader supplies, so it is not a claim about this tree.
_EXTERNAL_SUFFIX = (".tex", ".hdf5", ".h5")


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
            toks = [t for t in re.findall(r"[A-Za-z0-9_./{}-]+", text)
                    if t.rsplit("/", 1)[-1].count("-") <= 2]
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
    # and the families must account for every rule that actually runs
    listed = {m.lower() for m in re.findall(r"^  ([a-z]+) *-- ", doc, re.M)}
    executed = {fam for fam, _ in _rules()}
    for fam in sorted(listed - executed):
        bad.append(f"check_self_claims.py: the docstring describes a {fam!r} family that "
                   f"no executing rule provides")
    for fam in sorted(executed - listed):
        bad.append(f"check_self_claims.py: a {fam!r} rule executes that the docstring "
                   f"does not describe")

    for rel in ("README.md", "MANIFEST.md"):
        text = (ROOT / rel).read_text(encoding="utf-8")
        for w in re.findall(r"--validate-guard`? reintroduces ([a-z]+) real", text):
            if _WORDS.get(w) != n_mut:
                bad.append(f"{rel}: says --validate-guard reintroduces {w!r} discrepancies; "
                           f"the table holds {n_mut}")
    return bad


def check_own_counts_manuscript() -> list[str]:
    """The manuscript's statement of the guard's size must match the guard.

    Split out of `check_own_counts` because it reads a file outside the package:
    as one rule the whole family went silent in a standalone clone while still
    being counted clean, so a mutation of the manuscript's own family count could
    not be caught there and no skip was reported.
    """
    _WORDS = {"three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8,
              "nine": 9, "ten": 10, "eleven": 11, "twelve": 12, "thirteen": 13,
              "fourteen": 14, "fifteen": 15, "sixteen": 16, "seventeen": 17,
              "eighteen": 18, "nineteen": 19, "twenty": 20, "twenty-one": 21,
              "twenty-two": 22, "twenty-three": 23, "twenty-four": 24}
    src = (ROOT / "scripts" / "check_self_claims.py").read_text(encoding="utf-8")
    doc = src.split('"""')[1] if '"""' in src else ""
    n_fam = len(re.findall(r"^  ([a-z]+) *-- ", doc, re.M))
    n_mut = len(_MUTATIONS)
    bad: list[str] = []
    # Scan EVERY manuscript source, not one named file. This rule read only
    # sec7_boundaries.tex; the claim it polices had since moved to appendix.tex, so
    # the rule could not fire on the live manuscript and the "ten rule families"
    # sentence was unguarded while the rule was counted clean. Auditing which rules
    # --validate-guard can actually reach is what surfaced it: no mutation could
    # reach this one, because there was nothing in its file to mutate.
    for tex in sorted((ROOT.parent / "paper" / "tex").glob("*.tex")):
        t = tex.read_text(encoding="utf-8")
        for w in re.findall(r"holds ([a-z]+) rule families", t):
            if _WORDS.get(w) != n_fam:
                bad.append(f"{tex.name}: says {w!r} rule families; the guard has {n_fam}")
        for w in re.findall(r"reintroduces ([a-z]+) real past discrepancies", t):
            if _WORDS.get(w) != n_mut:
                bad.append(f"{tex.name}: says {w!r} reintroduced discrepancies; "
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


class _ParseRejected(Exception):
    """Raised in place of argparse's SystemExit so a rejection is a finding, not an exit."""


def _raise_parse(*a, **k):
    raise _ParseRejected(" ".join(str(x) for x in a if isinstance(x, str)) or "rejected")


def check_cli_examples() -> list[str]:
    """Every advertised CLI invocation must PARSE against the tool's real interface.

    Scope, stated exactly, because an earlier docstring said "runnable as written" and
    this rule cannot establish that: it inspects flags and layout declarations against
    the argument parser. It does NOT install the package or execute the command, so it
    cannot see a runtime failure. A reviewer found one it therefore missed -- the
    advertised HDF5 invocations failed after the advertised `pip install -e .` because
    h5py is an optional extra -- and that class of defect is out of this rule's reach
    by construction, not by oversight.

    Two defects of this kind shipped in the released instrument: a documented
    `--npy` flag that does not exist (the path is positional and `.npy` is
    detected by suffix), and a headline example that declares none of the three
    layouts the instrument itself calls mandatory, so on the layout its own
    `--spatial-dims 3` implies it exits non-zero with `ambiguous layout`.

    Checked mechanically, by handing each advertised `normscreen` line to the
    instrument's REAL parser (`normscreen.__main__.build_parser`) and requiring it
    to parse. That is stronger than the flag-membership test this rule used to do,
    which accepted `--eps not-a-number` while the real CLI exited 2. It also checks
    that no advertised line passes `--spatial-dims` while omitting all three layout
    declarations, which the instrument documents as refused.

    NOT swept: `instrument/examples/`. An earlier version of this docstring said the
    worked example there was swept the same way; it never was, and the surfaces list
    below is the whole of what this rule reads.
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
            st = re.sub(r"^\$\s*", "", st)
            st = st.split("&&")[-1].strip() if "&&" in st else st
            if "normscreen" not in st.split("#")[0]:
                continue
            flags = re.findall(r"(--[a-z0-9-]+)", st)
            for f in flags:
                if f not in known:
                    bad.append(f"{label}: advertises `{f}` in `{st}`, which the parser does not accept")
            # Hand the whole invocation to the REAL parser. Flag membership alone let
            # `--eps not-a-number` through while the shipped CLI exited 2, so the rule's
            # own headline was wider than its code until this call existed.
            # Parse only lines that ARE invocations. Prose mentioning the tool by name is
            # not a command, and handing it to argparse manufactures findings rather than
            # finding them -- a first cut of this check reported six such "rejections",
            # every one of them an English sentence.
            _is_invocation = bool(re.match(
                r"^(?:python3?\s+-m\s+)?normscreen(?:\s|$)", st.split("#")[0].strip()))
            if build_parser is not None and _is_invocation:
                try:
                    argv = shlex.split(st.split("#")[0])
                    for _drop in ("python3", "python", "-m", "normscreen"):
                        while argv and argv[0] == _drop:
                            argv.pop(0)
                    _p = build_parser()
                    _p.exit = _p.error = _raise_parse  # type: ignore[assignment]
                    _p.parse_args(argv)
                except _ParseRejected as exc:
                    bad.append(f"{label}: advertises `{st}`, which the parser rejects: {exc}")
                except Exception:
                    pass          # shlex/other failures are not parser verdicts
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
    pat = re.compile(
        r"(?:score|metric)[^.\n]{0,60}\b(?:caps?|capped|bounded above|saturates?|"
        r"maximum|ceiling)\b|caps at\s*`?1/sqrt", re.I)
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
    targets = (sorted(ROOT.glob("scripts/*.py")) + [ROOT / "verify.py"]
               + sorted(ROOT.glob("instrument/**/*.py")))
    for path in targets:
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
            bad.append(f"{path.relative_to(ROOT)}:{line}: defaults to writing {cand!r}, "
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


def check_factor_claims() -> list[str]:
    """A "factor of N" in the manuscript must trace to the number table.

    `check_unsourced.py` exempts bare one- and two-digit integers by design, and the
    manuscript says so. That exemption is where a wrong headline hid for several
    rounds: the introduction claimed the predictions "move by a factor of 47 across
    floors a released pipeline actually uses" when 47 requires the definitional limit
    and the released-floor span is 8.542. A magnitude claim is load-bearing whatever
    its digit count, so these are swept even where the general sweep declines.

    **Limits, measured rather than asserted.** Three, and they are large:

    1. It cannot catch a magnitude attached to the wrong SCOPE. The 47 above is a real
       leaf -- the definitional-limit span -- quoted in a clause about released floors.
       Four independent reviewers found that one; this rule would not have.
    2. A claim passes if it rounds to ANY leaf at its own precision, and the table has
       hundreds of leaves. At one and two significant digits that is close to no
       constraint at all: a reviewer measured 9 of 9 one-digit and 68 of 90 two-digit
       candidate values passing by collision with unrelated leaves (a frame count, a
       file count). Those are exactly the digit counts `check_unsourced.py` exempts and
       that this rule was added to cover.
    3. It matches the phrase "factor of N" only. "an N-fold span", "a factor N", and
       "moves by N times" all bypass it.

    It is therefore a guard against a fabricated magnitude, not a guarantee that every
    magnitude in the manuscript is sourced and correctly scoped.
    """
    tex = ROOT.parent / "paper" / "tex"
    ref = ROOT / "fixtures" / "numbers_reference.json"
    if not tex.is_dir() or not ref.exists():
        return []
    keys = set()
    def walk(o):
        if isinstance(o, dict):
            for v in o.values():
                walk(v)
        elif isinstance(o, list):
            for v in o:
                walk(v)
        elif isinstance(o, (int, float)) and not isinstance(o, bool):
            keys.add(float(o))
    walk(json.loads(ref.read_text(encoding="utf-8")))
    bad = []
    for f in sorted(tex.glob("*.tex")):
        for i, line in enumerate(f.read_text(encoding="utf-8").splitlines(), 1):
            for m in re.finditer(r"factor of \$?([0-9][0-9.]*)\$?", line):
                claim = m.group(1).rstrip(".")
                val = float(claim)
                # A claim may legitimately round its leaf ("a factor of 66" for 65.88),
                # so match at the precision the claim itself states.
                digits = len(claim.replace(".", "").lstrip("0")) or 1
                if not any(float(f"{k:.{digits}g}") == val for k in keys):
                    bad.append(f"paper/tex/{f.name}:{i}: claims a factor of {claim}, which "
                               f"no leaf of the number table rounds to at that precision")
    return bad


def check_instrument_copies() -> list[str]:
    """The instrument ships twice; the two trees must be identical.

    The package sweeps its own copy under `instrument/`. The submission also ships a
    root-level copy, which the freeze covers and which readers of the paper reach
    first. A defect corrected in one and not the other survived a full round exactly
    this way, so rather than sweep both, the copies are pinned equal: fix one and the
    other must follow, or this fails.
    """
    other = ROOT.parent / "instrument"
    if not other.is_dir():
        return []
    bad = []
    ours = {q.relative_to(ROOT / "instrument"): q for q in (ROOT / "instrument").rglob("*")
            if q.is_file() and "__pycache__" not in str(q)}
    theirs = {q.relative_to(other): q for q in other.rglob("*")
              if q.is_file() and "__pycache__" not in str(q)}
    for rel in sorted(set(ours) | set(theirs)):
        if rel not in ours:
            bad.append(f"instrument/{rel}: shipped at the submission root but not in the package")
        elif rel not in theirs:
            bad.append(f"instrument/{rel}: shipped in the package but not at the submission root")
        elif ours[rel].read_bytes() != theirs[rel].read_bytes():
            bad.append(f"instrument/{rel}: the two shipped copies differ")
    return bad


def _published_constants(ref) -> set:
    """4-sig-fig strings of every value the number table marks as PUBLISHED."""
    out = set()
    def walk(o, p=""):
        if isinstance(o, dict):
            for k, v in o.items():
                walk(v, f"{p}.{k}")
        elif isinstance(o, (int, float)) and not isinstance(o, bool):
            if p.endswith(".published"):
                out.add(f"{float(o):.4g}")
    walk(ref)
    return out


def check_floor_declarations() -> list[str]:
    """A floor-dependent value in the manuscript must name its floor nearby.

    This is the defect class that led four consecutive review rounds: a score, a
    ratio of scores, or a statistic computed on scores, quoted without saying at
    which denominator floor -- after which a comparison built on it silently
    reverses at the other floor the paper also reports. Instances found by hand
    included a 47x span, the unit-convention ratios, an estimator factor of 2.461
    (1.105 at the other floor), and eight correlation statistics whose significance
    pattern is a library-floor artefact.

    The rule: for every numeric leaf whose own path in the number table names a
    floor, find that value in the manuscript and require a floor declaration
    (`\epslib`, `\epsfix`, "definitional", or an explicit 10^{-N}) within the same
    sentence. Values that are floor-invariant are exempt by construction, because
    only leaves the table itself files under a floor are swept.

    **Limit.** It cannot see a value the table does not carry at 4 significant
    figures, and it treats a sentence boundary as the scope of a declaration.
    """
    tex = ROOT.parent / "paper" / "tex"
    ref = ROOT / "fixtures" / "numbers_reference.json"
    if not tex.is_dir() or not ref.exists():
        return []
    # A path component that names a FLOOR -- not a Rayleigh number inside a dataset
    # filename, which is why this matches components rather than substrings.
    # Floor-naming path components, in ALL the vocabularies the number table uses.
    # An earlier version matched only the eps_lib/eps_fix spellings and so missed the
    # entire `models` block -- every cell of the three principal tables -- which files
    # its floors as `lib`, `eps5`, `density_only_lib`, `velocity_only_eps5`, and so on.
    floory = re.compile(
        r"^(eps_?(lib|fix|0)\w*|eps_1e9|eps5|lib|definitional\w*|library_floor\w*|"
        r"\w*_eps_(lib|fix)\w*|\w*_(lib|eps5)|max_(lib|fix))$", re.I)
    _published_vals = _published_constants(json.loads(ref.read_text(encoding="utf-8")))
    _ours_floored: set = set()   # filled by walk() below: values under floor-named paths

    def _seg_floors(seg: str) -> set:
        """The floor(s) a single path segment names, or an empty set."""
        q = seg.lower()
        out = set()
        if "eps_fix" in q or q == "eps5" or q.endswith("_eps5") or "max_fix" in q:
            out.add("fix")
        if ("eps_lib" in q or "library_floor" in q or q in ("lib", "library")
                or q.endswith(("_lib", "_library"))):
            out.add("lib")
        if "definitional" in q or "eps_0" in q:
            out.add("def")
        if "eps_1e9" in q:
            out.add("e9")
        return out

    def _floor_of(path: str) -> frozenset:
        """The floor(s) a leaf path names. A path naming two is a span.

        The LEAF segment is authoritative; ancestors are consulted only when the
        leaf names no floor. `zscored_at_library_floor.definitional_raw_6_12` is a
        definitional value stored inside a block named for the library floor, and
        letting the block name contribute made it look like a two-floor span and
        collide its own value out of the swept set.
        """
        segs = path.split(".")
        own = _seg_floors(segs[-1])
        if own:
            return frozenset(own)
        for seg in reversed(segs[:-1]):
            own = _seg_floors(seg)
            if own:
                return frozenset(own)
        return frozenset()

    wanted: dict[str, str] = {}

    def walk(o, path=""):
        if isinstance(o, dict):
            for k, v in o.items():
                walk(v, f"{path}.{k}" if path else k)
        elif isinstance(o, (int, float)) and not isinstance(o, bool):
            # A leaf named `published` is a PUBLISHED constant even when it sits inside
            # a block whose path names a floor -- storing a published value beside the
            # derived value it is compared against is the natural layout. It is neither
            # one of OUR floor-dependent values nor subject to the floor requirement,
            # because the paper states those historical floors are unrecoverable.
            if path.rsplit(".", 1)[-1] == "published":
                return
            if any(floory.match(seg) for seg in path.split(".")):
                key = f"{float(o):.4g}"
                _ours_floored.add(key)
                # A 4-s.f. collision is ambiguous only if the colliding leaves sit
                # at DIFFERENT floors. Comparing paths instead dropped every value
                # recorded in more than one block -- and the more central a number is,
                # the more blocks record it, so the guard skipped exactly the values
                # the class recurred on (16.46, 91.46, 1.927 among them).
                if key in wanted:
                    if wanted[key] is not None and _floor_of(wanted[key]) != _floor_of(path):
                        wanted[key] = None
                else:
                    wanted[key] = path

    walk(json.loads(ref.read_text(encoding="utf-8")))
    # Only actual floor IDENTIFIERS count. The bare word "floor" is not one: a
    # sentence saying "under the better-conditioned floor \epsfix{}" names one floor,
    # not two, and treating the noun as a second name made it look ambiguous.
    # Floor IDENTIFIERS, in macro form and in the three worded forms the paper uses.
    # The bare noun "floor" is deliberately not one: a sentence saying "under the
    # better-conditioned floor \epsfix{}" names one floor, not two, and counting the
    # noun made single-floor sentences look ambiguous.
    decl = re.compile(r"\\epslib|\\epsfix|10\^\{-\d+\}|10\$\^\{-\d+\}\$|"
                      r"\\num\{1e-\d+\}|definitional(?:\s+limit)?|"
                      r"10\^\{-9\}|\$10\^\{-9\}\$|"
                      r"library'?s?\s+(?:own\s+)?(?:default\s+)?floor|library\s+metric|"
                      r"better-conditioned\s+floor|documented\s+floor|"
                      r"its\s+own\s+default\s+floor|library\s+VRMSE|"
                      # joint declarations: naming BOTH floors at once
                      r"(?:across|under|between)\s+(?:the\s+)?(?:same\s+)?two\s+floors|"
                      r"under\s+both\s+floors|both\s+floors|the\s+floor\s+change|"
                      r"across\s+(?:the\s+)?(?:same\s+)?span", re.I)
    bad = []
    for f in sorted(tex.glob("*.tex")):
        text = f.read_text(encoding="utf-8")
        # A float environment is ONE scope: a floor named in the caption governs
        # every cell in the body, so the sentence splitter must not cut between them.
        # Chunk the file, keeping each chunk's ABSOLUTE offset so a prose window can
        # be taken over the surrounding text. Re-finding a chunk by its own prefix
        # could land on a different occurrence entirely.
        scopes, rest = [], []
        pos = 0
        for m in re.finditer(r"\\begin\{(table|figure)\*?\}.*?\\end\{\1\*?\}",
                             text, re.S):
            rest.append((pos, text[pos:m.start()]))
            scopes.append((m.start(), m.group(0)))
            pos = m.end()
        rest.append((pos, text[pos:]))
        # split the non-float prose into sentences, keeping enough context that a
        # declaration one clause away still counts
        floats = set(range(len(scopes)))
        chunks = list(scopes)
        for _base, _chunk in rest:
            _off = _base
            for _piece in re.split(r"((?<=[.:;])\s+)", _chunk):
                if _piece and not _piece.isspace():
                    chunks.append((_off, _piece))
                _off += len(_piece)
        for _i, (_abs0, sent) in enumerate(chunks):
            # Inside a float, a floor named in the caption or a column header governs
            # every cell; in running prose the declaration must sit beside the value.
            in_float = _i in floats
                        # Two value forms, because the first alone was blind to every score the
            # manuscript typesets in scientific notation: a reviewer found a
            # floor-dependent \num{...} value this rule could not see, and it had to be
            # caught by hand. Plain decimals AND \num{}/e-notation are scanned now.
            # A mantissa carrying an explicit power of ten is NOT the bare mantissa:
            # `1.097\times10^{-11}` is a variance, and matching it against a stored
            # scalar 1.097 by mantissa alone flagged a number that is not floor-dependent
            # at all. Skip the literal when a \times10^{..} follows it -- comparing
            # magnitudes would need the exponent, and no stored leaf is written that way.
            _sci = {m.start(1) for m in re.finditer(
                r"(\d+\.\d{1,4})\s*\\times\s*10\^", sent)}
            _vals = [(m.start(), m.group(1)) for m in
                     re.finditer(r"\$?(\d+\.\d{1,4})\$?", sent)
                     if m.start(1) not in _sci]
            _vals += [(m.start(), m.group(1)) for m in re.finditer(
                r"\\num\{([0-9.]+e[+-]?\d+)\}|\$?(\d+\.?\d*e[+-]?\d+)\$?", sent)
                if m.group(1)]
                        # A sentence that cites the benchmark's own table is quoting PUBLISHED values,
            # whose floor this paper states is unrecoverable -- requiring a floor beside them
            # would demand a declaration the paper explicitly says cannot be made. This
            # exemption became necessary when a newly added leaf collided at four
            # significant figures with a published figure quoted from that table.
            _cites_benchmark = "thewell2024" in sent
            for m in [type("M", (), {"start": (lambda s, p=p: p), "group": (lambda s, i, g=g: g)})()
                      for p, g in _vals]:
                                # Exempt ONLY a value the number table itself records under a
                # `.published` path, quoted in a sentence citing the benchmark. A blanket
                # "cites the benchmark" exemption would also excuse OUR derived ratios,
                # which routinely appear in the same sentence as the published value they
                # are a ratio to -- that would trade a miss for a pass, which is the
                # failure mode this project keeps having.
                                # Exempt a published constant quoted in a benchmark-citing sentence --
                # but NOT when the same 4-s.f. value is also one of OUR floor-named leaves.
                # A reviewer found exactly that collision: our
                # tfno_rt_rollout_derived.eps_lib_1e7.leave_one_out_max is 1.486, the same
                # to four figures as the published RB one-step cell, so a blanket exemption
                # would let our value be quoted floorless in any citing sentence.
                _v4 = f"{float(m.group(1)):.4g}"
                if _cites_benchmark and _v4 in _published_vals and _v4 not in _ours_floored:
                    continue
                v = f"{float(m.group(1)):.4g}"
                if v not in wanted or wanted[v] is None:
                    continue
                # A declaration must sit near THIS value, not merely somewhere in the
                # sentence: a sentence that quotes two floors needs two declarations,
                # and stripping one of them used to pass on the strength of the other.
                # Ambiguity only arises when one sentence carries MORE THAN ONE
                # floor: then a value must have a declaration beside it, because the
                # other one governs a different number. With a single floor named --
                # or inside a float, where the caption governs the body -- one
                # declaration anywhere in the scope is unambiguous.
                # Each value must have ITS OWN floor named nearby. The leaf path says
                # which floor the value belongs to, so a sentence that names one floor
                # does not license a value computed at the other -- the hole that let a
                # phase-mean pair at two different floors pass on one declaration.
                def _canon(tok: str) -> str:
                    t = tok.lower()
                    # "the documented floor" is this manuscript's own name for eps_fix
                    # -- eleven occurrences, none of them the library floor. Mapping it
                    # to "lib" made the rule manufacture a library declaration out of
                    # the paper's standard phrase for the OTHER floor.
                    if ("epsfix" in t or "better-conditioned" in t or "10^{-5}" in t
                            or "documented floor" in t):
                        return "fix"
                    if ("two floors" in t or "both floors" in t
                            or "floor change" in t or "same span" in t):
                        return "both"
                    if "epslib" in t or "library" in t or "10^{-7}" in t:
                        return "lib"
                    if "10^{-9}" in t:
                        return "e9"
                    if "definitional" in t:
                        return "def"
                    return t

                path = wanted[v].lower()
                # A leaf naming TWO floors is a SPAN between them, not a value at one:
                # `ratio_definitional_to_eps_fix` has no single "own" floor, so naming
                # either endpoint is a sufficient declaration.
                own = set(_floor_of(path))
                if not own:
                    continue
                # In prose the window is taken over the surrounding TEXT, not clipped
                # at the sentence boundary: a reader resolving "at \epslib{}" from the
                # preceding clause does not stop at the period, and clipping there
                # flagged a dozen occurrences whose floor was one sentence away.
                if in_float:
                    scope = sent
                else:
                    _abs = _abs0 + m.start()
                    scope = text[max(0, _abs - 230):_abs + len(m.group(0)) + 230]
                _found = {_canon(d.group(0)) for d in decl.finditer(scope)}
                # A joint declaration ("across the two floors") names lib AND fix, so it
                # satisfies a value at either -- but NOT a value at a third floor, and
                # not a lone value that belongs to only one of them. Treating it as a
                # blanket pass reopened the very hole this rule was written to close.
                if "both" in _found:
                    _found |= {"lib", "fix"}
                if own & _found:
                    continue
                # An explicit statement that the floor is UNRECOVERABLE is a floor
                # declaration -- the only honest one available for a published cell whose
                # scoring configuration this paper says cannot be recovered. Without this,
                # the rule demands a name the paper's own boundaries forbid it to supply,
                # and the only way to satisfy it would be to delete the disclosure.
                                # ...but ONLY for a value the table records as PUBLISHED. Unrecoverability
                # is a property of the historical record, never of a number we computed
                # ourselves: our own floors are always known. Without this restriction the
                # clause is an escape hatch -- writing "at a floor the record does not
                # state" beside one of OUR values silenced the rule, which I verified
                # before narrowing it.
                                # The narrowing guard (a) received applies here too -- a value that is
                # ALSO one of our floor-named leaves must not be exempted by a claim about
                # the historical record. A real 4-s.f. collision exists (our
                # tfno_rt_rollout_derived.eps_lib_1e7.leave_one_out_max equals the published
                # RB one-step cell at 1.486), so for such a value the sentence must ALSO
                # assert, in words, that the number being quoted is the published one. That
                # turns the hatch from a value coincidence into a falsifiable statement an
                # editor can check. A previous round claimed this narrowing and shipped
                # without it; the negative control used a value that could not reach this
                # branch, so it passed for the wrong reason.
                _declares_unrecoverable = re.search(
                    r"(floor|scoring space)[^.]{0,120}?"
                    r"(not recoverable|does not state|cannot recover|unrecoverable|"
                    r"not stated|unstated)", scope, re.I | re.S)
                _asserted_published = re.search(r"\bas published\b|\bpublished\b[^.]{0,40}?"
                                                + re.escape(m.group(1)), scope, re.I | re.S)
                if _v4 in _published_vals and _declares_unrecoverable and (
                        _v4 not in _ours_floored or _asserted_published):
                    continue
                bad.append(f"paper/tex/{f.name}: quotes {m.group(1)} "
                           f"({wanted[v]}) with no floor named beside it")
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
    """No documented command may contain an unresolved placeholder.

    Scope, stated exactly: this greps for placeholder PATTERNS -- abbreviated shas,
    `<YOUR...>`, `TODO`, `<UPPER-CASE>`. It does not execute anything, does not check
    a flag against a parser, and does not check that a named script exists, so a
    command naming a missing file passes it. The headline used to say such a command
    "must be runnable as written", which this rule cannot establish; a reviewer added a
    documented command naming a script that does not exist to a clone, and the rule
    passed while the command itself exited 2.

    A revision sha abbreviated to `7d351350eaf6...` inside a reproduction command is
    not a command; it is a note that someone still has to look the value up. The full
    values are in the packaged fixtures' provenance blocks, so there is no reason for
    a placeholder to survive.
    """
    bad: list[str] = []
    for rel in ("MANIFEST.md", "README.md"):
        text = (ROOT / rel).read_text(encoding="utf-8")
        blocks = re.findall(r"```(?:bash|sh)?\n(.*?)```", text, re.S)
        blocks += [l for l in text.splitlines() if l.startswith("    ") and "--" in l]
        blocks += re.findall(r"`([^`\n]*--[^`\n]*)`", text)
        for block in blocks:
            for i, line in enumerate(block.splitlines() or [block], start=1):
                if (re.search(r"[0-9a-f]{6,}\.\.\.", line) or "<YOUR" in line
                        or "TODO" in line or re.search(r"<[A-Z][A-Z-]{3,}>", line)):
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


# Subjects that live one level up, outside the package. In a standalone clone these
# are absent and the rule cannot run; `run` reports it as SKIPPED rather than as a pass.
check_factor_claims._needs_source_repo = lambda: (ROOT.parent / "paper" / "tex").is_dir()
check_instrument_copies._needs_source_repo = lambda: (ROOT.parent / "instrument").is_dir()
check_withdrawal_markers._needs_source_repo = lambda: (ROOT.parent / "RT-QUANT.md").exists()
check_own_counts_manuscript._needs_source_repo = lambda: (
    ROOT.parent / "paper" / "tex" / "sec7_boundaries.tex").exists()
check_floor_declarations._needs_source_repo = lambda: (
    ROOT.parent / "paper" / "tex").is_dir()


_LAST_SKIPPED: list[str] = []


def _rules() -> list:
    """The rule functions `run` executes. Single source for what runs AND what is
    reported: an earlier version derived the count from the module docstring's bullet
    list, so deleting a rule from `run` still printed "ten rule families found nothing"
    and passed on the defect the missing rule existed to catch."""
    return [
        ("paths", check_paths),
        ("counts", check_counts), ("counts", check_own_counts), ("counts", check_own_counts_manuscript),
        ("counts", check_quoted_sizes), ("counts", check_factor_claims),
        ("counts", check_floor_declarations),
        ("stages", check_stage_tags), ("stages", check_transcript_order),
        ("cli", check_cli_examples),
        ("commands", check_command_placeholders),
        ("numbers", check_numeric_examples), ("numbers", check_saturation_claim),
        ("io", check_io_contract), ("io", check_instrument_copies),
        ("markers", check_withdrawal_markers),
        ("producers", check_producer_targets), ("producers", check_producer_relations),
        ("deps", check_deps),
    ]


def run() -> tuple[int, list[str]]:
    findings: list[str] = []
    skipped: list[str] = []
    for _family, rule in _rules():
        out = rule()
        # A rule whose subject lives outside the package returns [] in a standalone
        # clone. Reporting that as "found nothing" is itself a false self-description:
        # three rules ran zero checks in the artifact a reviewer downloads while the
        # stage still printed a full pass.
        if out == [] and getattr(rule, "_needs_source_repo", None) is not None:
            if not rule._needs_source_repo():
                skipped.append(rule.__name__)
                continue
        findings += out
    _LAST_SKIPPED[:] = skipped
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
    # Below: the seven rules that had NO mutation able to reach them. The validator
    # reported "15/15 caught" truthfully while exercising 12 of 19 rules -- including
    # check_floor_declarations, whose own guards shipped broken twice. Each mutation
    # below was confirmed to fire the named rule and only then added.
    ("the guard's own family count misstated in its docstring",          # check_own_counts
     "scripts/check_self_claims.py", "Ten families are checkable", "Nine families are checkable"),
    ("the manuscript misstating how many rule families the guard holds",  # ..._manuscript
     "../paper/tex/appendix.tex", "holds ten rule families", "holds nine rule families"),
    ("a quoted fixtures byte total that is not the tree's actual size",   # check_quoted_sizes
     "README.md", "957,824", "957,300"),
    ("a factor claim that no longer follows from its two operands",       # check_factor_claims
     "../paper/tex/sec1_intro.tex", "factor of $8.542$", "factor of $8.642$"),
    ("a floor-dependent value quoted with no floor named",                # check_floor_declarations
     "../paper/tex/appendix.tex", "$16.46$ at \\epslib{} by it", "$16.46$ by it"),
    ("a documented stage tag that no stage in verify.py emits",           # check_stage_tags
     "README.md", "[coverage] ...", "[covrage] ..."),
    ("two documented stages listed out of execution order",               # check_transcript_order
     "README.md", "[map] ...   [gate3] ...", "[gate3] ...   [map] ..."),
    # These three are the rules that no earlier mutation OWNED: every mutation reaching
    # them was also caught by a broader rule, so each could be replaced by `return []`
    # with the validator still reporting a full pass. Each edit below fires its rule
    # and no other.
    ("one shipped copy of the instrument edited without the other",       # ..._copies
     "instrument/README.md", "## Scope", "## Scopes"),
    ("the MANIFEST attributing a fixture to another shipped script",      # producer_relations
     "MANIFEST.md", "`scripts/rb_census.py`", "`scripts/aggregate_results.py`"),
]

# Which rule MUST catch each mutation, by label. "Some rule fired" is too weak: it lets a
# rule be neutered silently whenever a broader rule happens to cover the same edit.
_MUTATION_OWNER = {
    "a renamed script whose docstrings keep the old name": "check_paths",
    "a dependency floor below the APIs the scripts call": "check_deps",
    "a packaged-window count off by one": "check_counts",
    "a census row count off by one": "check_counts",
    "a producer string naming a script that was never shipped": "check_paths",
    "the enumerated check total changed on every prose surface at once": "check_counts",
    "a real filename claimed under a directory it does not live in": "check_paths",
    "an advertised flag the parser does not accept": "check_cli_examples",
    "a worked numeric example that no longer evaluates to its claim": "check_numeric_examples",
    "the row-file writer bypassing the single-sourced contract": "check_io_contract",
    "a stage comment contradicting the fixture it introduces": "check_counts",
    "a documented command left with an abbreviated identifier": "check_command_placeholders",
    "an unmarked restatement of a withdrawn claim in the packaged ledger":
        "check_withdrawal_markers",
    "a shipped surface claiming a floored score is capped": "check_saturation_claim",
    "a shipped script defaulting to a write outside the package": "check_producer_targets",
    "the guard's own family count misstated in its docstring": "check_own_counts",
    "the manuscript misstating how many rule families the guard holds":
        "check_own_counts_manuscript",
    "a quoted fixtures byte total that is not the tree's actual size": "check_quoted_sizes",
    "a factor claim that no longer follows from its two operands": "check_factor_claims",
    "a floor-dependent value quoted with no floor named": "check_floor_declarations",
    "a documented stage tag that no stage in verify.py emits": "check_stage_tags",
    "two documented stages listed out of execution order": "check_transcript_order",
    "one shipped copy of the instrument edited without the other": "check_instrument_copies",
    "the MANIFEST attributing a fixture to another shipped script": "check_producer_relations",
}


def validate_guard_fires() -> int:
    import os
    import shutil
    import subprocess
    import tempfile

    missed: list[str] = []
    skipped_m: list[str] = []
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
                        # Copy only regular files and directories. The parent is whatever directory
            # a reviewer happened to clone into, and copytree raises on anything else --
            # a unix socket or a dangling symlink there gave an uncaught shutil.Error at
            # exactly this line, in the one mode whose whole purpose is to return a
            # verdict. Skipping the unreadable entry is right: nothing this validator
            # checks lives in a socket, and a crash here reads as the guard failing.
            def _ignore(src, names, _ig=ignore):
                skip = set(_ig(src, names))
                for n in names:
                    p = os.path.join(src, n)
                    if not (os.path.isdir(p) or os.path.isfile(p)) or (
                            os.path.islink(p) and not os.path.exists(p)):
                        skip.add(n)
                return skip
            shutil.copytree(ROOT.parent, root, ignore=_ignore)
            d = root / ROOT.name
            f = (root / rel[len("../"):]) if rel.startswith("../") else (d / rel)
            # A mutation is unreachable when its OWNING RULE cannot run in this tree,
            # even though the subject file is present. In a standalone clone of the package
            # `instrument/README.md` exists inside it, but check_instrument_copies compares
            # against a sibling copy that does not, so the rule returns [] and the mutation
            # was recorded as MISSED -- making the advertised mode exit 1 in exactly the
            # artifact the MANIFEST tells a reviewer to download. Skip it, as with an
            # absent subject: unavailable is not the same as uncaught.
            _own = _MUTATION_OWNER.get(label)
            _ownfn = dict((r.__name__, r) for _f, r in _rules()).get(_own) if _own else None
            _need = getattr(_ownfn, "_needs_source_repo", None) if _ownfn else None
            if _need is not None and not _need():
                skipped_m.append(f"{label} (owning rule {_own} unavailable here)")
                continue
            if not f.exists():
                # The subject lives outside the package and is absent from a
                # standalone clone. Report it rather than raising: the mode a
                # reviewer is told to run must run in the artifact of record.
                skipped_m.append(f"{label} (subject absent: {rel})")
                continue
            src = f.read_text(encoding="utf-8")
            if old not in src:
                missed.append(f"{label}: anchor {old!r} no longer in {rel}")
                continue
            f.write_text(src.replace(old, new, 1), encoding="utf-8")
            owner = _MUTATION_OWNER.get(label)
            r = subprocess.run([sys.executable, "scripts/check_self_claims.py",
                                "--which-rules"], cwd=d, capture_output=True, text=True)
            fired = set(r.stdout.split("RULES_FIRED:")[-1].split()) if \
                "RULES_FIRED:" in r.stdout else set()
            if r.returncode == 0:
                missed.append(f"{label}: reintroduced in {rel}, guard still passed")
            elif owner and owner not in fired:
                # Caught, but by a DIFFERENT rule -- so the rule this mutation exists to
                # protect could be deleted and this entry would still report a pass.
                missed.append(f"{label}: caught, but not by {owner} "
                              f"(fired: {', '.join(sorted(fired)) or 'none'})")
            else:
                print(f"  caught: {label}" + (f"  [{owner}]" if owner else ""))
    if missed:
        print("FAIL: the guard does not catch what it was written for:")
        for m in missed:
            print(f"  - {m}")
        return 1
    _n = len(_MUTATIONS) - len(skipped_m)
    for _s in skipped_m:
        print(f"  skipped: {_s}")
    print(f"[selfclaims] guard validated: {_n}/{_n} reintroduced discrepancies caught"
          + (f"; {len(skipped_m)} skipped (subject outside this package)" if skipped_m else ""))
    return 0


def main() -> int:
    if "--which-rules" in sys.argv:
        # Names the rules that fired, so --validate-guard can require that a mutation is
        # caught by the rule it was written for rather than by any rule at all.
        _fired = []
        for _fam, _r in _rules():
            try:
                if _r():
                    _fired.append(_r.__name__)
            except Exception:
                _fired.append(_r.__name__)
        print("RULES_FIRED: " + " ".join(_fired))
        return 1 if _fired else 0
    if "--validate-guard" in sys.argv:
        return validate_guard_fires()
    n, findings = run()
    if findings:
        print(f"FAIL: {len(findings)} self-description(s) do not match the package:")
        for f in findings:
            print(f"  - {f}")
        return 1
    _sk = (f"; {len(_LAST_SKIPPED)} rule(s) SKIPPED -- their subject lives outside this "
           f"package and is absent here ({', '.join(_LAST_SKIPPED)})") if _LAST_SKIPPED else ""
    print(f"[selfclaims] OK over {n} surfaces: {len(_rules()) - len(_LAST_SKIPPED)} of "
          f"{len(_rules())} rules across {len({f for f, _ in _rules()})} families found nothing{_sk}. Each binds "
          f"a NAMED set, not a universal -- the counts checked are the ones enumerated in "
          f"the module docstring, the producer rule the relations listed there, the marker "
          f"rule literal fragments (a paraphrase of a withdrawn claim passes). It is also "
          f"one-directional: it checks that what the package says is true, never that "
          f"everything the package ships is said.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
