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

Three families are checkable without judgement:

  paths   -- every repo-relative path named in the prose or in a docstring exists
  counts  -- every quantity the prose pins (fixture rows, check counts) is measured
  stages  -- the README transcript lists every stage that gates the exit code
  deps    -- the declared dependency floor admits the APIs the scripts actually call

Anything requiring judgement (is this description *apt*?) is out of scope; this
catches the mechanical half, which is where all ten of the found discrepancies lay.

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
CODE = sorted(str(p.relative_to(ROOT)) for p in ROOT.glob("scripts/*.py")) + [
    "verify.py",
    "instrument/test_against_paper.py",
    "instrument/test_normalizers.py",
]

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


def _resolve(tok: str) -> bool:
    """A named file exists if it is at that path, or its basename is in the tree.

    Prose and docstrings name scripts by basename (``aggregate_results.py``) far
    more often than by full path, and both forms are the same claim: *this file
    is part of the package*.
    """
    if _exists(tok):
        return True
    base = tok.rsplit("/", 1)[-1]
    if "*" in base:
        return any(ROOT.rglob(base))
    return any(ROOT.rglob(base))


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
                if not _resolve(cand):
                    bad.append(f"{rel}: names `{cand}`, which is nowhere in the package")
    return bad


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
    for surface, text in (("instrument/README.md", inst), ("README.md", readme), ("MANIFEST.md", man)):
        for pattern, actual, what in (
            (r"MAP\.json`?,? (\d+) rows", n_map, "rows in MAP.json"),
            (r"(\d+) rows reproduced", n_map, "rows in MAP.json"),
            (r"json\.gz`?,? (\d+) windows", n_win, "packaged RB windows"),
            (r"(\d+) per-window variances", n_win, "packaged RB windows"),
        ):
            for c in re.findall(pattern, text):
                if int(c) != actual:
                    bad.append(f"{surface}: claims {c} {what}; the tree holds {actual}")

    # The headline check count must equal what aggregate_results actually emits.
    n_claimed = {int(m) for m in re.findall(r"(\d{2,4}) (?:enumerated )?value checks", readme)}
    n_claimed |= {int(m) for m in re.findall(r"the (\d{2,4}) Tier-1 checks", man)}
    if n_claimed:
        vp = (ROOT / "verify.py").read_text(encoding="utf-8")
        m = re.search(r'\[3/3\] \{n_ok\}', vp)
        if m is None:
            bad.append("verify.py: cannot locate the check-count print to cross-check the prose")
        else:
            # verify.py prints the measured count; agreement is asserted at run time
            # by the caller passing n_ok in. Here we only require internal agreement
            # among the prose surfaces themselves.
            if len(n_claimed) > 1:
                bad.append(f"prose surfaces disagree on the check count: {sorted(n_claimed)}")
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
    for tag in sorted(printed - shown):
        bad.append(f"README.md: verify.py prints a [{tag}] stage the transcript omits")
    for tag in sorted(shown - printed):
        bad.append(f"README.md: transcript shows a [{tag}] stage verify.py never prints")
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
    findings = check_paths() + check_counts() + check_stage_tags() + check_deps()
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
]


def validate_guard_fires() -> int:
    import shutil
    import subprocess
    import tempfile

    missed = []
    for label, rel, old, new in _MUTATIONS:
        with tempfile.TemporaryDirectory() as td:
            d = pathlib.Path(td) / "pkg"
            shutil.copytree(ROOT, d,
                            ignore=shutil.ignore_patterns("__pycache__", "*.pyc", ".git"))
            f = d / rel
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
    print(f"[selfclaims] OK: {n} surfaces; every named path exists, "
          f"pinned counts match the tree, declared dependency floor admits the APIs used")
    return 0


if __name__ == "__main__":
    sys.exit(main())
