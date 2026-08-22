#!/usr/bin/env python3
"""Every required reference block must produce a NAMED failure when it is absent.

Why this test exists, stated plainly because the history is the justification:

`verify.py` consumed its reference blocks with `reference.get(key, {})`. A missing
block therefore made the consuming stage iterate nothing, print its usual OK line,
and let the harness exit 0 -- while the manuscript went on citing numbers nothing
had checked. That defect was found and repaired once, in a single stage, with a
comment written above it naming the problem; two stages immediately BELOW that
comment still had it, as did the Gate-3 summary leaf. Three independent reviewers
demonstrated it by deleting a block from both number tables and watching PASS.

Repairing the sites was not enough, because the next stage added would reintroduce
it. This test makes the contract mechanical: for each key in
`verify.REQUIRED_REFERENCE_BLOCKS`, delete that key from BOTH number tables in a
private copy and require verify.py to exit non-zero AND to name the key. Deleting
from only one copy proves nothing -- the reference-identity stage catches that on
its own and would mask a genuinely silent stage.

Run: python3 scripts/test_absent_reference_blocks.py
"""
import json
import pathlib
import shutil
import subprocess
import sys
import tempfile

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parent
LANE = ROOT.parent
NUMBERS = ("paper/extracted/numbers.json",
           "repro-harness/fixtures/numbers_reference.json")


def main() -> int:
    sys.path.insert(0, str(ROOT))
    import verify as _v
    keys = list(_v.REQUIRED_REFERENCE_BLOCKS)
    if not keys:
        print("FAIL: verify.REQUIRED_REFERENCE_BLOCKS is empty")
        return 1

    bad = []
    with tempfile.TemporaryDirectory() as td:
        lane = pathlib.Path(td) / "lane"
        shutil.copytree(LANE, lane, ignore=shutil.ignore_patterns(
            "__pycache__", "*.pyc", ".git", "hf-cache", ".gate-work", "results",
            "tmp", "*.pdf"))
        pristine = {p: (lane / p).read_text() for p in NUMBERS}
        for key in keys:
            for p in NUMBERS:
                d = json.loads(pristine[p])
                d.pop(key, None)
                # keep the byte count close so a size-based check cannot stand in
                # for the stage under test
                # Pad to the ORIGINAL byte length exactly. An earlier version computed the
                # padding before inserting the padding key, so every mutated file came out
                # ~30 bytes short -- which tripped an unrelated fixtures-size check and made
                # this test pass for the wrong reason, hiding a stage that printed a failure
                # and still exited 0.
                d["_absence_test_padding"] = ""
                short = len(pristine[p]) - len(json.dumps(d, indent=1))
                d["_absence_test_padding"] = "x" * max(0, short)
                (lane / p).write_text(json.dumps(d, indent=1))
            r = subprocess.run([sys.executable, "verify.py"],
                               cwd=lane / "repro-harness",
                               capture_output=True, text=True)
                        # The acceptance criterion must be the SPECIFIC failure this test is about,
            # not "some failure happened and the key appears somewhere". A reviewer showed
            # that a nonzero exit from an unrelated check, plus the key appearing anywhere
            # in stdout, satisfied the old criterion -- so a reported 11/11 could be a
            # false positive for a block that was in fact still fail-open. We now require
            # the exact sentinel `_req` emits for an absent block.
                        # Require the sentinel `_req` emits for THIS key. Not "some failure plus the
            # key appears somewhere": a reviewer satisfied that weaker criterion with an
            # unrelated nonzero exit, making a reported 11/11 a false positive. The
            # end-of-run summary is deliberately NOT required -- a stage that returns 1
            # early is correct behaviour and makes the summary unreachable.
            sentinel = f"required reference block absent: {key}"
            named = sentinel in r.stdout
            if r.returncode == 0 or not named:
                bad.append(f"{key}: exit={r.returncode} sentinel={named}")
                print(f"  [FAIL] {key}: exit={r.returncode}, key named in output={named}")
            else:
                print(f"  [ok]   {key}: absence produces a named failure")
            for p in NUMBERS:
                (lane / p).write_text(pristine[p])

    if bad:
        print(f"\nFAIL: {len(bad)} of {len(keys)} required blocks can vanish without a "
              f"named failure")
        return 1
    print(f"\nPASS: all {len(keys)} required reference blocks fail loudly when absent")
    return 0


if __name__ == "__main__":
    sys.exit(main())
