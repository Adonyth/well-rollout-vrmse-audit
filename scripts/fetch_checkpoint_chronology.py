#!/usr/bin/env python3
"""Record, from the model hub's own commit API, when every audited checkpoint was uploaded.

Why this is a first-class artifact rather than a sentence in the paper: the whole
reproduction gap this paper reports is a comparison between (a) cells published in a
benchmark paper and (b) scores we compute from checkpoints downloaded today. If (b)'s
weights post-date (a), then no arithmetic we do can attribute the gap to an evaluation
protocol -- the comparator is a different artifact. That is a provenance fact about OUR
audited objects, so it is fetched from the hub and frozen, not asserted in prose.

Network pass. The frozen output ships with the harness; verify.py checks the paper against
the frozen copy, so the offline harness never needs the network. Re-run to refresh:

    python3 scripts/fetch_checkpoint_chronology.py

Both dates per repo are recorded. `latest` is the point: if a repo had been revised after
upload, the audited revision could not be pinned to a single date, and the paper would have
to say so. As fetched, no audited repo has any commit outside its upload day.
"""
import json
import os
import sys
import urllib.request
from datetime import date

HERE = os.path.dirname(os.path.abspath(__file__))
HARNESS = os.path.dirname(HERE)
DEST = os.path.join(HARNESS, "fixtures", "provenance", "checkpoint_chronology.json")

# The benchmark's arXiv submission history, fetched from the abstract page rather than
# inferred from the identifier's YYMM prefix. BOTH versions matter and the second one is the
# one that binds: the published cells this paper audits are quoted from v2
# (fixtures/generalization/well_table3_gt10.json records that provenance), so the decisive
# interval is checkpoint-upload minus v2, not minus v1. Quoting only the larger v1 gap would
# overstate the separation between our audited artifacts and the table we compare them to.
ARXIV_ABS = "https://arxiv.org/abs/2412.00568"
QUOTED_TABLES_VERSION = "v2"
DATASETS = ["rayleigh_taylor_instability", "rayleigh_benard"]
MODELS = ["FNO", "TFNO", "UNetClassic", "UNetConvNext"]
API = "https://huggingface.co/api/models/{repo}/commits/main"
_MONTHS = {m: i + 1 for i, m in enumerate(
    ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"])}


def fetch_arxiv_versions():
    """Parse the submission-history block: [vN] <day>, DD Mon YYYY HH:MM:SS UTC."""
    import re
    req = urllib.request.Request(ARXIV_ABS, headers={"User-Agent": "well-audit-harness"})
    with urllib.request.urlopen(req, timeout=60) as f:
        page = f.read().decode("utf-8", "replace")
    # Strip tags first: the version markers are wrapped inconsistently (v1 is a link, later
    # versions are bare), and matching the plain text is the stable read.
    text = re.sub(r"<[^>]+>", " ", page)
    hits = re.findall(r"\[(v\d+)\]\s*[A-Za-z]{3},\s*(\d{1,2})\s+([A-Za-z]{3})\s+(\d{4})", text)
    out = {}
    for ver, dd, mon, yyyy in hits:
        out[ver] = f"{int(yyyy):04d}-{_MONTHS[mon]:02d}-{int(dd):02d}"
    return out


def fetch(repo):
    with urllib.request.urlopen(API.format(repo=repo), timeout=60) as f:
        commits = json.load(f)
    dates = sorted(c["date"][:10] for c in commits)
    return {"n_commits": len(commits), "initial_commit": dates[0], "latest_commit": dates[-1],
            "revised_after_upload": dates[0] != dates[-1]}


def days_after(iso, anchor):
    return (date.fromisoformat(iso) - date.fromisoformat(anchor)).days


def main():
    versions = fetch_arxiv_versions()
    if QUOTED_TABLES_VERSION not in versions or "v1" not in versions:
        print(f"FAIL: arXiv submission history not parsed (got {versions})")
        return 1
    v1, vq = versions["v1"], versions[QUOTED_TABLES_VERSION]
    repos = {f"polymathic-ai/{m}-{d}": None for d in DATASETS for m in MODELS}
    out = {"benchmark_arxiv_versions": versions,
           "benchmark_arxiv_v1": v1,
           "quoted_tables_version": QUOTED_TABLES_VERSION,
           "quoted_tables_version_date": vq,
           "fetched_utc": None, "source": API, "arxiv_source": ARXIV_ABS,
           "repos": {}}
    for repo in repos:
        try:
            rec = fetch(repo)
        except Exception as exc:  # fail loudly: a partial chronology is worse than none
            print(f"FAIL: {repo}: {exc}")
            return 1
        rec["days_after_benchmark_paper"] = days_after(rec["initial_commit"], v1)
        rec["days_after_quoted_tables_version"] = days_after(rec["initial_commit"], vq)
        # repo is "polymathic-ai/<Model>-<dataset>"; splitting on the first "-"
        # cut inside the ORG name and produced "ai/FNO-rayleigh_benard".
        _model, _, _ds = repo.split("/", 1)[1].partition("-")
        rec["model"] = _model
        rec["dataset"] = _ds
        out["repos"][repo] = rec
        print(f"  {repo}: {rec['n_commits']} commits, {rec['initial_commit']} .. "
              f"{rec['latest_commit']} (+{rec['days_after_benchmark_paper']}d after v1, "
              f"+{rec['days_after_quoted_tables_version']}d after "
              f"{QUOTED_TABLES_VERSION})")

    initial = sorted({r["initial_commit"] for r in out["repos"].values()})
    out["summary"] = {
        "n_repos": len(out["repos"]),
        "all_initial_commits_same_day": len(initial) == 1,
        "initial_commit_dates": initial,
        "any_revised_after_upload": any(r["revised_after_upload"]
                                        for r in out["repos"].values()),
        "min_days_after_benchmark_paper": min(r["days_after_benchmark_paper"]
                                              for r in out["repos"].values()),
        "max_days_after_benchmark_paper": max(r["days_after_benchmark_paper"]
                                              for r in out["repos"].values()),
        "min_days_after_quoted_tables_version":
            min(r["days_after_quoted_tables_version"] for r in out["repos"].values()),
        "max_days_after_quoted_tables_version":
            max(r["days_after_quoted_tables_version"] for r in out["repos"].values()),
    }
    out["fetched_utc"] = os.environ.get("CHRONOLOGY_FETCH_UTC", "")
    os.makedirs(os.path.dirname(DEST), exist_ok=True)
    with open(DEST, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1, sort_keys=True)
    print(f"wrote {DEST}")
    s = out["summary"]
    print(f"  {s['n_repos']} audited checkpoints; every commit in "
          f"{s['initial_commit_dates']}; revised after upload: {s['any_revised_after_upload']}; "
          f"{s['min_days_after_benchmark_paper']}-{s['max_days_after_benchmark_paper']} days "
          f"after the benchmark's arXiv v1 ({out['benchmark_arxiv_v1']})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
