#!/usr/bin/env python3
"""The one place this package decides how a per-window row file is stored.

`rb_model_eval.py` writes these; `rb_checkpoint_audit.py` reads them. The two
statements used to live in two files and drifted: the writer named its output
`.json.gz` and then wrote it uncompressed, so the documented regeneration path
produced files the aggregator crashed on. Reported numbers were unaffected --
the packaged fixtures were written before the drift -- but a cold-start reader
following MANIFEST.md hit a `BadGzipFile`.

Both sides now call these two functions, and `check_self_claims.py` round-trips
them, so a future divergence fails the harness instead of the reader.
"""

from __future__ import annotations

import gzip
import json
import os
from typing import Any

SUFFIX = ".json.gz"


def write_rows(path: str, payload: dict[str, Any]) -> str:
    """Serialize a row payload to `path`, creating its directory."""
    if not path.endswith(SUFFIX):
        raise ValueError(f"row files must end in {SUFFIX}: {path}")
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8") as fh:
        json.dump(payload, fh)
    return path


def read_rows(path: str) -> dict[str, Any]:
    """Read back a payload written by `write_rows`."""
    with gzip.open(path, "rt", encoding="utf-8") as fh:
        return json.load(fh)
