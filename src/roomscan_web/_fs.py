"""File helpers shared by the job records and the derived-PLY cache."""

from __future__ import annotations

import time
from pathlib import Path


def replace_atomic(tmp: Path, dst: Path, *, attempts: int = 5) -> None:
    """`tmp.replace(dst)`, retried: on Windows a virus scanner or indexer can hold `dst` open for a moment."""
    for i in range(attempts):
        try:
            tmp.replace(dst)
            return
        except PermissionError:
            if i == attempts - 1:
                raise
            time.sleep(0.05 * (i + 1))
