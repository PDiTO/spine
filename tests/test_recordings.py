"""The committed recordings are published, so check them like retro's audit does."""

from __future__ import annotations

import json
import re

import pytest

from tests.conftest import ROOT

FILES = sorted((ROOT / "recordings").rglob("*.jsonl"))
PRIVATE = re.compile(
    r"(/home/|/Users/|sk-[A-Za-z0-9-]{16,}|Bearer\s|ts\.net|gho_|ghp_|BEGIN [A-Z ]*PRIVATE KEY)"
)


@pytest.mark.parametrize("path", FILES, ids=[str(p.relative_to(ROOT)) for p in FILES])
def test_recordings_hold_no_private_data(path: object) -> None:
    from pathlib import Path

    assert isinstance(path, Path)
    for n, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        hit = PRIVATE.search(line)
        assert hit is None, f"{path.name}:{n}: {hit.group(0) if hit else ''}"
        row = json.loads(line)
        assert set(row) == {"key", "request", "response"}
