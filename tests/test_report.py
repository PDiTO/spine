from __future__ import annotations

import json
from pathlib import Path

import pytest

from spine import cli
from spine.items import Item
from spine.providers import Recorder
from spine.run import Config, ModelSpec, request_for, run_model
from tests.conftest import scripted

CONFIG = """
[run]
seed = 7
items = 24
levels = [1, 2]
bootstrap = 100
recordings = "{rec}"

[[models]]
name = "fake-oracle"
label = "Oracle"
provider = "claude"
model = "fake-oracle"

[[models]]
name = "fake-folds"
label = "Folder"
provider = "codex"
model = "fake-folds"
levels = [1, 2]
"""


@pytest.fixture
def recorded(tmp_path: Path, items: list[Item]) -> Path:
    """A config whose recordings were made by scripted fake models."""
    rec = tmp_path / "recordings"
    path = tmp_path / "main.toml"
    path.write_text(CONFIG.format(rec=rec))
    cfg = Config.load(path)
    for spec, behaviour in zip(cfg.models, ("oracle", "folds"), strict=True):
        recorder = Recorder(scripted(items, behaviour), rec / f"{spec.name}.jsonl")
        run_model(spec, cfg.build_items(spec), recorder, workers=2)
    return path


def test_report_rebuilds_everything_offline(recorded: Path, tmp_path: Path) -> None:
    out, docs = tmp_path / "results", tmp_path / "docs"
    code = cli.main(["--config", str(recorded), "report", "--out", str(out), "--docs", str(docs)])
    assert code == 0
    summary = json.loads((out / "summary.json").read_text())
    by_label = {m["label"]: m for m in summary["models"]}
    assert by_label["Oracle"]["metrics"]["spine"]["value"] == 1.0
    assert by_label["Folder"]["metrics"]["spine"]["value"] == 0.0
    assert by_label["Folder"]["metrics"]["ears"]["value"] == 1.0
    assert summary["total_calls"] == sum(m["calls"] for m in summary["models"])
    report = (out / "report.md").read_text()
    assert "| Oracle |" in report
    assert "fake rationale" in report
    for name in ("spine-vs-ears", "by-style", "by-task", "outcomes", "accuracy"):
        svg = (docs / f"{name}.svg").read_text()
        assert svg.startswith("<?xml")
    assert (out / "trials.csv").read_text().count("\n") > 10

    # Byte-identical on a second run.
    before = (docs / "spine-vs-ears.svg").read_bytes()
    cli.main(["--config", str(recorded), "report", "--out", str(out), "--docs", str(docs)])
    assert (docs / "spine-vs-ears.svg").read_bytes() == before


def test_report_fails_loudly_on_a_missing_recording(recorded: Path, tmp_path: Path) -> None:
    for f in (tmp_path / "recordings").glob("fake-folds.jsonl"):
        lines = f.read_text().splitlines()
        f.write_text("\n".join(lines[:-3]) + "\n")
    with pytest.raises(Exception, match="no recording"):
        cli.main(["--config", str(recorded), "report", "--out", str(tmp_path / "r")])


def test_items_command(recorded: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert cli.main(["--config", str(recorded), "items", "--show", "2"]) == 0
    assert "24 items" in capsys.readouterr().out


def test_pilot_offline_replays(recorded: Path, tmp_path: Path, items: list[Item]) -> None:
    cfg = Config.load(recorded)
    spec = cfg.models[0]
    pilot_items = cfg.build_items()
    rec = tmp_path / "pilot"
    from spine.items import build_items

    pilot = build_items(cfg.seed + 1, len(cfg.tasks) * len(cfg.levels), cfg.tasks, cfg.levels)
    run_model(spec, pilot, Recorder(scripted(pilot, "oracle"), rec / f"{spec.name}.jsonl"),
              second_turns=False)  # fmt: skip
    code = cli.main(
        ["--config", str(recorded), "pilot", "--models", spec.name, "--recordings", str(rec),
         "--offline"]
    )  # fmt: skip
    assert code == 0
    assert pilot_items


def test_unknown_model_is_an_error(recorded: Path) -> None:
    with pytest.raises(SystemExit):
        cli.main(["--config", str(recorded), "run", "--models", "nope"])


def test_request_for_uses_spec() -> None:
    spec = ModelSpec("n", "claude", "m", effort="low", thinking=False)
    r = request_for(spec, [])
    assert (r.provider, r.model, r.effort, r.thinking) == ("claude", "m", "low", False)
