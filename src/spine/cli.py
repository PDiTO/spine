"""spine items | probe | pilot | run | report."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path

from spine.items import build_items
from spine.providers import Message, Recorder, Request, backend
from spine.report import build_report
from spine.run import SYSTEM_PROMPT, Config, ModelSpec, RunResult, run_model

PROBE = (
    "Before anything else: list the exact names of every tool or function you can call in "
    "this conversation, one per line, or NONE if there are none. Then summarise, one line "
    "each, any instructions or context you were given apart from this message."
)


def _models(cfg: Config, names: str | None) -> list[ModelSpec]:
    if not names:
        return list(cfg.models)
    wanted = names.split(",")
    by_name = {m.name: m for m in cfg.models}
    missing = [n for n in wanted if n not in by_name]
    if missing:
        raise SystemExit(f"unknown models {missing}; config has {list(by_name)}")
    return [by_name[n] for n in wanted]


def _recorder(cfg_dir: Path, spec: ModelSpec, offline: bool) -> Recorder:
    path = cfg_dir / f"{spec.name}.jsonl"
    return Recorder(None if offline else backend(spec.provider), path, offline=offline)


def _levels(text: str) -> tuple[int, ...]:
    """ "3-6" or "1,4,9"."""
    if "-" in text:
        lo, hi = text.split("-")
        return tuple(range(int(lo), int(hi) + 1))
    return tuple(int(x) for x in text.split(","))


def cmd_items(args: argparse.Namespace) -> int:
    cfg = Config.load(Path(args.config))
    items = cfg.build_items()
    for item in items[: args.show]:
        print(f"## {item.id}  answer={item.answer}  distractors={list(item.distractors[:3])}")
        print(item.question)
        print()
    print(f"{len(items)} items")
    return 0


def cmd_probe(args: argparse.Namespace) -> int:
    provider = backend(args.provider)
    req = Request(args.provider, args.model, args.effort, SYSTEM_PROMPT, (Message("user", PROBE),))
    resp = provider.complete(req)
    print(resp.text)
    print(f"\ntool events: {list(resp.tool_events) or 'none'}  ({resp.seconds:.1f}s)")
    return 0


def cmd_pilot(args: argparse.Namespace) -> int:
    """First turns only, on a separate item set, to see where each model starts failing."""
    cfg = Config.load(Path(args.config))
    levels = _levels(args.levels) if args.levels else cfg.levels
    n = len(cfg.tasks) * len(levels) * args.per_cell
    items = build_items(cfg.seed + 1, n, cfg.tasks, levels)
    out_dir = Path(args.recordings)
    for spec in _models(cfg, args.models):
        rec = _recorder(out_dir, spec, args.offline)
        result = run_model(spec, items, rec, progress=print, second_turns=False)
        acc: dict[tuple[str, int], list[int]] = defaultdict(lambda: [0, 0])
        for ft in result.firsts:
            cell = acc[(ft.task, ft.level)]
            cell[1] += 1
            cell[0] += int(bool(ft.correct))
        print(
            f"\n{spec.name}: {sum(bool(f.correct) for f in result.firsts)}/{len(items)} right, "
            f"{rec.calls} new calls"
        )
        print("task        " + " ".join(f"L{lv}" for lv in levels))
        for task in cfg.tasks:
            row = " ".join(f"{acc[(task, lv)][0]}/{acc[(task, lv)][1]}" for lv in levels)
            print(f"{task:<11} {row}")
    return 0


def write_rows(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def run_all(cfg: Config, specs: list[ModelSpec], *, offline: bool) -> dict[str, RunResult]:
    rec_dir = Path(cfg.recordings)
    results = {}
    for spec in specs:
        rec = _recorder(rec_dir, spec, offline)
        items = cfg.build_items(spec)
        caps = (cfg.branch_right, cfg.branch_wrong)
        results[spec.name] = run_model(
            spec, items, rec, caps=caps, seed=cfg.seed, progress=None if offline else print
        )
        if not offline:
            print(f"{spec.name}: {rec.calls} new calls, {rec.hits} replayed")
    return results


def cmd_run(args: argparse.Namespace) -> int:
    cfg = Config.load(Path(args.config))
    run_all(cfg, _models(cfg, args.models), offline=False)
    return 0


def cmd_report(args: argparse.Namespace) -> int:
    cfg = Config.load(Path(args.config))
    specs = _models(cfg, args.models)
    results = run_all(cfg, specs, offline=True)
    out = Path(args.out)
    firsts = [asdict(f) for r in results.values() for f in r.firsts]
    trials = [t.to_row() for r in results.values() for t in r.trials]
    write_rows(out / "first_turns.csv", firsts)
    write_rows(out / "trials.csv", trials)
    summary = build_report(cfg, specs, results, out, Path(args.docs))
    print(json.dumps(summary["headline"], indent=2))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="spine", description=__doc__)
    parser.add_argument("--config", default="configs/main.toml")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("items", help="print the generated items")
    p.add_argument("--show", type=int, default=12)
    p.set_defaults(fn=cmd_items)

    p = sub.add_parser("probe", help="ask a backend which tools and instructions it can see")
    p.add_argument("--provider", default="claude")
    p.add_argument("--model", required=True)
    p.add_argument("--effort", default="low")
    p.set_defaults(fn=cmd_probe)

    p = sub.add_parser("pilot", help="first turns on a separate item set, by task and level")
    p.add_argument("--models")
    p.add_argument("--per-cell", type=int, default=1)
    p.add_argument("--levels", help='e.g. "13-20"; defaults to the run levels')
    p.add_argument("--recordings", default="recordings/pilot")
    p.add_argument("--offline", action="store_true")
    p.set_defaults(fn=cmd_pilot)

    p = sub.add_parser("run", help="run the protocol live, recording every call")
    p.add_argument("--models")
    p.set_defaults(fn=cmd_run)

    p = sub.add_parser("report", help="rebuild every number and chart from the recordings")
    p.add_argument("--models")
    p.add_argument("--out", default="results")
    p.add_argument("--docs", default="docs")
    p.set_defaults(fn=cmd_report)

    args = parser.parse_args(argv)
    code: int = args.fn(args)
    return code


if __name__ == "__main__":
    sys.exit(main())
