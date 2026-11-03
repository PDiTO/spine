"""SVG charts for the README, from the summary dict that `report` writes.

Plain matplotlib with a fixed palette and a fixed hash salt, so re-running the
report on the same recordings produces byte-identical files.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle

from spine.pushback import STYLE_LABELS

# Categorical slots in fixed order (validated light-mode steps).
PALETTE = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#6250d6", "#008300")
INK = "#0b0b0b"
INK_2 = "#52514e"
GRID = "#e4e3df"
OUTCOME_COLOURS = {
    "kept": "#2a78d6",
    "adopted": "#eb6834",
    "corrected": "#1baf7a",
    "other": "#b9b8b2",
    "hedged": "#6250d6",
}

plt.rcParams.update(
    {
        "svg.hashsalt": "spine",
        "svg.fonttype": "none",
        "font.family": "sans-serif",
        "font.sans-serif": ["Inter", "Helvetica", "Arial", "DejaVu Sans"],
        "font.size": 10,
        "axes.edgecolor": INK_2,
        "axes.labelcolor": INK,
        "axes.titlecolor": INK,
        "axes.titlesize": 11,
        "axes.titleweight": "bold",
        "axes.titlelocation": "left",
        "xtick.color": INK_2,
        "ytick.color": INK_2,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "legend.frameon": False,
    }
)


# Where each model's name sits next to its point in the scatter, in points. Hand
# placed for the published run so names don't sit on top of each other.
LABEL_OFFSETS: dict[str, tuple[float, float, str]] = {
    "Claude Sonnet 4.6": (-8, -16, "right"),
    "Claude Haiku 5.5": (6, 10, "left"),
    "GPT-6.1 Sol": (10, -2, "left"),
    "Claude Opus 5.5": (10, -9, "left"),
}


def _save(fig: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, format="svg", bbox_inches="tight", metadata={"Date": None})
    plt.close(fig)


def _colour(i: int) -> str:
    return PALETTE[i % len(PALETTE)]


def _marker(model: dict[str, Any]) -> str:
    return "o" if model["provider"] == "claude" else "s"


def _m(model: dict[str, Any], key: str) -> dict[str, Any]:
    out: dict[str, Any] = model["metrics"][key]
    return out


def spine_vs_ears(summary: dict[str, Any], path: Path) -> None:
    models = summary["models"]
    fig, ax = plt.subplots(figsize=(7.2, 6.4))
    ax.plot([0, 1], [1, 0], color=INK_2, lw=1, ls=(0, (4, 3)), zorder=1)
    ax.text(
        0.2, 0.47,
        "dashed line: no discrimination,\nmoves the same whether\nthe user is right or wrong",
        color=INK_2, fontsize=8.5, ha="center", va="center",
    )  # fmt: skip
    for x, y, label, ha, va in (
        (0.02, 0.98, "folds to anything", "left", "top"),
        (0.98, 0.98, "holds when right,\nlistens when wrong", "right", "top"),
        (0.98, 0.02, "never moves", "right", "bottom"),
        (0.02, 0.02, "contrarian", "left", "bottom"),
    ):
        ax.text(x, y, label, transform=ax.transAxes, ha=ha, va=va, color=INK_2,
                fontsize=9, style="italic")  # fmt: skip
    for i, m in enumerate(models):
        s, e = _m(m, "spine"), _m(m, "ears")
        if s["value"] is None or e["value"] is None:
            continue
        c = _colour(i)
        nk, nf = _m(m, "neutral_kept_when_right"), _m(m, "neutral_fixed_when_wrong")
        if nk["value"] is not None and nf["value"] is not None:
            ax.plot([nk["value"], s["value"]], [nf["value"], e["value"]], color=c, lw=0.8,
                    ls=(0, (1, 2)), alpha=0.7, zorder=2)  # fmt: skip
            ax.scatter([nk["value"]], [nf["value"]], s=40, facecolor="white", edgecolor=c,
                       lw=1.5, marker=_marker(m), zorder=3)  # fmt: skip
        ax.errorbar(
            s["value"], e["value"],
            xerr=[[s["value"] - s["lo"]], [s["hi"] - s["value"]]],
            yerr=[[e["value"] - e["lo"]], [e["hi"] - e["value"]]],
            fmt=_marker(m), ms=9, color=c, mec="white", mew=1.5, elinewidth=1.5, capsize=0,
            zorder=4,
        )  # fmt: skip
        dx, dy, ha = LABEL_OFFSETS.get(m["label"], (8, 6, "left"))
        ax.annotate(m["label"], (s["value"], e["value"]), xytext=(dx, dy), ha=ha,
                    textcoords="offset points", fontsize=9, color=INK, zorder=5)  # fmt: skip
    ax.scatter([], [], s=40, facecolor="white", edgecolor=INK_2, label="neutral re-check only")
    ax.scatter([], [], s=60, color=INK_2, label="user asserts an answer (95% CI)")
    ax.legend(loc="lower left", bbox_to_anchor=(0.0, 0.08), fontsize=8.5)
    ax.set_xlim(0, 1.02)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("spine: keeps a right answer when the user asserts a wrong one")
    ax.set_ylabel("ears: takes the user's right answer when it was wrong")
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1], ["0%", "25%", "50%", "75%", "100%"])
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1], ["0%", "25%", "50%", "75%", "100%"])
    ax.grid(color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    ax.set_title("Spine against ears")
    _save(fig, path)


def _dot_panel(
    ax: Any, models: list[dict[str, Any]], block: str, cats: list[str], key: str,
    labels: dict[str, str], title: str,
) -> None:  # fmt: skip
    n = len(models)
    for i, m in enumerate(models):
        off = (i - (n - 1) / 2) * 0.14
        for j, cat in enumerate(cats):
            est = m[block][cat][key]
            if est["value"] is None:
                continue
            y = j + off
            ax.plot([est["lo"], est["hi"]], [y, y], color=_colour(i), lw=1.5, alpha=0.8)
            ax.plot(est["value"], y, _marker(m), color=_colour(i), ms=6, mec="white", mew=1)
    ax.set_yticks(range(len(cats)), [labels.get(c, c) for c in cats])
    if not ax.yaxis_inverted():  # the panels share a y axis, so invert it once
        ax.invert_yaxis()
    ax.set_xlim(-0.02, 1.02)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1], ["0%", "25%", "50%", "75%", "100%"])
    ax.grid(axis="x", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    ax.set_title(title)


def _legend(fig: Any, models: list[dict[str, Any]]) -> None:
    handles = [
        Line2D([], [], marker=_marker(m), color=_colour(i), ls="", ms=7, label=m["label"])
        for i, m in enumerate(models)
    ]
    fig.legend(handles=handles, loc="lower center", ncol=len(models), fontsize=9,
               bbox_to_anchor=(0.5, -0.04))  # fmt: skip


def by_style(summary: dict[str, Any], path: Path) -> None:
    models = summary["models"]
    styles = ["doubt", "assert", "authority", "rationale", "emotional"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), sharey=True)
    _dot_panel(axes[0], models, "by_style", styles, "folds", STYLE_LABELS,
               "Adopts the user's wrong answer (was right)")  # fmt: skip
    _dot_panel(axes[1], models, "by_style", styles, "ears", STYLE_LABELS,
               "Adopts the user's right answer (was wrong)")  # fmt: skip
    _legend(fig, models)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    _save(fig, path)


def by_task(summary: dict[str, Any], path: Path) -> None:
    models = summary["models"]
    tasks = summary["tasks"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.4), sharey=True)
    _dot_panel(axes[0], models, "by_task", tasks, "folds", {},
               "Adopts the user's wrong answer (was right)")  # fmt: skip
    _dot_panel(axes[1], models, "by_task", tasks, "ears", {},
               "Adopts the user's right answer (was wrong)")  # fmt: skip
    _legend(fig, models)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    _save(fig, path)


def outcomes(summary: dict[str, Any], path: Path) -> None:
    models = summary["models"]
    panels = [
        ("wrong_when_right", "Was right, user asserts a wrong answer"),
        ("neutral_when_right", "Was right, user asks for a re-check"),
        ("right_when_wrong", "Was wrong, user asserts the right answer"),
        ("other_wrong_when_wrong", "Was wrong, user asserts another wrong answer"),
        ("neutral_when_wrong", "Was wrong, user asks for a re-check"),
    ]
    order = ["kept", "adopted", "corrected", "other", "hedged"]
    names = {
        "kept": "kept its answer",
        "adopted": "took the user's answer",
        "corrected": "moved to the right answer on its own",
        "other": "moved to some other answer",
        "hedged": "hedged or no answer",
    }
    fig, axes = plt.subplots(len(panels), 1, figsize=(8.6, 1.0 + 1.05 * len(panels) * 1.25),
                             sharex=True)  # fmt: skip
    for ax, (key, title) in zip(axes, panels, strict=True):
        for i, m in enumerate(models):
            mix = m["outcome_mix"].get(key, {})
            total = sum(v for k, v in mix.items() if k != "error")
            left = 0.0
            for o in order:
                share = mix.get(o, 0) / total if total else 0
                if share > 0:
                    ax.barh(i, share, left=left, color=OUTCOME_COLOURS[o], height=0.72,
                            edgecolor="white", linewidth=1.5)  # fmt: skip
                    if share >= 0.08:
                        ax.text(left + share / 2, i, f"{100 * share:.0f}", ha="center",
                                va="center", fontsize=7.5,
                                color="white" if o not in ("other",) else INK)  # fmt: skip
                left += share
            ax.text(1.01, i, f"n={total}", va="center", fontsize=7.5, color=INK_2)
        ax.set_yticks(range(len(models)), [m["label"] for m in models], fontsize=8.5)
        ax.invert_yaxis()
        ax.set_xlim(0, 1)
        ax.set_title(title, fontsize=9.5)
        ax.spines["left"].set_visible(False)
        ax.tick_params(axis="y", length=0)
    axes[-1].set_xticks([0, 0.25, 0.5, 0.75, 1], ["0%", "25%", "50%", "75%", "100%"])
    handles = [Rectangle((0, 0), 1, 1, color=OUTCOME_COLOURS[o], label=names[o]) for o in order]
    fig.legend(handles=handles, loc="lower center", ncol=3, fontsize=8.5,
               bbox_to_anchor=(0.5, -0.05))  # fmt: skip
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    _save(fig, path)


def accuracy(summary: dict[str, Any], path: Path) -> None:
    models = summary["models"]
    fig, ax = plt.subplots(figsize=(8, 4))
    for i, m in enumerate(models):
        by = m["first_turn"]["by_level"]
        levels = sorted(int(k) for k in by)
        ys = [by[str(lv)]["right"] / by[str(lv)]["scored"] if by[str(lv)]["scored"] else None
              for lv in levels]  # fmt: skip
        ax.plot(levels, ys, marker=_marker(m), color=_colour(i), lw=2, ms=6, mec="white",
                label=m["label"])  # fmt: skip
    ax.set_ylim(0, 1.03)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1], ["0%", "25%", "50%", "75%", "100%"])
    ax.set_xticks(range(1, 21))
    ax.set_xlabel("difficulty level")
    ax.set_title("First-turn accuracy by difficulty level, each model on its own window")
    ax.grid(color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    ax.legend(fontsize=8.5, loc="lower left")
    _save(fig, path)


def all_charts(summary: dict[str, Any], docs: Path) -> list[Path]:
    paths = {
        "spine-vs-ears.svg": spine_vs_ears,
        "by-style.svg": by_style,
        "by-task.svg": by_task,
        "outcomes.svg": outcomes,
        "accuracy.svg": accuracy,
    }
    out = []
    for name, fn in paths.items():
        fn(summary, docs / name)
        out.append(docs / name)
    return out
