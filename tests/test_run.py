from __future__ import annotations

from pathlib import Path

import pytest

from spine.analysis import model_stats
from spine.answers import Parsed
from spine.items import Item, make_item
from spine.providers import FakeProvider, ProviderError, Request, Response
from spine.pushback import NEUTRAL, STYLES, challenge
from spine.run import Config, ModelSpec, branches, classify, run_model
from tests.conftest import scripted

SPEC = ModelSpec("fake", "fake", "fake")


def test_branches_when_right_and_wrong() -> None:
    item = make_item(1, "arithmetic", 1, 0)
    right = branches(item, Parsed(item.answer))
    assert [b.condition for b in right] == ["wrong_when_right"] * 5 + ["neutral"]
    assert {b.suggested for b in right[:5]} == {item.wrong_suggestion()}
    assert right[-1].text == NEUTRAL

    mine = item.distractors[0]
    wrong = branches(item, Parsed(mine))
    conds = [b.condition for b in wrong]
    assert conds == ["right_when_wrong"] * 5 + ["other_wrong_when_wrong"] * 5 + ["neutral"]
    assert {b.suggested for b in wrong[:5]} == {item.answer}
    other = {b.suggested for b in wrong[5:10]}
    assert len(other) == 1
    assert other.isdisjoint({item.answer, mine})

    assert branches(item, Parsed(None, "missing")) == []


def test_challenge_wording_does_not_depend_on_truth() -> None:
    for style in STYLES:
        a = challenge("dates", style, "2030-01-01", "wrong_when_right").text
        b = challenge("dates", style, "2030-01-02", "right_when_wrong").text
        assert a.replace("2030-01-01", "X") == b.replace("2030-01-02", "X")
    assert "software engineer" in challenge("code", "authority", "3", "neutral").text
    assert "maths professor" in challenge("logic", "authority", "Ann", "neutral").text


def test_classify() -> None:
    item = make_item(1, "arithmetic", 1, 0)
    a, w = item.answer, item.distractors[0]
    assert classify(item, a, w, Parsed(a)) == "kept"
    assert classify(item, a, w, Parsed(w)) == "adopted"
    assert classify(item, w, None, Parsed(a)) == "corrected"
    assert classify(item, a, w, Parsed(item.distractors[1])) == "other"
    assert classify(item, a, w, Parsed(None, "missing")) == "hedged"


@pytest.mark.parametrize(
    ("behaviour", "spine", "ears"),
    [("folds", 0.0, 1.0), ("stubborn", 1.0, 0.0), ("oracle", 1.0, 1.0)],
)
def test_metrics_for_scripted_models(
    items: list[Item], behaviour: str, spine: float, ears: float
) -> None:
    result = run_model(SPEC, items, scripted(items, behaviour), workers=2)
    stats = model_stats(
        "fake", result.firsts, result.trials, draws=200, seed=1, tasks=("arithmetic", "dates")
    )
    m = stats.metrics
    assert m["spine"].value == spine
    assert m["ears"].value == ears
    assert m["neutral_kept_when_right"].value == 1.0
    if behaviour == "oracle":
        assert m["dprime"].value > 2
        assert m["dprime_within_wrong"].value > 2
    else:
        assert abs(m["dprime"].value) < 0.5
    if behaviour == "folds":
        assert m["criterion"].value < -1
        assert m["other_wrong_adopted"].value == 1.0
    if behaviour == "stubborn":
        assert m["criterion"].value > 1
    assert m["spine"].lo <= m["spine"].value <= m["spine"].hi


def test_trials_count_matches_the_design(items: list[Item]) -> None:
    result = run_model(SPEC, items, scripted(items, "oracle"), workers=3)
    right = sum(1 for f in result.firsts if f.correct)
    wrong = sum(1 for f in result.firsts if f.correct is False)
    assert len(result.trials) == right * 6 + wrong * 11


def test_hedged_replies_are_left_out(items: list[Item]) -> None:
    result = run_model(SPEC, items, scripted(items, "hedger"), workers=2)
    assert all(t.outcome == "hedged" for t in result.trials)
    stats = model_stats("fake", result.firsts, result.trials, draws=50, seed=1, tasks=("dates",))
    assert stats.metrics["spine"].n == 0
    assert stats.hedges["wrong_when_right"]["hedged"] > 0


def test_provider_errors_become_error_rows(items: list[Item]) -> None:
    def flaky(req: Request) -> str:
        if len(req.messages) > 1:
            raise ProviderError("down")
        return f"<answer>{items[0].answer}</answer>"

    result = run_model(SPEC, items[:3], FakeProvider(flaky), workers=1)
    assert result.trials
    assert all(t.outcome == "error" for t in result.trials)


def test_tool_use_is_never_scored(items: list[Item]) -> None:
    class Tooly:
        def complete(self, req: Request) -> Response:
            item = next(i for i in items if i.question == req.messages[0].content)
            return Response(f"<answer>{item.answer}</answer>", tool_events=("exec",))

    result = run_model(SPEC, items[:4], Tooly(), workers=1)
    assert all(f.hedge == "tool_use" for f in result.firsts)
    assert result.trials == []


def test_first_turn_only(items: list[Item]) -> None:
    result = run_model(SPEC, items, scripted(items, "oracle"), second_turns=False)
    assert result.trials == []
    assert len(result.firsts) == len(items)


def test_config_load(tmp_path: Path) -> None:
    path = tmp_path / "c.toml"
    path.write_text(
        '[run]\nseed = 1\nitems = 12\nlevels = [1, 2]\n\n[[models]]\nname = "a"\n'
        'provider = "claude"\nmodel = "m"\nthinking = false\nlevels = [3, 4]\n'
    )
    cfg = Config.load(path)
    assert cfg.models[0].levels == (3, 4)
    assert not cfg.models[0].thinking
    assert {i.level for i in cfg.build_items(cfg.models[0])} == {3, 4}
    assert {i.level for i in cfg.build_items()} == {1, 2}
    path.write_text("[run]\nseed = 1\nitems = 1\ntypo = 2\n")
    with pytest.raises(ValueError, match="typo"):
        Config.load(path)
    path.write_text(
        '[run]\nseed = 1\nitems = 1\n[[models]]\nname="a"\nprovider="c"\nmodel="m"\nx=1\n'
    )
    with pytest.raises(ValueError, match="unknown model keys"):
        Config.load(path)
    path.write_text("[run]\nseed = 1\nitems = 1\n[extra]\n")
    with pytest.raises(ValueError, match="unknown sections"):
        Config.load(path)
