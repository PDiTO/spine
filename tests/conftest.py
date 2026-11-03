from __future__ import annotations

import random
import re
from collections.abc import Callable
from pathlib import Path

import pytest

from spine.items import Item, build_items
from spine.providers import FakeProvider, Request
from spine.run import Config, ModelSpec

ROOT = Path(__file__).resolve().parent.parent

Behaviour = Callable[[Item, str, str | None, bool], str]


def scripted(
    items: list[Item], behaviour: str, *, accuracy: float = 0.6, seed: int = 0
) -> FakeProvider:
    """A fake model with a fixed first-turn accuracy and a fixed way of taking pushback.

    behaviours:
    - "folds": always takes the user's answer
    - "stubborn": never changes its answer
    - "oracle": takes the user's answer only when it is right
    - "hedger": answers turn 1, then never gives a tag
    """
    by_question = {i.question: i for i in items}
    rng = random.Random(seed)
    right = {i.id: rng.random() < accuracy for i in items}

    def first(item: Item) -> str:
        if right[item.id]:
            return item.answer
        return item.distractors[-1]

    def policy(req: Request) -> str:
        item = by_question[req.messages[0].content]
        mine = first(item)
        if len(req.messages) == 1:
            return f"Quick answer.\n<answer>{mine}</answer>"
        push = req.messages[-1].content
        suggested = next((c for c in [item.answer, *item.distractors] if _mentions(push, c)), None)
        if behaviour == "hedger":
            return "It could be either."
        if suggested is None:  # neutral re-check
            return f"<answer>{mine}</answer>"
        if behaviour == "folds" or (behaviour == "oracle" and suggested == item.answer):
            return f"You're right.\n<answer>{suggested}</answer>"
        return f"I'll stick with it.\n<answer>{mine}</answer>"

    return FakeProvider(policy)


def _mentions(text: str, value: str) -> bool:
    return re.search(rf"(?<![\w-]){re.escape(value)}(?![\w])", text) is not None


@pytest.fixture
def config() -> Config:
    return Config(
        seed=7,
        items=24,
        models=(ModelSpec("fake-a", "fake", "fake-a", label="Fake A"),),
        levels=(1, 2),
        bootstrap=200,
    )


@pytest.fixture
def items(config: Config) -> list[Item]:
    return build_items(config.seed, config.items, config.tasks, config.levels)
