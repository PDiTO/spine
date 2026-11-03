from __future__ import annotations

import itertools
import re
from datetime import date, datetime, timedelta

import pytest

from spine.items import LEVELS, TASKS, _run, build_items, make_item


@pytest.mark.parametrize(("task", "level"), list(itertools.product(TASKS, LEVELS)))
def test_every_cell_builds_with_plausible_distractors(task: str, level: int) -> None:
    item = make_item(3, task, level, 0)
    assert item.answer not in item.distractors
    assert len(item.distractors) >= 2
    assert len(set(item.distractors)) == len(item.distractors)
    assert item.wrong_suggestion() != item.answer
    other = item.wrong_suggestion(avoid=item.distractors[0])
    assert other not in (item.answer, item.distractors[0])


def test_items_are_a_pure_function_of_the_seed() -> None:
    assert build_items(5, 30) == build_items(5, 30)
    assert build_items(5, 30) != build_items(6, 30)


def test_overlapping_windows_share_items() -> None:
    a = {i.id: i for i in build_items(5, 6 * 4 * 2, levels=(3, 4, 5, 6))}
    b = {i.id: i for i in build_items(5, 6 * 4 * 2, levels=(5, 6, 7, 8))}
    shared = set(a) & set(b)
    assert len(shared) == 6 * 2 * 2
    assert all(a[k] == b[k] for k in shared)


def test_every_cell_gets_its_share() -> None:
    items = build_items(1, len(TASKS) * 3 * 2, levels=(1, 2, 3))
    cells = {(i.task, i.level) for i in items}
    assert len(cells) == len(TASKS) * 3


@pytest.mark.parametrize("level", LEVELS)
def test_arithmetic_answers(level: int) -> None:
    item = make_item(11, "arithmetic", level, 0)
    expr = re.search(r"What is (.*)\?", item.question)
    assert expr
    assert str(eval(expr.group(1).replace("×", "*").replace("−", "-"))) == item.answer


def _parse_date(text: str) -> date:
    return datetime.strptime(text, "%d %B %Y").date()


@pytest.mark.parametrize("level", [1, 2, 4, 8, 11])
def test_date_addition_answers(level: int) -> None:
    item = make_item(2, "dates", level, 1)
    m = re.search(r"date (\d+) days after (\d+ \w+ \d+)", item.question)
    assert m
    want = _parse_date(m.group(2)) + timedelta(days=int(m.group(1)))
    assert item.answer == want.isoformat()


@pytest.mark.parametrize("level", [6, 7, 10])
def test_weekday_answers(level: int) -> None:
    item = make_item(2, "dates", level, 0)
    m = re.search(r"week was (\d+ \w+ \d+)", item.question)
    assert m
    assert item.answer == _parse_date(m.group(1)).strftime("%A")


@pytest.mark.parametrize("level", LEVELS)
def test_code_answers_match_execution(level: int) -> None:
    item = make_item(4, "code", level, 0)
    src = item.question.split("```python\n")[1].split("\n```")[0]
    assert str(_run(src)) == item.answer


@pytest.mark.parametrize("level", [1, 4, 8, 10])
def test_logic_puzzles_have_one_solution(level: int) -> None:
    item = make_item(9, "logic", level, 0)
    people = list(item.choices)
    n = len(people)
    clues = [line[2:] for line in item.question.splitlines() if line.startswith("- ")]
    solutions = []
    for order in itertools.permutations(people):
        pos = {p: i for i, p in enumerate(order)}
        if all(_holds(c, pos, n) for c in clues):
            solutions.append(order)
    assert len(solutions) == 1
    seat = int(re.search(r"chair (\d+)\?", item.question).group(1))  # type: ignore[union-attr]
    assert solutions[0][seat - 1] == item.answer


def _holds(clue: str, pos: dict[str, int], n: int) -> bool:
    """An independent reading of the clue sentences."""
    if m := re.fullmatch(r"(\w+) sits immediately to the left of (\w+)\.", clue):
        return pos[m[1]] + 1 == pos[m[2]]
    if m := re.fullmatch(r"(\w+) sits somewhere to the left of (\w+)\.", clue):
        return pos[m[1]] < pos[m[2]]
    if m := re.fullmatch(r"(\w+) and (\w+) do not sit next to each other\.", clue):
        return abs(pos[m[1]] - pos[m[2]]) > 1
    if m := re.fullmatch(r"There are exactly (\d+) people between (\w+) and (\w+)\.", clue):
        return abs(pos[m[2]] - pos[m[3]]) - 1 == int(m[1])
    if m := re.fullmatch(r"(\w+) sits at one of the two ends\.", clue):
        return pos[m[1]] in (0, n - 1)
    if m := re.fullmatch(r"(\w+) does not sit at either end\.", clue):
        return pos[m[1]] not in (0, n - 1)
    raise AssertionError(f"unknown clue {clue!r}")


def test_counting_letter_answer() -> None:
    item = make_item(1, "counting", 1, 0)
    letter = re.search(r"letter '(\w)'", item.question).group(1)  # type: ignore[union-attr]
    text = item.question.split("\n\n")[1]
    assert str(text.count(letter)) == item.answer


def test_unit_distractors_are_one_component_off() -> None:
    item = make_item(1, "units", 3, 0)
    diffs = {abs(int(d) - int(item.answer)) for d in item.distractors}
    assert diffs <= {1, 12, 36}
