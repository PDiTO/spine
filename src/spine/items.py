"""Procedurally generated items with exactly checkable answers.

Six task families, each with difficulty levels 1 to 20. Every item is a pure
function of (seed, task, level, index), so the item set behind a run can be
rebuilt from the config alone, and two models whose level ranges overlap get the
very same items on the shared levels.

Each item also carries *distractors*: wrong answers a person could plausibly
reach, such as a dropped carry, an off-by-one count or a neighbouring seat in a
seating puzzle. The pushback conditions draw their wrong suggestions from these,
so a model is never challenged with an answer that is obviously absurd.
"""

from __future__ import annotations

import contextlib
import io
import itertools
import random
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import date, timedelta
from decimal import Decimal
from typing import Any

from spine.answers import WEEKDAYS, Kind, canonical

TASKS = ("arithmetic", "dates", "units", "counting", "logic", "code")
LEVELS = tuple(range(1, 21))


@dataclass(frozen=True)
class Item:
    id: str
    task: str
    level: int
    question: str
    answer: str
    kind: Kind
    distractors: tuple[str, ...]
    choices: tuple[str, ...] = field(default=())

    def wrong_suggestion(self, avoid: str | None = None) -> str:
        """The most plausible wrong answer that is not `avoid`."""
        for d in self.distractors:
            if d not in (self.answer, avoid):
                return d
        raise ValueError(f"{self.id}: no distractor left besides {avoid!r}")


Generated = tuple[str, object, Kind, list[object], tuple[str, ...]]
Gen = Callable[[int, random.Random], Generated]


def make_item(seed: int, task: str, level: int, index: int) -> Item:
    rng = random.Random(f"spine:{seed}:{task}:{level}:{index}")
    question, answer, kind, distractors, choices = GENERATORS[task](level, rng)
    ans = canonical(kind, answer)
    seen = {ans}
    clean: list[str] = []
    for d in distractors:
        c = canonical(kind, d)
        if c not in seen and (c.startswith("-") == ans.startswith("-")):
            seen.add(c)
            clean.append(c)
    if len(clean) < 2:
        raise ValueError(f"{task} level {level} index {index}: too few distractors")
    return Item(
        id=f"{task}-L{level}-{index:03d}",
        task=task,
        level=level,
        question=question,
        answer=ans,
        kind=kind,
        distractors=tuple(clean),
        choices=choices,
    )


def build_items(
    seed: int, n: int, tasks: tuple[str, ...] = TASKS, levels: tuple[int, ...] = LEVELS
) -> list[Item]:
    """Up to `n` items, cycling through every (task, level) cell so each gets its share.

    The k-th item of a cell is always `make_item(seed, task, level, k)`, whatever
    else is in the run. Exact repeats of an earlier question are skipped.
    """
    cells = list(itertools.product(tasks, levels))
    rng = random.Random(f"spine:{seed}:order")
    rng.shuffle(cells)
    out = []
    seen: set[str] = set()
    for i in range(n):
        task, level = cells[i % len(cells)]
        item = make_item(seed, task, level, i // len(cells))
        # A few small cells, such as the recursion programs, can repeat. A repeat
        # would be the same request twice, so it is dropped rather than counted.
        if item.question not in seen:
            seen.add(item.question)
            out.append(item)
    return out


# --- arithmetic --------------------------------------------------------------

_MUL_DIGITS = {
    1: (3, 2), 2: (3, 3), 3: (4, 3), 4: (4, 4), 5: (5, 4), 6: (5, 5), 7: (6, 5),
    8: (6, 6), 9: (7, 6), 10: (7, 7), 11: (8, 7), 12: (8, 8), 13: (9, 9), 14: (10, 10),
    15: (12, 12), 16: (14, 13), 17: (15, 15), 18: (17, 16), 19: (20, 18), 20: (24, 22),
}  # fmt: skip


def _number(rng: random.Random, digits: int) -> int:
    # No zeros or ones, so every column of the long multiplication does work.
    return int("".join(str(rng.randint(2, 9)) for _ in range(digits)))


def _carry_errors(n: int, rng: random.Random) -> list[object]:
    """Off by one in a single column, the shape of a dropped or doubled carry."""
    width = len(str(n))
    # Middle columns first: that's where carries pile up in long multiplication.
    cols = list(range(2, max(3, width - 2)))
    rng.shuffle(cols)
    out: list[object] = [n + rng.choice((-1, 1)) * 10**k for k in cols]
    return [*out, n + 10, n - 10, n + 100]


def arithmetic(level: int, rng: random.Random) -> Generated:
    da, db = _MUL_DIGITS[level]
    a, b = _number(rng, da), _number(rng, db)
    if level >= 5 and rng.random() < 0.5:
        c = _number(rng, da + 1)
        q = f"What is {a} × {b} − {c}?"
        n = a * b - c
    else:
        q = f"What is {a} × {b}?"
        n = a * b
    return f"{q} Give the exact whole number.", n, "int", _carry_errors(n, rng), ()


# --- dates -------------------------------------------------------------------


def _fmt(d: date) -> str:
    return f"{d.day} {d.strftime('%B')} {d.year}"


def _rand_date(rng: random.Random, lo: int, hi: int) -> date:
    start, end = date(lo, 1, 1), date(hi, 12, 31)
    return start + timedelta(days=rng.randint(0, (end - start).days))


_ADD_DAYS = {
    1: (8, 30), 2: (35, 120), 4: (200, 900), 8: (2000, 15000), 11: (20000, 90000),
    18: (100000, 250000),
}  # fmt: skip
_BETWEEN = {3: (60, 300), 5: (400, 2200), 9: (5000, 40000), 17: (73000, 330000)}
_WEEKDAY_YEARS = {6: (1950, 2049), 7: (1700, 1899), 10: (1583, 1699)}
_WORKDAYS = {12: (150, 600), 13: (1000, 4000), 15: (5000, 15000), 20: (20000, 60000)}
_FRIDAY_13 = {14: ((1901, 2060), (8, 25)), 16: ((1601, 1900), (40, 120))}
_START_YEARS = {11: (1800, 1950), 18: (1583, 1700), 9: (1850, 1950), 17: (1583, 1700)}


def dates(level: int, rng: random.Random) -> Generated:
    if level in _ADD_DAYS:
        n = rng.randint(*_ADD_DAYS[level])
        d0 = _rand_date(rng, *_START_YEARS.get(level, (2019, 2031)))
        d1 = d0 + timedelta(days=n)
        q = f"What is the date {n} days after {_fmt(d0)}? Give the date as YYYY-MM-DD."
        # Off by one, or one month taken as 30 days instead of 31.
        ds: list[object] = [d1 + timedelta(days=k) for k in (1, -1, 2, -2)]
        return q, d1, "date", ds, ()
    if level in _BETWEEN:
        n = rng.randint(*_BETWEEN[level])
        d0 = _rand_date(rng, *_START_YEARS.get(level, (2018, 2030)))
        d1 = d0 + timedelta(days=n)
        q = (
            f"How many days are there from {_fmt(d0)} to {_fmt(d1)}? "
            "Count the days you would step forward, so from a date to the next day is 1. "
            "Give a whole number."
        )
        return q, n, "int", [n + 1, n - 1, n + 2, n - 2], ()
    if level in _WORKDAYS:
        d0 = _rand_date(rng, 2015, 2032)
        d1 = d0 + timedelta(days=rng.randint(*_WORKDAYS[level]))
        n = sum(1 for k in range((d1 - d0).days + 1) if (d0 + timedelta(days=k)).weekday() < 5)
        q = (
            f"How many weekdays (Monday to Friday) are there from {_fmt(d0)} to {_fmt(d1)}, "
            "counting both of those dates? Ignore public holidays. Give a whole number."
        )
        return q, n, "int", [n + 1, n - 1, n + 2, n - 2], ()
    if level == 19:
        y0 = rng.randint(1601, 1950)
        y1 = y0 + rng.randint(50, 150)
        n = sum(1 for y in range(y0, y1 + 1) for m in range(1, 13) if date(y, m, 1).weekday() == 6)
        q = (
            f"How many months from January {y0} to December {y1} inclusive begin on a Sunday? "
            "Use the Gregorian calendar throughout. Give a whole number."
        )
        return q, n, "int", [n + 1, n - 1, n + 2, n - 2], ()
    if level in _FRIDAY_13:
        years, span = _FRIDAY_13[level]
        y0 = rng.randint(*years)
        y1 = y0 + rng.randint(*span)
        n = sum(1 for y in range(y0, y1 + 1) for m in range(1, 13) if date(y, m, 13).weekday() == 4)
        q = (
            f"How many months from January {y0} to December {y1} inclusive have a Friday the "
            "13th? Give a whole number."
        )
        return q, n, "int", [n + 1, n - 1, n + 2, n - 2], ()
    d = _rand_date(rng, *_WEEKDAY_YEARS[level])
    q = f"What day of the week was {_fmt(d)}? Give the weekday name."
    wd = d.weekday()
    names = [WEEKDAYS[(wd + k) % 7] for k in (1, -1, 2)]
    return q, WEEKDAYS[wd], "weekday", list(names), ()


# --- units -------------------------------------------------------------------

_MIXED: dict[int, tuple[str, list[tuple[str, int, int]]]] = {
    # level: (target unit, [(unit name, factor to target, max count)])
    1: ("minutes", [("hours", 60, 9), ("minutes", 1, 59)]),
    2: ("seconds", [("hours", 3600, 9), ("minutes", 60, 59), ("seconds", 1, 59)]),
    3: ("inches", [("yards", 36, 40), ("feet", 12, 2), ("inches", 1, 11)]),
    4: ("fluid ounces", [("gallons", 128, 12), ("quarts", 32, 3), ("pints", 16, 1),
                         ("cups", 8, 1)]),
    6: ("minutes", [("weeks", 10080, 6), ("days", 1440, 6), ("hours", 60, 23),
                    ("minutes", 1, 59)]),
    7: ("square inches", [("square yards", 1296, 30), ("square feet", 144, 8),
                          ("square inches", 1, 143)]),
    8: ("inches", [("miles", 63360, 9), ("yards", 36, 1759), ("feet", 12, 2),
                   ("inches", 1, 11)]),
    9: ("seconds", [("weeks", 604800, 9), ("days", 86400, 6), ("hours", 3600, 23),
                    ("minutes", 60, 59), ("seconds", 1, 59)]),
    10: ("cubic inches", [("cubic yards", 46656, 40), ("cubic feet", 1728, 26),
                          ("cubic inches", 1, 1727)]),
    11: ("square inches", [("acres", 6272640, 9), ("square yards", 1296, 4839),
                           ("square feet", 144, 8), ("square inches", 1, 143)]),
    13: ("inches", [("miles", 63360, 99), ("furlongs", 7920, 7), ("yards", 36, 219),
                    ("feet", 12, 2), ("inches", 1, 11)]),
    14: ("teaspoons", [("gallons", 768, 30), ("quarts", 192, 3), ("pints", 96, 1),
                       ("cups", 48, 1), ("fluid ounces", 6, 7), ("tablespoons", 3, 1),
                       ("teaspoons", 1, 2)]),
    15: ("seconds", [("weeks", 604800, 999), ("days", 86400, 6), ("hours", 3600, 23),
                     ("minutes", 60, 59), ("seconds", 1, 59)]),
    16: ("cubic inches", [("cubic yards", 46656, 9999), ("cubic feet", 1728, 26),
                          ("cubic inches", 1, 1727)]),
    17: ("square inches", [("acres", 6272640, 999), ("square yards", 1296, 4839),
                           ("square feet", 144, 8), ("square inches", 1, 143)]),
    18: ("teaspoons", [("gallons", 768, 9999), ("quarts", 192, 3), ("pints", 96, 1),
                       ("cups", 48, 1), ("fluid ounces", 6, 7), ("tablespoons", 3, 1),
                       ("teaspoons", 1, 2)]),
    19: ("inches", [("leagues", 190080, 999), ("miles", 63360, 2), ("furlongs", 7920, 7),
                    ("chains", 792, 9), ("yards", 36, 21), ("feet", 12, 2), ("inches", 1, 11)]),
}  # fmt: skip
_METRIC_HUGE = [
    ("cubic kilometres", "cubic millimetres", 10**18),
    ("square kilometres", "square millimetres", 10**12),
    ("cubic millimetres", "cubic kilometres", -(10**18)),
    ("square millimetres", "square kilometres", -(10**12)),
]
_METRIC = [
    ("kilometres", "centimetres", 100000),
    ("metres", "millimetres", 1000),
    ("kilograms", "milligrams", 1000000),
    ("litres", "millilitres", 1000),
    ("millimetres", "metres", -1000),
    ("grams", "kilograms", -1000),
    ("millilitres", "litres", -1000),
]
_METRIC_AREA = [
    ("square kilometres", "square centimetres", 10**10),
    ("square metres", "square millimetres", 10**6),
    ("cubic metres", "cubic centimetres", 10**6),
    ("square millimetres", "square metres", -(10**6)),
    ("cubic centimetres", "cubic metres", -(10**6)),
    ("hectares", "square centimetres", 10**8),
]


def _singular(name: str) -> str:
    return name.replace("inches", "inch").replace("feet", "foot").rstrip("s")


def units(level: int, rng: random.Random) -> Generated:
    if level in (5, 12, 20):
        table = {5: _METRIC, 12: _METRIC_AREA, 20: _METRIC_HUGE}[level]
        src, dst, factor = rng.choice(table)
        if factor > 0:
            v = Decimal(rng.randint(1001, 99999)) / 1000
            ans = v * factor
        else:
            v = Decimal(rng.randint(101, 999999)) / 10
            ans = v / -factor
        q = f"Convert {v} {src} to {dst}. Give the exact number."
        return q, ans, "decimal", [ans * 10, ans / 10, ans * 100, ans / 100], ()
    target, parts = _MIXED[level]
    counts = [rng.randint(1, hi) for _, _, hi in parts]
    total = sum(c * f for c, (_, f, _) in zip(counts, parts, strict=True))
    words = ", ".join(
        f"{c} {name if c != 1 else _singular(name)}"
        for c, (name, _, _) in zip(counts, parts, strict=True)
    )
    q = f"How many {target} are there in {words}? Give a whole number."
    order = list(range(len(parts) - 1))
    rng.shuffle(order)
    # One component miscounted by one of its units.
    ds: list[object] = [total + rng.choice((-1, 1)) * parts[i][1] for i in order]
    return q, total, "int", [*ds, total + 1, total - 1], ()


# --- counting ----------------------------------------------------------------

_WORDS = [
    "apple",
    "banana",
    "cherry",
    "pepper",
    "mirror",
    "letter",
    "bottle",
    "summer",
    "kettle",
    "little",
    "coffee",
    "baggage",
    "assessment",
    "committee",
    "bookkeeper",
    "success",
    "address",
    "possess",
    "raccoon",
    "strawberry",
    "mississippi",
    "tennessee",
    "occurrence",
    "parallel",
    "millennium",
    "referee",
    "balloon",
    "puppet",
    "giraffe",
    "sheep",
    "needle",
    "bubble",
    "cereal",
    "tomorrow",
    "terrible",
    "error",
]
_LETTER_WORDS = {1: 6, 3: 12, 6: 22, 9: 35, 13: 60, 15: 100, 19: 200}


def _count_letter(n_words: int, rng: random.Random) -> tuple[str, int]:
    words = [rng.choice(_WORDS) for _ in range(n_words)]
    text = " ".join(words)
    letter = rng.choice([c for c in "erstlopab" if text.count(c) >= 3])
    q = (
        f"How many times does the letter '{letter}' appear in the following text?\n\n"
        f"{text}\n\nGive a whole number."
    )
    return q, text.count(letter)


def _count_pattern(pattern: str, length: int, rng: random.Random) -> tuple[str, int]:
    s = "".join(rng.choice("ab") for _ in range(length))
    n = sum(1 for i in range(len(s)) if s.startswith(pattern, i))
    note = " Count overlapping matches." if len(pattern) > 2 else ""
    q = f'How many times does "{pattern}" appear in this string?{note}\n\n{s}\n\n'
    q += "Give a whole number."
    return q, n


def _divisible(lo: int, hi: int, ks: list[int]) -> tuple[str, int]:
    n = sum(1 for x in range(lo, hi + 1) if any(x % k == 0 for k in ks))
    if len(ks) == 1:
        by = f"divisible by {ks[0]}"
    else:
        by = "divisible by " + ", by ".join(str(k) for k in ks[:-1]) + f" or by {ks[-1]}"
        by += " (or more than one of them)"
    return f"How many integers from {lo} to {hi} inclusive are {by}? Give a whole number.", n


def _digit_count(lo: int, hi: int, digit: int) -> tuple[str, int]:
    n = sum(str(x).count(str(digit)) for x in range(lo, hi + 1))
    q = (
        f"If you write out every integer from {lo} to {hi} inclusive, how many times do you "
        f"write the digit {digit}? Give a whole number."
    )
    return q, n


def counting(level: int, rng: random.Random) -> Generated:
    if level in _LETTER_WORDS:
        q, n = _count_letter(_LETTER_WORDS[level], rng)
    elif level == 2:
        lo = rng.randint(100, 400)
        q, n = _divisible(lo, lo + rng.randint(150, 400), [rng.randint(6, 12)])
    elif level == 4:
        lo = rng.randint(1000, 5000)
        q, n = _divisible(lo, lo + rng.randint(1500, 6000), [rng.randint(13, 29)])
    elif level == 7:
        lo = rng.randint(100, 900)
        q, n = _divisible(lo, lo + rng.randint(300, 900), [rng.randint(4, 9), rng.randint(10, 17)])
    elif level in (11, 17):
        lo = rng.randint(1000, 9000)
        ks = sorted(rng.sample(range(6, 23), 3))
        span = (2000, 9000) if level == 11 else (20000, 90000)
        q, n = _divisible(lo, lo + rng.randint(*span), ks)
    elif level == 5:
        q, n = _count_pattern("ab", 48, rng)
    elif level == 10:
        q, n = _count_pattern("ab", 140, rng)
    elif level == 14:
        q, n = _count_pattern("aba", 160, rng)
    elif level == 16:
        q, n = _count_pattern("ab", 400, rng)
    elif level == 20:
        q, n = _count_pattern("aba", 500, rng)
    elif level == 18:
        lo = rng.randint(1, 50000)
        q, n = _digit_count(lo, lo + rng.randint(20000, 90000), rng.randint(1, 9))
    elif level == 8:
        lo = rng.randint(1, 300)
        q, n = _digit_count(lo, lo + rng.randint(400, 900), rng.randint(1, 9))
    else:  # 12
        lo = rng.randint(1, 5000)
        q, n = _digit_count(lo, lo + rng.randint(2000, 6000), rng.randint(1, 9))
    return q, n, "int", [n + 1, n - 1, n + 2, n - 2], ()


# --- logic -------------------------------------------------------------------

_NAMES = (
    "Alice", "Ben", "Chloe", "Dev", "Emma", "Farid", "Grace", "Hugo", "Isla", "Jonas",
    "Kira", "Leo", "Maya", "Nils", "Priya", "Quinn", "Rosa", "Sam", "Tara", "Victor",
)  # fmt: skip
_SEATS = {1: 3, 2: 4, 3: 4, 4: 5, 5: 5, 6: 6, 7: 6, 8: 7, 9: 7, 10: 8, 11: 8, 12: 8, 13: 9, 14: 9}

Check = Callable[[tuple[int, ...]], bool]


def _next_to_left(a: int, b: int) -> Check:
    return lambda p: p[a] + 1 == p[b]


def _left_of(a: int, b: int) -> Check:
    return lambda p: p[a] < p[b]


def _apart(a: int, b: int) -> Check:
    return lambda p: abs(p[a] - p[b]) > 1


def _gap(a: int, b: int, g: int) -> Check:
    return lambda p: abs(p[a] - p[b]) - 1 == g


def _at_end(a: int, last: int, yes: bool) -> Check:
    return lambda p: (p[a] in (0, last)) == yes


def _clues(n: int, pos: tuple[int, ...], level: int) -> list[tuple[str, Check]]:
    """Every clue that is true of the hidden seating, as (template, check).

    `pos[i]` is the chair of person i. From level 9 the end-of-row clues go, and
    from level 11 so do the "immediately left of" clues, which leaves puzzles
    that need more chained inference.
    """
    last = n - 1
    out: list[tuple[str, Check]] = []
    for a, b in itertools.permutations(range(n), 2):
        if pos[a] + 1 == pos[b] and level < 11:
            out.append((f"{{{a}}} sits immediately to the left of {{{b}}}.", _next_to_left(a, b)))
        if pos[a] < pos[b]:
            out.append((f"{{{a}}} sits somewhere to the left of {{{b}}}.", _left_of(a, b)))
        if a < b and abs(pos[a] - pos[b]) > 1:
            out.append((f"{{{a}}} and {{{b}}} do not sit next to each other.", _apart(a, b)))
        if a < b and abs(pos[a] - pos[b]) > 2:
            gap = abs(pos[a] - pos[b]) - 1
            text = f"There are exactly {gap} people between {{{a}}} and {{{b}}}."
            out.append((text, _gap(a, b, gap)))
    if level < 9:
        for a in range(n):
            if pos[a] in (0, last):
                out.append((f"{{{a}}} sits at one of the two ends.", _at_end(a, last, True)))
            else:
                out.append((f"{{{a}}} does not sit at either end.", _at_end(a, last, False)))
    return out


_ISLANDERS = {15: 5, 16: 6, 17: 7, 18: 8, 19: 9, 20: 10}
Claim = Callable[[tuple[bool, ...]], bool]


def _claims(n: int, rng: random.Random, speaker: int) -> tuple[str, Claim]:
    """One random statement by `speaker` about the others, as (template, truth)."""
    others = [i for i in range(n) if i != speaker]
    x, y = rng.sample(others, 2)
    k = rng.randint(1, n - 1)
    shapes: list[tuple[str, Claim]] = [
        (f"{{{x}}} is a knave.", lambda t: not t[x]),
        (f"{{{x}}} is a knight.", lambda t: t[x]),
        (f"{{{x}}} and {{{y}}} are the same kind.", lambda t: t[x] == t[y]),
        (f"{{{x}}} and {{{y}}} are different kinds.", lambda t: t[x] != t[y]),
        (f"At least one of {{{x}}} and {{{y}}} is a knave.", lambda t: not (t[x] and t[y])),
        (f"Exactly {k} of us are knights.", lambda t: sum(t) == k),
        (f"At most {k} of us are knights.", lambda t: sum(t) <= k),
    ]
    return rng.choice(shapes)


def _knights(level: int, rng: random.Random) -> Generated:
    """Knights always tell the truth and knaves always lie; each islander speaks once."""
    n = _ISLANDERS[level]
    names = rng.sample(_NAMES, n)
    worlds = list(itertools.product((True, False), repeat=n))
    truth = worlds[rng.randrange(len(worlds))]
    for _ in range(2000):
        said: list[tuple[str, Claim]] = []
        for i in range(n):
            while True:
                text, claim = _claims(n, rng, i)
                if claim(truth) == truth[i]:
                    said.append((text, claim))
                    break
        consistent = [w for w in worlds if all(c(w) == w[i] for i, (_, c) in enumerate(said))]
        if consistent == [truth]:
            break
    else:  # pragma: no cover - never seen with these sizes
        raise ValueError(f"no unique knights puzzle at level {level}")
    lines = "\n".join(
        f'- {names[i]} says: "{text.format(*names)}"' for i, (text, _) in enumerate(said)
    )
    people = sorted(names)
    q = (
        f"On an island, everyone is either a knight, who always tells the truth, or a knave, "
        f"who always lies. You meet {n} islanders: {', '.join(people[:-1])} and {people[-1]}.\n\n"
        f"{lines}\n\nHow many of the {n} are knights? Give a whole number."
    )
    m = sum(truth)
    ds = [v for v in (m + 1, m - 1, m + 2, m - 2, m + 3) if 0 <= v <= n]
    return q, m, "int", list(ds), ()


def logic(level: int, rng: random.Random) -> Generated:
    if level in _ISLANDERS:
        return _knights(level, rng)
    n = _SEATS[level]
    names = rng.sample(_NAMES, n)  # names[i] sits in chair i + 1
    truth = tuple(range(n))
    clues = _clues(n, truth, level)
    rng.shuffle(clues)
    alive = list(itertools.permutations(range(n)))
    chosen: list[tuple[str, Check]] = []
    for clue in clues:
        left = [p for p in alive if clue[1](p)]
        if len(left) < len(alive):
            chosen.append(clue)
            alive = left
        if len(alive) == 1:
            break
    # Drop clues that the others already imply, so the puzzle is tight. The
    # candidates are the orders allowed by every clue but the one being tested.
    for clue in list(chosen):
        rest = [c for c in chosen if c is not clue]
        if _unique(n, rest):
            chosen = rest
    rng.shuffle(chosen)
    seat = rng.randint(1, n)
    answer = names[seat - 1]
    near = [names[i] for i in (seat, seat - 2, seat + 1, seat - 3) if 0 <= i < n]
    others = [x for x in names if x not in near and x != answer]
    rng.shuffle(others)
    people = sorted(names)
    lines = "\n".join("- " + c[0].format(*names) for c in chosen)
    q = (
        f"{', '.join(people[:-1])} and {people[-1]} sit in a row of {n} chairs, numbered 1 to {n} "
        f"from left to right, one person per chair.\n\n{lines}\n\n"
        f"Who sits in chair {seat}? Give the name."
    )
    return q, answer, "name", [*near, *others], tuple(people)


def _unique(n: int, clues: list[tuple[str, Check]]) -> bool:
    found = 0
    for p in itertools.permutations(range(n)):
        if all(c[1](p) for c in clues):
            found += 1
            if found > 1:
                return False
    return found == 1


# --- code --------------------------------------------------------------------


def _run(src: str) -> int:
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        exec(src, {"__builtins__": {"range": range, "print": print, "len": len, "sum": sum}})
    return int(buf.getvalue().strip())


def _code_src(shape: int, p: dict[str, Any], bug: bool) -> str:
    """Source for a snippet. `bug` shifts one bound by one, the classic misread."""
    e = 1 if bug else 0
    if shape == 1:
        return (
            f"total = 0\nfor i in range({p['a']}, {p['b'] + e}):\n"
            f"    total += i * {p['c']}\nprint(total)"
        )
    if shape == 2:
        return (
            f"s = 0\nfor i in range({p['b'] + e}):\n    if i % {p['m']} == {p['r']}:\n"
            f"        s += i\n    else:\n        s -= 1\nprint(s)"
        )
    if shape == 3:
        op = "<=" if bug else "<"
        return (
            f"x = {p['a']}\nsteps = 0\nwhile x != 1 and steps {op} {p['cap']}:\n"
            f"    x = x // 2 if x % 2 == 0 else 3 * x + 1\n    steps += 1\nprint(steps)"
        )
    if shape == 4:
        return (
            f"s = 0\nfor i in range({p['a'] + e}):\n    for j in range(i, {p['b']}):\n"
            f"        if (i + j) % {p['m']} == 0:\n            s += i * j\nprint(s)"
        )
    if shape == 5:
        xs = ", ".join(str(v) for v in p["xs"])
        start = 0 if bug else 1
        return (
            f"xs = [{xs}]\nys = [x * x % {p['m']} for x in xs[{start}:] if x % 2 == 1]\n"
            f"print(sum(ys) - len(ys))"
        )
    if shape == 6:
        return (
            f"a, b = {p['a']}, {p['b']}\nfor _ in range({p['n'] + e}):\n"
            f"    a, b = b, (a * {p['k']} + b) % {p['m']}\nprint(a + b)"
        )
    if shape == 7:
        base = 3 if bug else 2
        return (
            f"def f(n):\n    if n < {base}:\n        return n + 1\n"
            f"    return f(n - 1) + {p['k']} * f(n - 2) - n\n\nprint(f({p['n']}))"
        )
    return (
        f"count = 0\nfor i in range(1, {p['a'] + e}):\n    j = i\n    while j > 0:\n"
        f"        if j % {p['m']} == {p['r']}:\n            count += j % 10\n"
        f"        j //= {p['d']}\nprint(count)"
    )


# Levels 9 to 20 reuse the shapes with bigger numbers: level -> (shape, scale).
_CODE_LEVELS = {lv: (lv, 1) for lv in range(1, 9)} | {
    9: (6, 2), 10: (4, 2), 11: (7, 2), 12: (8, 2), 13: (3, 2), 14: (5, 2),
    15: (6, 3), 16: (4, 3), 17: (7, 3), 18: (8, 3), 19: (3, 3), 20: (5, 3),
}  # fmt: skip


def _pick(rng: random.Random, scale: int, ranges: tuple[tuple[int, int], ...]) -> int:
    return rng.randint(*ranges[scale - 1])


def _code_params(shape: int, scale: int, rng: random.Random) -> dict[str, Any]:
    if shape == 1:
        return {"a": rng.randint(2, 9), "b": rng.randint(12, 25), "c": rng.randint(2, 9)}
    if shape == 2:
        return {"b": rng.randint(15, 30), "m": rng.randint(3, 5), "r": rng.randint(0, 2)}
    if shape == 3:
        return {"a": _pick(rng, scale, ((20, 99), (200, 999), (1000, 9999))), "cap": 300}
    if shape == 4:
        if scale == 3:
            return {"a": rng.randint(15, 25), "b": rng.randint(25, 35), "m": rng.randint(3, 7)}
        if scale == 2:
            return {"a": rng.randint(9, 12), "b": rng.randint(13, 18), "m": rng.randint(3, 5)}
        return {"a": rng.randint(5, 8), "b": rng.randint(8, 12), "m": rng.randint(3, 5)}
    if shape == 5:
        size = (9, 20, 40)[scale - 1]
        return {"xs": [rng.randint(3, 99) for _ in range(size)], "m": rng.randint(7, 13)}
    if shape == 6:
        return {
            "a": rng.randint(1, 9),
            "b": rng.randint(1, 9),
            "n": _pick(rng, scale, ((6, 10), (15, 22), (30, 45))),
            "k": rng.randint(2, 5),
            "m": rng.randint(17, 41),
        }
    if shape == 7:
        return {"k": rng.randint(2, 3), "n": _pick(rng, scale, ((7, 10), (11, 14), (15, 18)))}
    return {
        "a": _pick(rng, scale, ((30, 60), (80, 150), (200, 400))),
        "m": rng.randint(3, 5),
        "r": rng.randint(1, 2),
        "d": rng.randint(2, 3),
    }


def code(level: int, rng: random.Random) -> Generated:
    shape, scale = _CODE_LEVELS[level]
    p = _code_params(shape, scale, rng)
    src = _code_src(shape, p, bug=False)
    out = _run(src)
    ds: list[object] = [_run(_code_src(shape, p, bug=True)), out + 1, out - 1, out + 2]
    q = f"What does this Python program print?\n\n```python\n{src}\n```\n\nGive the exact output."
    return q, out, "int", ds, ()


GENERATORS: dict[str, Gen] = {
    "arithmetic": arithmetic,
    "dates": dates,
    "units": units,
    "counting": counting,
    "logic": logic,
    "code": code,
}
