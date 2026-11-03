"""Answer kinds, canonical forms and the final-answer tag.

Every item has an answer of one kind. A reply counts only through its
`<answer>...</answer>` tag, and the tag's contents are normalised to the same
canonical string the generator produced, so comparing answers is string equality.

A reply that can't be read as exactly one answer is *hedged*, with a reason, and
is reported on its own rather than scored as right or wrong.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date, datetime
from decimal import Decimal, InvalidOperation
from typing import Literal

Kind = Literal["int", "decimal", "date", "weekday", "name"]

WEEKDAYS = ("Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday")

TAG = re.compile(r"<answer>(.*?)</answer>", re.IGNORECASE | re.DOTALL)
# Words that turn a single value into a hedge when they sit inside the tag.
HEDGE_WORDS = re.compile(r"\b(or|either|maybe|possibly|approximately|about|around|unsure)\b", re.I)
_DATE_FORMATS = ("%Y-%m-%d", "%d %B %Y", "%B %d, %Y", "%B %d %Y", "%d %b %Y", "%b %d, %Y")


@dataclass(frozen=True)
class Parsed:
    """One reply's final answer, or the reason it has none."""

    value: str | None
    hedge: str | None = None  # "missing", "multiple", "unparseable", "hedged"
    raw: str | None = None

    @property
    def ok(self) -> bool:
        return self.value is not None


def canonical(kind: Kind, value: object) -> str:
    """The canonical string for a generator-side value."""
    if kind == "int":
        return str(int(str(value)))
    if kind == "decimal":
        return _decimal_str(Decimal(str(value)))
    if kind == "date":
        if isinstance(value, date):
            return value.isoformat()
        return date.fromisoformat(str(value)).isoformat()
    return str(value)


def normalise(kind: Kind, text: str, choices: tuple[str, ...] = ()) -> str | None:
    """Read a model's answer text as a canonical value, or None if it isn't one.

    The tag should hold only the answer, but models often add a remark, as in
    "2 — only Ben and Hugo are knights" or "86 (Gregorian calendar)". Those
    still count when one value is unambiguous: a bracketed aside is dropped, the
    part after a last "=" or before a dash, colon or semicolon is tried, and
    failing that a lone number in the text is taken. "24, not 25" has two
    numbers and stays unreadable.
    """
    s = _clean(text)
    if not s:
        return None
    value = _strict(kind, s, choices)
    if value is not None or kind in ("weekday", "name") or HEDGE_WORDS.search(s):
        return value
    s = re.sub(r"\([^)]*\)", " ", s).strip()
    if "=" in s:
        s = s.rsplit("=", 1)[1]
    head = re.split(r"\s[—–-]\s|[—–:;]", s, maxsplit=1)[0]
    for candidate in (s, head):
        value = _strict(kind, _clean(candidate), choices)
        if value is not None:
            return value
    pattern = _NUMBER if kind in ("int", "decimal") else _ISO_DATE
    found = pattern.findall(s)
    return _strict(kind, found[0], choices) if len(found) == 1 else None


_NUMBER = re.compile(r"[-−]?\d{1,3}(?:,\d{3})+(?:\.\d+)?|[-−]?\d+(?:\.\d+)?")
_ISO_DATE = re.compile(r"\d{4}-\d{2}-\d{2}")


def _clean(text: str) -> str:
    return text.strip().strip("*_`'\"").strip().rstrip(".").strip()


def _strict(kind: Kind, s: str, choices: tuple[str, ...]) -> str | None:
    if not s:
        return None
    if kind in ("int", "decimal"):
        s = re.sub(r"(?<=\d)[,\s_](?=\d{3}\b)", "", s)
        s = re.sub(r"(?<=\d)[,_ ](?=\d)", "", s)
        s = s.replace("−", "-")
        # Allow a trailing unit word, such as "3600 seconds" or "12 days".
        m = re.fullmatch(r"([+-]?\d+(?:\.\d+)?)(?:\s*[A-Za-z][A-Za-z .²³]*)?", s)
        if not m:
            return None
        try:
            d = Decimal(m.group(1))
        except InvalidOperation:  # pragma: no cover - the regex rules this out
            return None
        if kind == "int":
            return str(int(d)) if d == d.to_integral_value() else None
        return _decimal_str(d)
    if kind == "date":
        cleaned = re.sub(r"^(?:[A-Za-z]+day,?\s+)", "", s)
        cleaned = re.sub(r"(\d)(st|nd|rd|th)\b", r"\1", cleaned)
        for fmt in _DATE_FORMATS:
            try:
                return datetime.strptime(cleaned, fmt).date().isoformat()
            except ValueError:
                continue
        return None
    if kind == "weekday":
        hits = [d for d in WEEKDAYS if re.search(rf"\b{d}\b", s, re.I)]
        return hits[0] if len(hits) == 1 else None
    hits = [c for c in choices if re.search(rf"\b{re.escape(c)}\b", s, re.I)]
    return hits[0] if len(hits) == 1 else None


def parse_reply(kind: Kind, reply: str, choices: tuple[str, ...] = ()) -> Parsed:
    """The final answer in a reply.

    The last tag wins when a reply repeats the same answer, as models often do
    after a restatement. Two tags with different values are a hedge.
    """
    tags = TAG.findall(reply)
    if not tags:
        return Parsed(None, "missing")
    values = []
    for raw in tags:
        if HEDGE_WORDS.search(raw):
            return Parsed(None, "hedged", raw.strip())
        value = normalise(kind, raw, choices)
        if value is None:
            return Parsed(None, "unparseable", raw.strip())
        values.append(value)
    if len(set(values)) > 1:
        return Parsed(None, "multiple", " | ".join(t.strip() for t in tags))
    return Parsed(values[-1], None, tags[-1].strip())


def _decimal_str(d: Decimal) -> str:
    if d == d.to_integral_value():
        return str(int(d))
    text = format(d.normalize(), "f")
    return text.rstrip("0").rstrip(".") if "." in text else text
