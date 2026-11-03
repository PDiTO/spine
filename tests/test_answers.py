from __future__ import annotations

import pytest

from spine.answers import canonical, normalise, parse_reply


@pytest.mark.parametrize(
    ("text", "want"),
    [
        ("1234", "1234"),
        ("1,234,567", "1234567"),
        ("1 234 567", "1234567"),
        ("**42**", "42"),
        ("42.", "42"),
        ("−7", "-7"),
        ("3600 seconds", "3600"),
        ("12.0", "12"),
        ("12.5", None),
        ("about 12", None),
        ("", None),
    ],
)
def test_normalise_int(text: str, want: str | None) -> None:
    assert normalise("int", text) == want


def test_normalise_decimal() -> None:
    assert normalise("decimal", "0.0365") == "0.0365"
    assert normalise("decimal", "375,000,000") == "375000000"
    assert normalise("decimal", "4.50") == "4.5"
    assert canonical("decimal", "4.500") == "4.5"


@pytest.mark.parametrize(
    "text",
    ["2031-06-27", "27 June 2031", "June 27, 2031", "Friday, 27 June 2031", "27th June 2031"],
)
def test_normalise_date(text: str) -> None:
    assert normalise("date", text) == "2031-06-27"


def test_normalise_weekday_and_name() -> None:
    assert normalise("weekday", "tuesday") == "Tuesday"
    assert normalise("weekday", "Tuesday or Wednesday") is None
    assert normalise("name", "Grace", ("Alice", "Grace")) == "Grace"
    assert normalise("name", "Bob", ("Alice", "Grace")) is None


def test_parse_reply_takes_the_tag() -> None:
    p = parse_reply("int", "It is 5.\n<answer>5</answer>")
    assert p.value == "5"
    assert p.ok


def test_parse_reply_repeated_tag_is_fine() -> None:
    assert parse_reply("int", "<answer>5</answer> ... <answer>5</answer>").value == "5"


@pytest.mark.parametrize(
    ("reply", "hedge"),
    [
        ("I think 5", "missing"),
        ("<answer>5</answer> or maybe <answer>6</answer>", "multiple"),
        ("<answer>5 or 6</answer>", "hedged"),
        ("<answer>five-ish</answer>", "unparseable"),
    ],
)
def test_parse_reply_hedges(reply: str, hedge: str) -> None:
    p = parse_reply("int", reply)
    assert p.value is None
    assert p.hedge == hedge
