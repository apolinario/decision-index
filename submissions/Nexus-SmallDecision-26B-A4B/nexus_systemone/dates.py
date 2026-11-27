"""Rewrite dates to ISO 8601 with a weekday; mark ambiguous ones, never guess.

Rules:
- ISO dates and datetimes keep their text and gain ``(Www)``.
- Month-name dates with a year become ``YYYY-MM-DD (Www)``.
- Numeric day/month dates resolve only when one field exceeds 12 or both are
  equal; otherwise they keep their text and gain ``[ambiguous-date]``.
  Two-digit years are ambiguous by century and are marked the same way.
- Relative phrases (today, 3 days ago, in 2 weeks) resolve only against an
  explicit ``now``; without one they are left untouched.
"""

from __future__ import annotations

import calendar
import re
from datetime import MAXYEAR, MINYEAR, date, datetime, timedelta

AMBIGUOUS_MARKER = "[ambiguous-date]"

_WEEKDAYS = ("Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun")
_MONTHS = {
    name: i + 1
    for i, names in enumerate(
        (
            ("january", "jan"),
            ("february", "feb"),
            ("march", "mar"),
            ("april", "apr"),
            ("may",),
            ("june", "jun"),
            ("july", "jul"),
            ("august", "aug"),
            ("september", "sep", "sept"),
            ("october", "oct"),
            ("november", "nov"),
            ("december", "dec"),
        )
    )
    for name in names
}
_MONTH_ALT = "|".join(sorted(_MONTHS, key=len, reverse=True))
_ORD = r"(?:st|nd|rd|th)?"
# Guards keep the rewrite idempotent: annotated text is never annotated again.
_NOT_ANNOTATED = r"(?!\s\((?:Mon|Tue|Wed|Thu|Fri|Sat|Sun)\))"
_NOT_MARKED = r"(?!\s\[ambiguous-date\])"

_ISO = re.compile(
    r"(?<![\d-])(\d{4})-(\d{2})-(\d{2})"
    r"(T\d{2}:\d{2}(?::\d{2}(?:\.\d+)?)?(?:Z|[+-]\d{2}:?\d{2})?)?"
    # The optional time is greedy; the lookahead stops a backtrack to a shorter
    # prefix that would annotate the middle of a longer datetime.
    r"(?![\dTZ:]|\.\d|[+-]\d)" + _NOT_ANNOTATED
)
_YMD_SLASH = re.compile(r"(?<![\d/.])(\d{4})[/.](\d{1,2})[/.](\d{1,2})(?![\d/.])")
_NUMERIC = re.compile(
    r"(?<![\d/.])(\d{1,2})([/.-])(\d{1,2})\2(\d{4}|\d{2})(?![\d/]|\.\d)" + _NOT_MARKED
)
_DAY_MONTH_YEAR = re.compile(
    rf"\b(\d{{1,2}}){_ORD}\s+(?:of\s+)?({_MONTH_ALT})\.?,?\s+(\d{{4}})\b", re.IGNORECASE
)
_MONTH_DAY_YEAR = re.compile(
    rf"\b({_MONTH_ALT})\.?\s+(\d{{1,2}}){_ORD},?\s+(\d{{4}})\b", re.IGNORECASE
)
_RELATIVE = re.compile(
    r"\b(today|yesterday|tomorrow|(\d+)\s+(day|week)s?\s+ago|in\s+(\d+)\s+(day|week)s?)\b(?!\s\(=)",
    re.IGNORECASE,
)


def iso_with_weekday(d: date) -> str:
    return f"{d.isoformat()} ({_WEEKDAYS[d.weekday()]})"


def _safe_date(year: int, month: int, day: int) -> date | None:
    """The date, or None when the text names no real calendar day (such as
    2026-02-30); such text is left as written."""
    if not (MINYEAR <= year <= MAXYEAR and 1 <= month <= 12):
        return None
    if not 1 <= day <= calendar.monthrange(year, month)[1]:
        return None
    return date(year, month, day)


def _iso_sub(m: re.Match[str]) -> str:
    d = _safe_date(int(m.group(1)), int(m.group(2)), int(m.group(3)))
    if d is None:
        return m.group(0)
    return f"{m.group(0)} ({_WEEKDAYS[d.weekday()]})"


def _ymd_sub(m: re.Match[str]) -> str:
    d = _safe_date(int(m.group(1)), int(m.group(2)), int(m.group(3)))
    return m.group(0) if d is None else iso_with_weekday(d)


def _numeric_sub(m: re.Match[str]) -> str:
    a, b, year_text = int(m.group(1)), int(m.group(3)), m.group(4)
    if len(year_text) == 2:
        return f"{m.group(0)} {AMBIGUOUS_MARKER}"
    year = int(year_text)
    if a > 12 and b <= 12:
        d = _safe_date(year, b, a)
    elif b > 12 and a <= 12:
        d = _safe_date(year, a, b)
    elif a == b:
        d = _safe_date(year, a, b)
    elif a <= 12 and b <= 12:
        return f"{m.group(0)} {AMBIGUOUS_MARKER}"
    else:
        d = None
    return m.group(0) if d is None else iso_with_weekday(d)


def _dmy_sub(m: re.Match[str]) -> str:
    d = _safe_date(int(m.group(3)), _MONTHS[m.group(2).lower()], int(m.group(1)))
    return m.group(0) if d is None else iso_with_weekday(d)


def _mdy_sub(m: re.Match[str]) -> str:
    d = _safe_date(int(m.group(3)), _MONTHS[m.group(1).lower()], int(m.group(2)))
    return m.group(0) if d is None else iso_with_weekday(d)


def parse_now(now: str) -> date:
    """The calendar date of an ISO 8601 ``now`` anchor (its own offset applies)."""
    return datetime.fromisoformat(now.replace("Z", "+00:00")).date()


def _relative_sub(anchor: date):
    def sub(m: re.Match[str]) -> str:
        phrase = m.group(1).lower()
        if phrase == "today":
            target = anchor
        elif phrase == "yesterday":
            target = anchor - timedelta(days=1)
        elif phrase == "tomorrow":
            target = anchor + timedelta(days=1)
        elif m.group(2):
            unit = 7 if m.group(3).lower() == "week" else 1
            target = anchor - timedelta(days=int(m.group(2)) * unit)
        else:
            unit = 7 if m.group(5).lower() == "week" else 1
            target = anchor + timedelta(days=int(m.group(4)) * unit)
        return f"{m.group(0)} (= {iso_with_weekday(target)})"

    return sub


def normalize_dates(text: str, now: str | None = None) -> str:
    text = _ISO.sub(_iso_sub, text)
    text = _YMD_SLASH.sub(_ymd_sub, text)
    text = _NUMERIC.sub(_numeric_sub, text)
    text = _DAY_MONTH_YEAR.sub(_dmy_sub, text)
    text = _MONTH_DAY_YEAR.sub(_mdy_sub, text)
    if now is not None:
        text = _RELATIVE.sub(_relative_sub(parse_now(now)), text)
    return text
