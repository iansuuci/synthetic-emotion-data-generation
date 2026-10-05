"""Word count and emotion-word leakage. Shared so premise generation and story filters stay in sync."""

from __future__ import annotations

import re

from pilot.config import BANNED_WORDS

_LEAKAGE = re.compile(
    "|".join(r"\b" + re.escape(word).replace(r"\ ", r"\s+") + r"\b" for word in BANNED_WORDS),
    re.IGNORECASE,
)


def word_count(text: str) -> int:
    return len(re.findall(r"[A-Za-z0-9]+(?:'[A-Za-z]+)?", text or ""))


def leakage_hits(text: str) -> list[str]:
    return sorted({match.group(0).lower() for match in _LEAKAGE.finditer(text or "")})
