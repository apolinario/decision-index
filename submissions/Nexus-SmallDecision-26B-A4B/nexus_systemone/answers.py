"""Turn answer-slot logits into API answers.

Probabilities are rounded to 6 decimals and then renormalized in integer
micro-units, so every reported distribution sums to exactly 1.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

from .schema import ChoiceAnswer, CountAnswer, NoulAnswer, ScoreAnswer

DECIMALS = 6
_UNITS = 10**DECIMALS


def softmax(logits: Sequence[float], temperature: float = 1.0) -> list[float]:
    if temperature <= 0:
        raise ValueError(f"temperature must be positive, got {temperature}")
    scaled = [x / temperature for x in logits]
    top = max(scaled)
    exps = [math.exp(x - top) for x in scaled]
    total = math.fsum(exps)
    return [e / total for e in exps]


def round_renormalize(probs: Sequence[float]) -> list[float]:
    """Round to 6 decimals; the rounding residual goes to the largest entry."""
    units = [round(p * _UNITS) for p in probs]
    top = max(range(len(units)), key=lambda i: (probs[i], -i))
    units[top] += _UNITS - sum(units)
    if units[top] < 0:
        raise ValueError(f"probabilities do not form a distribution: {list(probs)}")
    return [u / _UNITS for u in units]


def confidence(probs: Sequence[float]) -> float:
    """(n * p_max - 1) / (n - 1), clamped to [0, 1]; 1.0 for a single option."""
    n = len(probs)
    if n == 1:
        return 1.0
    value = (n * max(probs) - 1) / (n - 1)
    return round(min(1.0, max(0.0, value)), DECIMALS)


def argmax_first(probs: Sequence[float]) -> int:
    """Index of the maximum; ties go to the earliest (canonical order) entry."""
    best = 0
    for i, p in enumerate(probs):
        if p > probs[best]:
            best = i
    return best


def noul_answer(logits: Sequence[float], temperature: float = 1.0) -> NoulAnswer:
    p_yes, _ = round_renormalize(softmax(logits, temperature))
    return NoulAnswer(noul=p_yes)


def choice_distribution_answer(
    raw: Sequence[float], keys: Sequence[str]
) -> ChoiceAnswer:
    """The answer for an already normalized distribution over ``keys``."""
    probs = round_renormalize(raw)
    return ChoiceAnswer(
        choice=keys[argmax_first(raw)],
        probabilities=dict(zip(keys, probs)),
        confidence=confidence(raw),
    )


def score_answer(
    logits: Sequence[float], levels: Sequence[Any], temperature: float = 1.0
) -> ScoreAnswer:
    raw = softmax(logits, temperature)
    probs = round_renormalize(raw)
    return ScoreAnswer(
        score=round(math.fsum(i * p for i, p in enumerate(raw)), DECIMALS),
        legend={str(i): level for i, level in enumerate(levels)},
        probabilities={str(i): p for i, p in enumerate(probs)},
        confidence=confidence(raw),
    )


def poisson_binomial(probs: Sequence[float]) -> list[float]:
    """P(count = k) for k = 0..n of independent Bernoulli(p_i) items."""
    dist = [1.0]
    for p in probs:
        nxt = [0.0] * (len(dist) + 1)
        for k, mass in enumerate(dist):
            nxt[k] += mass * (1.0 - p)
            nxt[k + 1] += mass * p
        dist = nxt
    return dist


def count_answer(
    item_logits: Sequence[Sequence[float]], temperature: float = 1.0
) -> CountAnswer:
    item_probs = [softmax(logits, temperature)[0] for logits in item_logits]
    dist = round_renormalize(poisson_binomial(item_probs))
    return CountAnswer(
        expected=round(math.fsum(item_probs), DECIMALS),
        distribution={str(k): p for k, p in enumerate(dist)},
        items=[round(p, DECIMALS) for p in item_probs],
    )
