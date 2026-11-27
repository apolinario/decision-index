"""Request and response models for ``POST /v1/systemone``.

The base shapes follow TypeSafe's documented System One API exactly; every
extension lives under an ``x_`` name so strict clients never see it.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Annotated, Any, Literal, Union

from pydantic import AfterValidator, BaseModel, ConfigDict, Field, field_validator

MAX_CHOICE_OPTIONS = 255
MIN_SCORE_LEVELS = 2
MAX_SCORE_LEVELS = 10


def _structured(value: Any) -> Any:
    if not isinstance(value, (str, dict, list)):
        raise ValueError("must be a string, object or array")
    return value


# What a caller may place in state, instructions or criteria. One validator
# rather than a Union, so a 422 names the field itself and not a union member.
Structured = Annotated[Any, AfterValidator(_structured)]


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class NoulCriteria(_Strict):
    true: Structured | None = None
    false: Structured | None = None


class NoulQuestion(_Strict):
    type: Literal["noul"]
    instructions: Structured
    criteria: NoulCriteria | None = None


class ChoiceQuestion(_Strict):
    type: Literal["choice"]
    instructions: Structured
    criteria: dict[str, Structured | None]

    @field_validator("criteria")
    @classmethod
    def _option_count(cls, value: dict[str, Any]) -> dict[str, Any]:
        if not 1 <= len(value) <= MAX_CHOICE_OPTIONS:
            raise ValueError(
                f"choice criteria must have 1 to {MAX_CHOICE_OPTIONS} options, got {len(value)}"
            )
        return value


class ScoreQuestion(_Strict):
    type: Literal["score"]
    instructions: Structured
    criteria: list[Structured]

    @field_validator("criteria")
    @classmethod
    def _level_count(cls, value: list[Any]) -> list[Any]:
        if not MIN_SCORE_LEVELS <= len(value) <= MAX_SCORE_LEVELS:
            raise ValueError(
                f"score criteria must have {MIN_SCORE_LEVELS} to {MAX_SCORE_LEVELS} levels, "
                f"got {len(value)}"
            )
        return value


class CountQuestion(_Strict):
    """Extension: one Noul per item of a state array, combined by Poisson binomial."""

    type: Literal["x_count"]
    items: str = Field(min_length=1, description="Dotted path to an array inside state")
    instructions: Structured


Question = Annotated[
    Union[NoulQuestion, ChoiceQuestion, ScoreQuestion, CountQuestion],
    Field(discriminator="type"),
]


class RequestOptions(_Strict):
    deterministic: bool = False
    now: str | None = Field(
        default=None, description="ISO 8601 anchor for resolving relative dates"
    )
    image_token_budget: int | None = Field(default=None, ge=16, le=4096)


class DecisionRequest(BaseModel):
    # Unknown top-level keys are ignored rather than rejected so SDK versions
    # that send extra metadata keep working.
    model_config = ConfigDict(extra="ignore")

    state: Structured
    model: str | None = None
    questions: dict[str, Question]
    x_nexus: RequestOptions | None = None

    @field_validator("questions")
    @classmethod
    def _non_empty(cls, value: dict[str, Any]) -> dict[str, Any]:
        if not value:
            raise ValueError("questions must contain at least one question")
        return value


class NoulAnswer(BaseModel):
    type: Literal["noul"] = "noul"
    noul: float


class ChoiceAnswer(BaseModel):
    type: Literal["choice"] = "choice"
    choice: str
    probabilities: dict[str, float]
    confidence: float


class ScoreAnswer(BaseModel):
    type: Literal["score"] = "score"
    score: float
    legend: dict[str, Structured]
    probabilities: dict[str, float]
    confidence: float


class CountAnswer(BaseModel):
    type: Literal["x_count"] = "x_count"
    expected: float
    distribution: dict[str, float]
    items: list[float]


Answer = Union[NoulAnswer, ChoiceAnswer, ScoreAnswer, CountAnswer]


class Usage(BaseModel):
    input_tokens: int
    output_tokens: int


class ResponseMetadata(BaseModel):
    fingerprint: str
    deterministic: bool


class DecisionResponse(BaseModel):
    model: str
    answers: dict[str, Answer]
    usage: Usage
    x_nexus: ResponseMetadata | None = None


class ModelMetadata(BaseModel):
    """One entry of ``GET /v1/models``, in the shape the official SDKs read."""

    name: str
    description: str
    release_date: str = Field(pattern=r"^\d{4}-\d{2}-\d{2}$")


class ModelMetadataList(BaseModel):
    models: list[ModelMetadata]


# Discriminated-union errors carry the tag value as an extra path segment
# (``questions.team.choice.criteria``); callers only know the JSON path.
_UNION_TAGS = {"noul", "choice", "score", "x_count"}


def field_path(loc: Sequence[int | str]) -> str:
    """Dotted JSON path for a pydantic error location, minus framework segments."""
    parts: list[str] = []
    segments = list(loc)
    if segments and segments[0] == "body":
        segments = segments[1:]
    for i, segment in enumerate(segments):
        if (
            isinstance(segment, str)
            and segment in _UNION_TAGS
            and i >= 2
            and segments[i - 2] == "questions"
        ):
            continue
        parts.append(f"[{segment}]" if isinstance(segment, int) else str(segment))
    return ".".join(parts).replace(".[", "[")


def error_body(errors: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """The JSON body returned with HTTP 422: the official API's ``loc``, ``msg``
    and ``type`` (which the SDKs turn into their error message), plus ``field``,
    the dotted JSON path."""
    return {
        "detail": [
            {
                "loc": ["body", *err.get("loc", ())],
                "msg": err.get("msg", "invalid value"),
                "type": err.get("type", "value_error"),
                "field": field_path(err.get("loc", ())),
            }
            for err in errors
        ]
    }
