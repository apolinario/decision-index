"""Answer a validated ``DecisionRequest``: preprocess, render one branch per
question, read answer-slot logits from a backend, and build the response."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol

from .answers import (
    choice_distribution_answer,
    count_answer,
    noul_answer,
    score_answer,
    softmax,
)
from .preprocess import InvalidImage, Preprocessor
from .render import (
    ChatLayout,
    Layout,
    QuestionRenderer,
    QuestionSpec,
    RenderedRequest,
    UnsupportedInput,
)
from .schema import (
    Answer,
    ChoiceQuestion,
    CountAnswer,
    CountQuestion,
    DecisionRequest,
    DecisionResponse,
    NoulQuestion,
    RequestOptions,
    ResponseMetadata,
    ScoreQuestion,
    Usage,
)

QUESTION_TYPES = ("noul", "choice", "score", "x_count")


class ContextTooLong(ValueError):
    """A prompt does not fit the model's context window."""


class InvalidRequest(ValueError):
    """A schema-valid request the engine cannot answer; ``loc`` is the path
    into the request body the 422 body reports, e.g. ``("questions", qid)``."""

    def __init__(self, loc: tuple[str | int, ...], message: str):
        super().__init__(message)
        self.loc = loc


class Backend(Protocol):
    """``branch_logits`` must be callable from several threads at once: the
    server overlaps requests and runs deterministic ones alone."""

    name: str

    def branch_logits(
        self, rendered: RenderedRequest, *, deterministic: bool = False
    ) -> tuple[list[list[float]], int]: ...


class TokenizerView(Protocol):
    """What rendering needs from the serving tokenizer."""

    def special_strings(self) -> list[str]: ...

    def encode(self, text: str) -> list[int]: ...

    def render_chat(
        self, messages: Sequence[dict[str, str]], add_generation_prompt: bool
    ) -> str: ...


@dataclass(frozen=True)
class EngineConfig:
    model_id: str
    fingerprint: str
    layout: Layout = "state_first"
    temperatures: Mapping[str, float] = field(
        default_factory=lambda: dict.fromkeys(QUESTION_TYPES, 1.0)
    )

    def __post_init__(self) -> None:
        missing = set(QUESTION_TYPES) - set(self.temperatures)
        if missing:
            raise ValueError(f"temperatures missing for {sorted(missing)}")


def resolve_items(state: Any, path: str, loc: tuple[str | int, ...]) -> list[Any]:
    """The array at dotted ``path`` inside ``state``; list indices are integers.

    Raises ``InvalidRequest`` at ``loc`` when the path does not lead to
    an array.
    """
    node = state
    for segment in path.split("."):
        if isinstance(node, dict) and segment in node:
            node = node[segment]
        elif isinstance(node, list) and segment.isdigit() and int(segment) < len(node):
            node = node[int(segment)]
        else:
            raise InvalidRequest(
                loc, f"state has no value at {path!r} (missing {segment!r})"
            )
    if not isinstance(node, list):
        raise InvalidRequest(
            loc, f"state value at {path!r} is {type(node).__name__}, not an array"
        )
    return node


class RequestRenderer:
    """Preprocess and render requests into the prompts the model was trained on."""

    def __init__(self, view: TokenizerView, config: EngineConfig):
        self.view = view
        self.config = config
        self.layout = ChatLayout(
            view.render_chat,
            lambda text: len(view.encode(text)),
            layout=config.layout,
        )

    def _specs(
        self, request: DecisionRequest, renderer: QuestionRenderer
    ) -> tuple[list[tuple[str, int, QuestionSpec]], dict[str, int]]:
        """Branch specs in request order, plus the item count of each x_count."""
        specs: list[tuple[str, int, QuestionSpec]] = []
        counts: dict[str, int] = {}
        for qid, q in request.questions.items():
            try:
                if isinstance(q, NoulQuestion):
                    criteria = q.criteria.model_dump() if q.criteria else None
                    specs.append((qid, 0, renderer.noul(q.instructions, criteria)))
                elif isinstance(q, ChoiceQuestion):
                    specs.append((qid, 0, renderer.choice(q.instructions, q.criteria)))
                elif isinstance(q, CountQuestion):
                    items = resolve_items(
                        request.state, q.items, ("questions", qid, "items")
                    )
                    counts[qid] = len(items)
                    specs += [
                        (
                            qid,
                            i,
                            renderer.count_item(q.instructions, item, i, len(items)),
                        )
                        for i, item in enumerate(items, 1)
                    ]
                elif isinstance(q, ScoreQuestion):
                    specs.append((qid, 0, renderer.score(q.instructions, q.criteria)))
                else:
                    raise TypeError(f"unhandled question type {type(q).__name__}")
            except UnsupportedInput as exc:
                raise InvalidRequest(("questions", qid), str(exc)) from exc
        return specs, counts

    def render(
        self, request: DecisionRequest
    ) -> tuple[
        RenderedRequest | None, list[tuple[str, int, QuestionSpec]], dict[str, int]
    ]:
        """The rendered prefix and branches (None when every question fans out
        to zero items), the branch specs, and each x_count's item count."""
        options = request.x_nexus or RequestOptions()
        if options.image_token_budget is not None:
            raise InvalidRequest(
                ("x_nexus", "image_token_budget"),
                "this runtime is text-only and does not take an image token budget",
            )
        pre = Preprocessor(self.view.special_strings(), now=options.now)
        try:
            state = pre.render(request.state)
        except InvalidImage as exc:
            raise InvalidRequest(("state",), str(exc)) from exc
        if state.images:
            raise InvalidRequest(
                ("state",), "this runtime is text-only and does not accept images"
            )
        renderer = QuestionRenderer(pre)
        specs, counts = self._specs(request, renderer)
        if not specs:
            return None, specs, counts
        return self.layout.render(state.text, specs), specs, counts


class DecisionEngine:
    def __init__(self, view: TokenizerView, backend: Backend, config: EngineConfig):
        self.view = view
        self.backend = backend
        self.config = config
        self.renderer = RequestRenderer(view, config)

    @property
    def model_id(self) -> str:
        return self.config.model_id

    def render(
        self, request: DecisionRequest
    ) -> tuple[
        RenderedRequest | None, list[tuple[str, int, QuestionSpec]], dict[str, int]
    ]:
        return self.renderer.render(request)

    def _branch_logits(
        self, rendered: RenderedRequest, options: RequestOptions
    ) -> tuple[list[list[float]], int]:
        try:
            return self.backend.branch_logits(
                rendered, deterministic=options.deterministic
            )
        except (UnsupportedInput, ContextTooLong) as exc:
            raise InvalidRequest(("state",), str(exc)) from exc

    def decide(self, request: DecisionRequest) -> DecisionResponse:
        options = request.x_nexus or RequestOptions()
        rendered, specs, counts = self.render(request)
        logits: list[list[float]] = []
        input_tokens = 0
        if rendered is not None:
            logits, input_tokens = self._branch_logits(rendered, options)

        answers = self._answers(request, specs, logits, counts)
        metadata = None
        if request.x_nexus is not None:
            metadata = ResponseMetadata(
                fingerprint=self.config.fingerprint,
                deterministic=options.deterministic,
            )
        return DecisionResponse(
            model=self.config.model_id,
            answers=answers,
            usage=Usage(input_tokens=input_tokens, output_tokens=len(specs)),
            x_nexus=metadata,
        )

    def _answers(
        self,
        request: DecisionRequest,
        specs: Sequence[tuple[str, int, QuestionSpec]],
        logits: Sequence[list[float]],
        counts: Mapping[str, int],
    ) -> dict[str, Answer]:
        by_qid: dict[str, list[tuple[QuestionSpec, list[float]]]] = {}
        for (qid, _, spec), row in zip(specs, logits, strict=True):
            by_qid.setdefault(qid, []).append((spec, row))
        temps = self.config.temperatures
        answers: dict[str, Answer] = {}
        for qid, q in request.questions.items():
            if isinstance(q, CountQuestion):
                rows = [row for _, row in by_qid.get(qid, [])]
                if len(rows) != counts[qid]:
                    raise RuntimeError(
                        f"{qid}: {len(rows)} item answers for {counts[qid]} items"
                    )
                answers[qid] = (
                    count_answer(rows, temps["x_count"])
                    if rows
                    else CountAnswer(expected=0.0, distribution={"0": 1.0}, items=[])
                )
                continue
            ((spec, row),) = by_qid[qid]
            if isinstance(q, ChoiceQuestion):
                answers[qid] = choice_distribution_answer(
                    softmax(row, temps["choice"]), spec.labels
                )
            elif isinstance(q, NoulQuestion):
                answers[qid] = noul_answer(row, temps["noul"])
            else:
                answers[qid] = score_answer(row, q.criteria, temps["score"])
        return answers
