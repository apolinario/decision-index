"""Laws for the wyat-2 Decision Index adapter, over generated requests.

The adapter adds nothing to the model: it passes the complete public record to
``model.predict``, maps each typed answer to the kit's response and refuses
what the model cannot answer. A stand-in model returns known answers, so these
laws check the adapter alone; ``wyat2_runtime_laws.py`` runs a real artifact.

    tlaw check tests/wyat2_engine_laws.py --explain
"""

from types import SimpleNamespace

import tlaw

from decision_index.engines import Unsupported, validate
from examples.wyat2_engine import CAPACITY_REASONS, Wyat2Engine

Word = tlaw.one_of(["alpha", "beta", "délai", "日本語", "x", "a longer option text"])
Words = tlaw.list_of(Word, min_size=1, max_size=4)
Question = tlaw.record(
    kind=tlaw.one_of(["choice", "noul"]),
    options=tlaw.integer(2, 6),
    instructions=Words,
    texts=tlaw.list_of(Words, min_size=6, max_size=6),
)
Questions = tlaw.list_of(Question, min_size=1, max_size=3)
Reason = tlaw.one_of([*CAPACITY_REASONS, "malformed-record", "device-failure"])
OtherType = tlaw.one_of(["score", "text", "ranking"])
OptionCount = tlaw.one_of([0, 1, 2, 3, 254, 255, 256, 300])
Device = tlaw.one_of(["cpu", "cuda"])


class StandInInputError(ValueError):
    def __init__(self, reason):
        self.reason = reason
        super().__init__("a capacity message must not carry the request")


def _questions(drawn):
    questions = {}
    for index, item in enumerate(drawn):
        question = {
            "type": item["kind"],
            "instructions": " ".join(item["instructions"]),
        }
        if item["kind"] == "choice":
            question["criteria"] = {
                f"o{j}": " ".join(item["texts"][j]) for j in range(item["options"])
            }
        else:
            question["criteria"] = {"false": "No.", "true": "Yes."}
        questions[f"q{index}"] = question
    return questions


def _answers(questions):
    """Known typed answers: probabilities grow with the option's position."""
    answers = {}
    for key, question in questions.items():
        if question["type"] == "choice":
            ids = list(question["criteria"])
            total = sum(range(1, len(ids) + 1))
            probabilities = {o: (j + 1) / total for j, o in enumerate(ids)}
            answers[key] = SimpleNamespace(
                choice=ids[-1], probabilities_by_id=probabilities
            )
        else:
            answers[key] = SimpleNamespace(probability_true=0.25 + 0.5 / len(key))
    return answers


def _engine(predict):
    engine = Wyat2Engine.__new__(Wyat2Engine)
    engine.model = SimpleNamespace(model_id="ouro-stand-in", predict=predict)
    engine.wyat = SimpleNamespace(ModelInputError=StandInInputError)
    return engine


@tlaw.law
def every_answer_is_the_models_and_passes_the_kits_validation(
    drawn: Questions, state: Words
):
    questions = _questions(drawn)
    native = _answers(questions)
    received = []

    def predict(record):
        received.append(record)
        return native

    response, raw = _engine(predict)(" ".join(state), questions)
    validate(questions, response)
    expected = {
        key: (
            {
                "type": "choice",
                "choice": native[key].choice,
                "probabilities": native[key].probabilities_by_id,
            }
            if question["type"] == "choice"
            else {"type": "noul", "noul": native[key].probability_true}
        )
        for key, question in questions.items()
    }
    return tlaw.same(
        [response["answers"], response["model"], raw, len(received)],
        [expected, "ouro-stand-in", None, 1],
    )


@tlaw.law
def the_model_receives_the_complete_record_unchanged(drawn: Questions, state: Words):
    questions = _questions(drawn)
    record_state = {"text": " ".join(state), "turns": list(state)}
    received = []

    def predict(record):
        received.append(record)
        return _answers(questions)

    _engine(predict)(record_state, questions)
    (record,) = received
    return record["state"] is record_state and record["questions"] is questions


@tlaw.law
def another_question_type_is_unsupported_before_the_model_runs(
    drawn: Questions, other: OtherType
):
    questions = _questions(drawn)
    questions["extra"] = {
        "type": other,
        "instructions": "Rate.",
        "criteria": ["a", "b"],
    }
    calls = []
    engine = _engine(lambda record: calls.append(record))
    return tlaw.raises(Unsupported, engine, "state", questions).and_then(
        lambda _: tlaw.same(calls, [])
    )


@tlaw.law
def a_choice_reaches_the_model_exactly_when_it_has_2_to_255_options(
    drawn: Questions, count: OptionCount
):
    questions = _questions(drawn)
    questions["wide"] = {
        "type": "choice",
        "instructions": "Pick.",
        "criteria": {f"o{j}": f"option {j}" for j in range(count)},
    }
    calls = []

    def predict(record):
        calls.append(record)
        return _answers(questions)

    engine = _engine(predict)
    if 2 <= count <= 255:
        response, _ = engine("state", questions)
        validate(questions, response)
        return tlaw.same(len(calls), 1)
    return tlaw.raises(Unsupported, engine, "state", questions).and_then(
        lambda _: tlaw.same(calls, [])
    )


@tlaw.law
def synchronize_waits_for_the_device_only_on_cuda(device: Device):
    waited = []
    engine = Wyat2Engine.__new__(Wyat2Engine)
    engine.device = SimpleNamespace(type=device)
    engine.torch = SimpleNamespace(cuda=SimpleNamespace(synchronize=waited.append))
    engine.synchronize()
    return tlaw.same(waited, [engine.device] if device == "cuda" else [])


@tlaw.law
def declared_capacity_is_unsupported_and_anything_else_stays_an_error(
    drawn: Questions, reason: Reason
):
    def predict(record):
        raise StandInInputError(reason)

    engine = _engine(predict)
    expected = Unsupported if reason in CAPACITY_REASONS else StandInInputError
    outcome = tlaw.raises(expected, engine, "state", _questions(drawn))
    if expected is Unsupported:
        return outcome.and_then(
            lambda error: tlaw.same(str(error), "wyat-2 declared capacity: " + reason)
        )
    return outcome
