"""ESDB test evaluation is admitted only inside certified orchestration."""
from contextlib import contextmanager
from contextvars import ContextVar
from functools import wraps
import importlib

_authorization = ContextVar("esdb_test_authorization", default=None)


def require(values):
    for row in values:
        if row.get("split") == "test" and row.get("_evaluation", {}).get("run_id", "").startswith("esdb-"):
            authorization = _authorization.get()
            if not authorization or row["_evaluation"]["run_id"] not in authorization:
                raise ValueError("ESDB final test requires independent review and a sealed, claimed protocol; use the final-test orchestrator")


@contextmanager
def authorized(protocol_path, edition, *, _staging=False):
    from pathlib import Path
    from .common import file_hash, read_json, rows
    module = importlib.import_module(f"decision_index.cyber.{edition}.protocol")
    protocol = module.check_frozen_protocol(protocol_path)
    dataset = Path(protocol["dataset"])
    if not _staging:
        claim = read_json(dataset / ".final-test-started.json")
        if claim["protocol_sha256"] != file_hash(protocol_path):
            raise ValueError("Final test has no matching execution claim")
    values = list(rows(dataset / "inputs/test.jsonl.gz"))
    token = _authorization.set({r["_evaluation"]["run_id"] for r in values})
    try:
        yield protocol
    finally:
        _authorization.reset(token)


def final_scoring(function):
    """Validate before staging/finishing; scoring receives only this edition."""
    @wraps(function)
    def wrapped(protocol_path, *args, **kwargs):
        edition = function.__module__.split(".")[-2]
        with authorized(protocol_path, edition, _staging=function.__name__ == "stage"):
            return function(protocol_path, *args, **kwargs)
    return wrapped
