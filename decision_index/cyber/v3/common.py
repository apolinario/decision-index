import copy
import hashlib

from decision_index.suite.io import dumps

from ..common import digest, record

VERSION = "esdb-v0.3"
REVISION = "7f9f3f53b9baae041c64a71d4b3867c6700fd524"


def payload_hash(state, questions):
    return hashlib.sha256(dumps({"state": state, "questions": questions}).encode()).hexdigest()


def refresh(row):
    # Preserve option order in the request receipt.
    row["_evaluation"]["payload_sha256"] = payload_hash(row["state"], row["questions"])
    row["_evaluation"]["request_sha256"] = row["_evaluation"]["payload_sha256"]
    return row


def make(track, source_id, split, group, state, questions, expected, metadata, provenance):
    row = record(track, [VERSION, source_id], split, group, state, questions, expected, metadata, provenance)
    row["_evaluation"]["run_id"] = f"{VERSION}:{track}:{row['id']}"
    return refresh(row)


def carry(row, diagnostic=False):
    row = copy.deepcopy(row)
    old = row["id"]
    row["id"] = digest([VERSION, old])
    row["metadata"]["v01_id"] = old
    row["metadata"]["diagnostic_only"] = diagnostic
    row["_evaluation"].update(run_id=f"{VERSION}:{row['family']}:{row['id']}", group_id=row["id"])
    return refresh(row)
