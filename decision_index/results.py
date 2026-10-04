"""Validate imported predictions before any benchmark or index calculation."""
import hashlib

from .engines.base import validate
from .suite.io import dumps


def checked(row, result):
    if result is None:
        return None
    expected = row["_evaluation"].get("payload_sha256")
    if expected is not None and result.get("payload_sha256") != expected:
        raise ValueError(f"Prediction payload receipt mismatch: {row['_evaluation']['run_id']}")
    if result.get("request_sha256") is not None:
        actual = hashlib.sha256(dumps({"state": row["state"], "questions": row["questions"]}).encode()).hexdigest()
        if result["request_sha256"] != actual:
            raise ValueError(f"Prediction ordered request mismatch: {row['_evaluation']['run_id']}")
    if result.get("status") != "ok":
        return result
    try:
        validate(row["questions"], result["response"])
    except (ValueError, KeyError, TypeError, AttributeError) as error:
        return {**result, "status": "invalid", "error": str(error)}
    return result


def checked_results(rows, results):
    from .cyber.gate import require
    require(rows)
    out = dict(results)
    for row in rows:
        rid = row["_evaluation"]["run_id"]
        if rid in out:
            out[rid] = checked(row, out[rid])
    return out
