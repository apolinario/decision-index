import ast
import collections
import hashlib
import json
import math
import mmap
import pickle
import struct

from decision_index.scoring.metrics import chess_score, consensus_score, forecast_score, router_score
from decision_index.suite.build.layout import dump_ascii, git_revision, sha256, write_requests


def bag_records(path):
    with path.open("rb") as f, mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as buf:
        index = struct.unpack_from("<Q", buf, len(buf) - 8)[0]
        assert 0 < index < len(buf) and (len(buf) - index) % 8 == 0
        start = 0
        for offset in range(index, len(buf), 8):
            end = struct.unpack_from("<Q", buf, offset)[0]
            assert start <= end <= index
            yield buf[start:end]
            start = end
        assert start == index


def decode_bag(record):
    offset = 0
    strings = []
    for _ in range(2):
        length = shift = 0
        while True:
            byte = record[offset]
            offset += 1
            length |= (byte & 127) << shift
            if byte < 128:
                break
            shift += 7
            assert shift < 64
        strings.append(record[offset:offset + length].decode("utf-8"))
        offset += length
    assert len(record) - offset == 8
    value = struct.unpack_from(">d", record, offset)[0]
    assert math.isfinite(value) and 0 <= value <= 1
    return strings[0], strings[1], value


def chessbench(layout):
    import chess

    source = layout.downloads / "chessbench-test-action-value.bag"
    repo = layout.repos / "searchless_chess"
    revision = git_revision(repo)
    checksum = hashlib.file_digest(source.open("rb"), "sha256").hexdigest()
    by_fen = collections.defaultdict(dict)
    record_count = duplicate_count = 0
    for record in bag_records(source):
        fen, move, value = decode_bag(record)
        if move in by_fen[fen]:
            assert by_fen[fen][move] == value, (fen, move, "conflicting reference value")
            duplicate_count += 1
        by_fen[fen][move] = value
        record_count += 1
    output = layout.normalized / "ChessBench-legal-move.jsonl"
    exclusions = []
    count = ties = 0
    histogram = collections.Counter()
    with output.with_suffix(".partial").open("w") as f:
        for fen, values in sorted(by_fen.items()):
            board = chess.Board(fen)
            moves = {move.uci(): move for move in board.legal_moves}
            source_id = hashlib.sha256(fen.encode()).hexdigest()
            if set(moves) != set(values):
                exclusions.append({"id": source_id, "reason": "incomplete legal-move coverage", "missing": sorted(set(moves) - set(values)), "extra": sorted(set(values) - set(moves))})
                continue
            if not 2 <= len(moves) <= 255:
                exclusions.append({"id": source_id, "reason": "non-decision or API choice limit", "legal_moves": len(moves)})
                continue
            best = max(values.values())
            accepted = sorted(k for k, value in values.items() if value == best)
            order = sorted(moves, key=lambda move: hashlib.sha256((fen + ":" + move).encode()).digest())
            criteria = {move: {"uci": move, "san": board.san(moves[move])} for move in order}
            row = {"id": "ChessBench:test:" + source_id, "family": "ChessBench-legal-move", "split": "test", "state": {"fen": fen, "board": str(board), "side_to_move": "white" if board.turn else "black"}, "questions": {"move": {"type": "choice", "instructions": "Choose the strongest legal move for the side to move. Board rows run from rank 8 to rank 1; columns a to h. Uppercase pieces are white. All legal moves are supplied.", "criteria": criteria}}, "expected": {"move": accepted[0]}, "scoring": {"type": "chess_reference_value", "values": values, "accepted": accepted, "scorer": "evaluation.adapters.chessbench:score_choice", "tie_tolerance": 0.0}, "metadata": {"source_sha256": checksum, "source_revision": revision, "source": layout.rel(source), "group_id": source_id, "piece_count": len(board.piece_map()), "legal_move_count": len(moves), "note": "Use the benchmark scorer: expected is only one representative of potentially several equally best moves."}}
            assert all(chess_score(row, move)["best_move"] for move in accepted)
            f.write(dump_ascii(row) + "\n")
            count += 1
            ties += len(accepted) > 1
            histogram[len(moves)] += 1
    output.with_suffix(".partial").replace(output)
    audit = {"source_records": record_count, "source_positions": len(by_fen), "duplicate_move_records": duplicate_count, "eligible_positions": count, "positions_with_tied_best_moves": ties, "source_sha256": checksum, "source_revision": revision, "exclusions": exclusions, "legal_move_counts": dict(histogram), "sampling": "none; full eligible official test positions", "primary_metrics": ["best-move accuracy accepting all exact reference ties", "mean reference value regret"], "reference": "https://github.com/google-deepmind/searchless_chess", "output_sha256": hashlib.file_digest(output.open("rb"), "sha256").hexdigest()}
    (layout.suite / "chessbench-adaptation.json").write_text(json.dumps(audit, indent=2) + "\n")
    return {k: v for k, v in audit.items() if k not in {"exclusions", "legal_move_counts"}}


def habermas(layout):
    import pyarrow.parquet as pq

    repo = layout.repos / "habermas_machine"
    source = repo / "hm_all_candidate_comparisons.parquet"
    columns = ["metadata.id", "metadata.version", "metadata.status", "round_id", "iteration_index", "metadata.participant_id", "question.id", "question.split", "question.text", "rankings.metadata.status", "rankings.candidate_ids", "rankings.numerical_ranks", "candidates.metadata.id", "candidates.text", "own_opinion.metadata.id", "own_opinion.text", "other_opinions.metadata.id", "other_opinions.text"]
    raw = pq.read_table(source, columns=columns).to_pylist()
    groups = collections.defaultdict(list)
    filters = collections.Counter()
    for row in raw:
        if not row["metadata.version"].startswith("EVAL") or row["question.split"] not in {"IID_TEST", "OOD_TEST"}:
            filters["non_evaluation_cohort"] += 1
            continue
        if row["metadata.status"] != "COMPLETED" or row["rankings.metadata.status"] != "COMPLETED":
            filters["incomplete_response"] += 1
            continue
        if row["iteration_index"] != 0:
            filters["later_iteration_outside_initial_consensus_track"] += 1
            continue
        key = (row["round_id"], tuple(sorted(row["rankings.candidate_ids"] or [])))
        groups[key].append(row)
    with source.open("rb") as f:
        checksum = hashlib.file_digest(f, "sha256").hexdigest()
    revision = git_revision(repo)
    excluded = []
    rows = []
    for (round_id, candidates), panel in sorted(groups.items()):
        reference = panel[0]
        expected_panel = set(reference["other_opinions.metadata.id"]) | {reference["own_opinion.metadata.id"]}
        actual_panel = [r["own_opinion.metadata.id"] for r in panel]
        if set(actual_panel) != expected_panel or len(actual_panel) != len(set(actual_panel)):
            excluded.append({"round": round_id, "reason": "incomplete or repeated human panel"})
            continue
        if not 2 <= len(candidates) <= 255:
            excluded.append({"round": round_id, "reason": "candidate count outside choice primitive"})
            continue
        text_map = dict(zip(reference["candidates.metadata.id"], reference["candidates.text"], strict=True))
        opinions = {}
        sums = dict.fromkeys(candidates, 0)
        human_ranks = []
        for r in panel:
            assert r["question.text"] == reference["question.text"]
            assert set(r["other_opinions.metadata.id"]) | {r["own_opinion.metadata.id"]} == expected_panel
            assert dict(zip(r["candidates.metadata.id"], r["candidates.text"], strict=True)) == text_map
            ranks = dict(zip(r["rankings.candidate_ids"], r["rankings.numerical_ranks"], strict=True))
            assert set(ranks) == set(candidates)
            assert all(type(v) is int and 0 <= v < len(candidates) for v in ranks.values())
            opinions[r["own_opinion.metadata.id"]] = r["own_opinion.text"]
            human_ranks.append(ranks)
            for candidate, rank in ranks.items():
                sums[candidate] += rank
        if not all(isinstance(text_map[c], str) and text_map[c].strip() for c in candidates):
            excluded.append({"round": round_id, "reason": "missing or blank original candidate statement"})
            continue
        if len({" ".join(text_map[c].split()) for c in candidates}) != len(candidates):
            excluded.append({"round": round_id, "reason": "indistinguishable candidate texts with separate human ranking IDs"})
            continue
        order = sorted(candidates, key=lambda c: hashlib.sha256((round_id + c).encode()).digest())
        keys = {candidate: f"option_{i}" for i, candidate in enumerate(order)}
        summed = {keys[c]: sums[c] for c in order}
        accepted = [keys[c] for c in order if sums[c] == min(sums.values())]
        row = {"id": "Habermas-consensus:" + round_id, "family": "Habermas-consensus", "split": reference["question.split"], "state": {"policy_question": reference["question.text"], "participant_opinions": [opinions[k] for k in sorted(opinions)]}, "questions": {"consensus": {"type": "choice", "instructions": "Choose the consensus statement you predict this group would rank highest on average, given their expressed opinions. Each participant has equal weight. Assess the group's preferences, rather than your own policy preference.", "criteria": {keys[c]: text_map[c] for c in order}}}, "expected": {"consensus": accepted[0]}, "scoring": {"type": "human_group_rank", "summed_ranks": summed, "accepted": accepted, "panel_size": len(panel), "human_ranks": [{keys[c]: ranks[c] for c in order} for ranks in human_ranks], "scorer": "evaluation.adapters.habermas_consensus:score_choice"}, "metadata": {"group_id": round_id, "cohort": reference["metadata.version"], "question_id": reference["question.id"], "source_rows": [r["metadata.id"] for r in panel], "source_sha256": checksum, "source_revision": revision, "protocol": "Initial iteration; complete human panel; minimum summed rank; exact ties accepted. Named consensus-selection adaptation, not the original end-to-end mediation experiment."}}
        assert all(consensus_score(row, key)["group_preferred"] for key in accepted)
        rows.append(row)
    assert len({r["id"] for r in rows}) == len(rows), "multiple candidate pools in one round require distinct case IDs"
    out = layout.normalized / "Habermas-consensus.jsonl"
    with out.open("w") as f:
        for row in rows:
            f.write(dump_ascii(row) + "\n")
    report = {"source_comparison_rows": len(raw), "initial_evaluation_candidate_panels": len(groups), "eligible_complete_panels": len(rows), "cohort_counts": dict(collections.Counter(r["metadata"]["cohort"] for r in rows)), "split_counts": dict(collections.Counter(r["split"] for r in rows)), "source_row_filters": dict(filters), "panel_exclusions": excluded, "tied_best_panels": sum(len(r["scoring"]["accepted"]) > 1 for r in rows), "source_sha256": checksum, "source_revision": revision, "sampling": "No count cap; all eligible complete initial panels in the official evaluation cohorts."}
    (layout.suite / "habermas-consensus-adaptation.json").write_text(json.dumps(report, indent=2) + "\n")
    return {k: v for k, v in report.items() if k != "panel_exclusions"}


LAMBDA = 10.0
ALLOWED = {("pandas.core.frame", "DataFrame"), ("pandas.core.internals.managers", "BlockManager"), ("pandas._libs.internals", "_unpickle_block"), ("numpy.core.multiarray", "_reconstruct"), ("numpy.core.numeric", "_frombuffer"), ("numpy", "ndarray"), ("numpy", "dtype"), ("builtins", "slice"), ("pandas.core.indexes.base", "_new_Index"), ("pandas.core.indexes.base", "Index"), ("pandas.core.indexes.range", "RangeIndex")}


class TableUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if (module, name) not in ALLOWED:
            raise ValueError(f"Unapproved pickle global: {module}.{name}")
        return super().find_class(module, name)


def digest(text):
    return hashlib.sha256(text.encode()).hexdigest()


def routerbench(layout):
    tables = {}
    exclusions = []
    parent = {}

    def find(x):
        parent.setdefault(x, x)
        if parent[x] != x:
            parent[x] = find(parent[x])
        return parent[x]

    def union(a, b):
        a, b = find(a), find(b)
        parent[max(a, b)] = min(a, b)

    for shot in (0, 5):
        path = layout.raw / f"routerbench/routerbench_{shot}shot.pkl"
        with path.open("rb") as f:
            table = TableUnpickler(f).load()
        models = sorted(c for c in table if c + "|total_cost" in table)
        assert len(models) == 11
        records = []
        for ix, r in enumerate(table.to_dict("records")):
            outcome = {m: {"quality": float(r[m]), "cost_usd": float(r[m + "|total_cost"])} for m in models}
            if not all(math.isfinite(o["quality"]) and 0 <= o["quality"] <= 1 and math.isfinite(o["cost_usd"]) and o["cost_usd"] >= 0 for o in outcome.values()):
                exclusions.append({"shot": shot, "source_index": ix, "reason": "incomplete or invalid outcomes"})
                continue
            prompt = ast.literal_eval(r["prompt"]) if isinstance(r["prompt"], str) and r["prompt"].startswith("[") else r["prompt"]
            assert isinstance(prompt, (str, list))
            sid = str(r["sample_id"])
            phash = digest(json.dumps(prompt, ensure_ascii=True, sort_keys=True))
            union("id:" + sid, "prompt:" + phash)
            records.append({"sample_id": sid, "prompt": prompt, "task": r["eval_name"], "outcomes": outcome, "source_index": ix})
        tables[shot] = (path, models, records)
    reports = {}
    for shot, (path, models, records) in tables.items():
        for r in records:
            r["group"] = find("id:" + r["sample_id"])
            r["calibration"] = int(digest("routerbench-calibration-v1:" + r["group"])[:8], 16) % 5 == 0
        dev = [r for r in records if r["calibration"]]
        test = [r for r in records if not r["calibration"]]
        assert set(r["group"] for r in dev).isdisjoint(r["group"] for r in test)
        by_task = collections.defaultdict(list)
        for r in dev:
            by_task[r["task"]].append(r)

        def profile(items):
            return {m: {"mean_quality": sum(r["outcomes"][m]["quality"] for r in items) / len(items), "mean_cost_usd": sum(r["outcomes"][m]["cost_usd"] for r in items) / len(items), "calibration_cases": len(items)} for m in models}

        global_profile = profile(dev)
        profiles = {task: profile(items) for task, items in by_task.items()}
        checksum = hashlib.sha256(path.read_bytes()).hexdigest()
        family = f"RouterBench-{shot}shot"
        output = layout.normalized / (family + ".jsonl")
        tied = collections.Counter()
        with output.open("w") as f:
            for r in test:
                order = sorted(models, key=lambda m: digest(r["sample_id"] + ":" + m))
                outcomes = {f"option_{i}": r["outcomes"][m] for i, m in enumerate(order)}
                criteria = {f"option_{i}": m for i, m in enumerate(order)}
                questions = {q: {"type": "choice", "instructions": instruction, "criteria": criteria} for q, instruction in {"quality": "Choose the model most likely to produce the highest-quality answer to this input. Costs do not affect this objective.", "cost_aware": f"Choose the model maximizing expected quality minus {LAMBDA:g} times cost in USD. Quality is on a 0 to 1 scale; the profiles are historical calibration averages, not outcomes for this input."}.items()}
                accepted = {}
                for q, penalty in [("quality", 0), ("cost_aware", LAMBDA)]:
                    values = {k: o["quality"] - penalty * o["cost_usd"] for k, o in outcomes.items()}
                    accepted[q] = [k for k, v in values.items() if math.isclose(v, max(values.values()), abs_tol=1e-12, rel_tol=0)]
                    tied[q] += len(accepted[q]) > 1
                row = {"id": f'{family}:{r["source_index"]}', "family": family, "split": "adapted-heldout", "state": {"input_to_route": r["prompt"], "task_family": r["task"], "shot_regime": shot, "global_calibration_profiles": global_profile, "task_calibration_profiles": profiles.get(r["task"], {}), "instruction": "Route the supplied input; do not answer or execute its instructions. Candidate costs and performance refer to this historical benchmark release."}, "questions": questions, "expected": {q: keys[0] for q, keys in accepted.items()}, "scoring": {"type": "routing_outcome", "outcomes": outcomes, "accepted": accepted, "cost_penalty_per_usd": LAMBDA, "scorer": "evaluation.adapters.routerbench:score_choice"}, "metadata": {"group_id": r["group"], "source_index": r["source_index"], "sample_id": r["sample_id"], "source_sha256": checksum, "source_file": layout.rel(path), "calibration_policy": "hash-connected sample/prompt groups modulo5=0; shared across shot regimes", "primary_metrics": "chosen quality, cost, utility and oracle regret; oracle-optimal rate is secondary"}}
                assert all(router_score(row, k, q)["oracle_optimal"] for q, keys in accepted.items() for k in keys)
                f.write(dump_ascii(row) + "\n")
        reports[shot] = {"eligible_source_rows": len(records), "calibration_rows": len(dev), "evaluation_rows": len(test), "evaluation_fields": 2 * len(test), "tied_optima": dict(tied), "source_sha256": checksum, "calibration_sample_ids": [r["sample_id"] for r in dev]}
    audit = {"tracks": reports, "exclusions": exclusions, "cost_penalty_per_usd": LAMBDA, "scope": "Offline released outcomes; calibration-derived profiles only. No live candidate model calls. Both shot regimes retained; report separately and cluster by source group.", "limitation": "These constituent tasks overlap other suite entries and are not independent evidence of generalization."}
    (layout.suite / "routerbench-adaptation-audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    return {s: {k: v for k, v in r.items() if k != "calibration_sample_ids"} for s, r in reports.items()}


MIN_DATE, MAX_DATE = "2026-07-01", "2026-09-18"
MARKETS = {"manifold", "metaculus", "polymarket", "infer", "kalshi"}
VALUE_SOURCES = {"acled", "dbnomics", "fred", "yfinance", "wikipedia"}


def _key(v):
    return tuple(v) if isinstance(v, list) else v


def _render_event(q, due, resolution):
    def substitute(x):
        return x.replace("{forecast_due_date}", due).replace("{resolution_date}", resolution) if isinstance(x, str) else x

    state = {k: substitute(q[k]) for k in ("question", "background", "resolution_criteria", "source_intro", "market_info_open_datetime", "market_info_close_datetime", "market_info_resolution_criteria", "url", "freeze_datetime") if k in q and q[k] != "N/A"}
    source = q["source"]
    if source in VALUE_SOURCES:
        for k in ("freeze_datetime_value", "freeze_datetime_value_explanation"):
            if k in q:
                state[k] = substitute(q[k])
    elif source not in MARKETS:
        raise ValueError("Unreviewed source " + str(source))
    return state


def forecastbench(layout):
    base = layout.repos / "forecastbench-datasets"
    rows = []
    excluded = []
    matched = 0
    total = 0
    source_counts = collections.Counter()
    seen = {}
    for rp in sorted((base / "datasets/resolution_sets").glob("*_resolution_set.json")):
        rr = json.loads(rp.read_text())
        qp = base / "datasets/question_sets" / rr["question_set"]
        qq = json.loads(qp.read_text())
        assert rr["forecast_due_date"] == qq["forecast_due_date"]
        qm = {(_key(x["source"]), _key(x["id"])): x for x in qq["questions"]}
        assert len(qm) == len(qq["questions"])
        hashes = {"question_sha256": sha256(qp), "resolution_sha256": sha256(rp)}
        for ix, res in enumerate(rr["resolutions"]):
            total += 1
            q = qm.get((_key(res["source"]), _key(res["id"])))
            matched += q is not None
            reason = None
            if q is None:
                reason = "missing source/id question join"
            elif res.get("resolved") is not True or type(res.get("resolved_to")) not in (int, float) or res["resolved_to"] not in (0, 1):
                reason = "unresolved or nonbinary outcome"
            elif not MIN_DATE <= str(res.get("resolution_date", "")) <= MAX_DATE:
                reason = "resolution date outside policy"
            elif rr["forecast_due_date"] > res["resolution_date"]:
                reason = "resolution precedes forecast date"
            if reason:
                excluded.append({"file": rp.name, "source_index": ix, "reason": reason})
                continue
            due = rr["forecast_due_date"]
            date = res["resolution_date"]
            direction = res.get("direction")
            state = {"forecast_as_of": due, "resolution_date": date, "information_policy": "Use the frozen pre-forecast information below. No browsing. Market aggregate probability is omitted in this unaided track."}
            if direction is None:
                components = [q]
                state["event"] = _render_event(q, due, date)
                instructions = "Estimate the probability that the event in state resolves Yes, using only information available as of the forecast date."
            else:
                components = q.get("combination_of")
                if not isinstance(components, list) or not isinstance(direction, list) or len(components) != len(direction) or not all(x in (-1, 1) for x in direction):
                    excluded.append({"file": rp.name, "source_index": ix, "reason": "unsupported composite semantics"})
                    continue
                assert [x["id"] for x in components] == q["id"]
                state["component_events"] = [{"name": f"event_{j+1}", "event": _render_event(c, due, date), "required_outcome": "yes" if d == 1 else "no"} for j, (c, d) in enumerate(zip(components, direction))]
                instructions = "Estimate the probability that ALL component events have their specified required_outcome. This is a joint conjunction probability, not a conditional probability. Use only information available as of the forecast date."
            assert all(str(c.get("freeze_datetime", ""))[:10] <= due for c in components)
            expected = {"answer": "yes" if res["resolved_to"] == 1 else "no"}
            questions = {"answer": {"type": "choice", "instructions": instructions, "criteria": {"yes": "The specified event/conjunction occurs.", "no": "The specified event/conjunction does not occur."}}}
            wire = json.dumps({"state": state, "questions": questions}, sort_keys=True, ensure_ascii=True)
            fingerprint = hashlib.sha256(wire.encode()).hexdigest()
            if fingerprint in seen:
                assert seen[fingerprint] == expected, "Conflicting outcome for identical frozen request"
                excluded.append({"file": rp.name, "source_index": ix, "reason": "identical frozen payload and outcome"})
                continue
            seen[fingerprint] = expected
            event_groups = [f'{c["source"]}:{c["id"]}' for c in components]
            row = {"id": f"ForecastBench:{rp.stem}:{ix}", "family": "ForecastBench-binary", "split": "evaluation-only", "state": state, "questions": questions, "expected": expected, "scoring": {"type": "forecast_probability", "scorer": "evaluation.adapters.forecastbench:score_probability", "primary": "Brier score", "secondary": ["log_loss", "accuracy"]}, "metadata": {"group_id": f"{rp.stem}:{ix}", "underlying_event_groups": event_groups, "round": due, "direction": direction, "source": q["source"], "resolution_date": date, "provenance": {"question_source": layout.rel(qp), "resolution_source": layout.rel(rp), "resolution_index": ix, **hashes}, "cutoff_assumption": "User accepts resolutions July2026 onward; Jev knowledge cutoff is unpublished, so contamination is not certified."}}
            rows.append(row)
            source_counts[q["source"]] += 1
    norm = layout.normalized / "ForecastBench-binary.jsonl"
    with norm.open("w") as f:
        for r in rows:
            f.write(dump_ascii(r) + "\n")
    write_requests(layout, "ForecastBench-binary", rows)
    report = {"resolution_records": total, "matched_source_id_records": matched, "eligible_rows": len(rows), "source_counts": dict(source_counts), "joint_event_rows": sum(r["metadata"]["direction"] is not None for r in rows), "exclusions": excluded, "exclusion_counts": dict(collections.Counter(x["reason"] for x in excluded)), "policy": {"resolution_date_min": MIN_DATE, "resolution_date_max": MAX_DATE}, "probability_scoring": "Separate scalar and joint-event Brier/log-loss; cluster uncertainty by underlying component event IDs across rounds. No model evaluation performed."}
    (layout.suite / "forecastbench-adaptation-audit.json").write_text(json.dumps(report, indent=2, ensure_ascii=True) + "\n")
    assert forecast_score(rows[0], 0.5)["valid"] if rows else True
    return {k: v for k, v in report.items() if k not in ("exclusions", "policy")}


BUILDERS = {31: chessbench, 50: habermas, 6: routerbench, 48: forecastbench}
