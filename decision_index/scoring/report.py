import collections
import statistics

from decision_index.scoring.metrics import binary_f1, chess_score, consensus_score, forecast_score, macro_f1, mean, percentile, prediction, router_score, score_query, semantic_label

F1_BENCHMARKS = (4, 5, 10, 11, 12, 37, 39, 40, 41, 42)
CASE_EXACT_BENCHMARKS = (1, 9, 33, 38)
RANKING_BENCHMARKS = (2, 36)
ACCEPTED_BENCHMARKS = (31, 50)


def compact(r):
    return {k: r[k] for k in ("run_id", "catalog_id", "status", "response", "total_wall_ms", "model_request_wall_ms", "http_wall_ms") if k in r}


def load_results(path):
    from decision_index.suite.io import read_jsonl

    out = {}
    for r in read_jsonl(path, complete_lines_only=True):
        out[r["run_id"]] = compact(r)
    return out


def score(rows, results):
    number = rows[0]["_evaluation"]["catalog_id"]
    groups = collections.defaultdict(list)
    for row in rows:
        groups[row["_evaluation"]["group_id"]].append(row)
    successful, field_hits, pairs, case_exact = [], [], [], []
    by_field, special, cluster, subgroups = collections.defaultdict(list), collections.defaultdict(list), collections.defaultdict(list), collections.defaultdict(list)
    for row in rows:
        e = row["_evaluation"]
        r = results.get(e["run_id"])
        if not r or r["status"] != "ok":
            continue
        successful.append(r)
        answers = r["response"]["answers"]
        sc = row.get("scoring", {}).get("type")
        if sc == "retrieval_ranking":
            continue
        if sc == "forecast_probability":
            v = forecast_score(row, answers["answer"]["probabilities"]["yes"])
            special["brier"].append(v["brier"])
            special["log_loss"].append(v["log_loss"])
            special["accuracy"].append(v["correct"])
            label = "joint" if row["metadata"].get("direction") is not None else "scalar"
            subgroups[label].append(v["brier"])
            continue
        if sc == "routing_outcome":
            for q in row["questions"]:
                s = router_score(row, answers[q]["choice"], q)
                for k, v in s.items():
                    if k != "valid":
                        special[q + "_" + k].append(v)
            continue
        hits = []
        for key, q in row["questions"].items():
            pred = prediction(q, answers[key])
            gold = row["expected"][key]
            if sc == "chess_reference_value":
                s = chess_score(row, pred)
                hit = s["best_move"]
                special["value_regret"].append(s["value_regret"])
            elif sc == "human_group_rank":
                s = consensus_score(row, pred)
                hit = s["group_preferred"]
                special["normalized_rank_regret"].append(s["normalized_rank_regret"])
            else:
                hit = pred == gold
            assert gold is not None, "Null gold may only be handled by ranking scorer"
            hits.append(hit)
            field_hits.append(hit)
            by_field[key].append((gold, pred))
            if number in F1_BENCHMARKS:
                pairs.append((semantic_label(q, gold), semantic_label(q, pred)))
            else:
                pairs.append((gold, pred))
        if number == 22:
            cluster[str(row["metadata"]["song_id"])].extend(hits)
            subgroups[row["metadata"]["gold_class"]].extend(hits)
        if number == 23:
            cluster[str(row["metadata"]["user_id"])].extend(hits)
        if number == 20:
            subgroups[row["metadata"]["variant"]].extend(hits)
        if number == 9:
            for key, hit in zip(row["questions"], hits):
                subgroups[row["metadata"]["question_roles"][key].split(":")[0]].append(hit)
    for group, rs in groups.items():
        if not all(results.get(r["_evaluation"]["run_id"], {}).get("status") == "ok" for r in rs):
            continue
        if number in RANKING_BENCHMARKS:
            probabilities = {}
            for r in rs:
                ans = results[r["_evaluation"]["run_id"]]["response"]["answers"]
                probabilities.update({did: ans[f]["probabilities"]["yes"] for f, did in r["scoring"]["field_to_document"].items()})
            s = rs[0]["scoring"]
            v = score_query(s, probabilities)
            if not v["valid"]:
                raise ValueError(v)
            for k, value in v.items():
                if k != "valid" and isinstance(value, (int, float)):
                    special[k].append(value)
            n = len(s["retrieved_ids"])
            baseline = {did: 1 - i / (n + 1) for i, did in enumerate(s["retrieved_ids"])}
            special["bm25_ndcg_at_10"].append(score_query(s, baseline)["ndcg_at_10"])
            continue
        if number in (6, 48):
            continue
        hits = []
        for r in rs:
            ans = results[r["_evaluation"]["run_id"]]["response"]["answers"]
            for q, g in r["expected"].items():
                p = prediction(r["questions"][q], ans[q])
                hits.append(p in r["scoring"]["accepted"] if number in ACCEPTED_BENCHMARKS else p == g)
        case_exact.append(all(hits))
    durations = [r["total_wall_ms"] for r in successful]
    http = [r["http_wall_ms"] for r in successful if "http_wall_ms" in r]
    report = dict(
        selected_cases=len(groups),
        selected_requests=len(rows),
        successful_requests=len(successful),
        failed_requests=sum(e["_evaluation"]["run_id"] in results and results[e["_evaluation"]["run_id"]]["status"] != "ok" for e in rows),
        pending_requests=sum(e["_evaluation"]["run_id"] not in results for e in rows),
        completed_cases=sum(all(results.get(r["_evaluation"]["run_id"], {}).get("status") == "ok" for r in rs) for rs in groups.values()),
        fields_scored=len(field_hits),
        field_accuracy=mean(field_hits),
        case_exact_accuracy=mean(case_exact),
        http_latency_median_ms=percentile(http, 0.5),
        http_latency_p95_ms=percentile(http, 0.95),
        total_latency_median_ms=percentile(durations, 0.5),
        total_latency_p95_ms=percentile(durations, 0.95),
        input_tokens=sum((r.get("response") or {}).get("usage", {}).get("input_tokens", 0) for r in successful),
    )
    report["custom_metrics"] = {k: mean(v) for k, v in special.items()}
    if cluster:
        report["cluster_macro_accuracy"] = mean([mean(x) for x in cluster.values()])
    if subgroups:
        report["subgroups"] = {k: {"n": len(v), "mean": mean(v)} for k, v in sorted(subgroups.items())}
    if number in F1_BENCHMARKS:
        report["macro_f1"] = macro_f1(pairs)
    if number in (1, 33, 38):
        binary = [p for values in by_field.values() for p in values]
        report["positive_micro_f1"] = binary_f1(binary)
    if number == 40:
        report["positive_f1_by_field"] = {k: binary_f1(v) for k, v in by_field.items() if all(g in ("yes", "no") for g, p in v)}
        if report["positive_f1_by_field"]:
            report["category_macro_f1"] = mean(list(report["positive_f1_by_field"].values()))
    if number in RANKING_BENCHMARKS:
        report["primary_metric"], report["primary_value"] = "nDCG@10", report["custom_metrics"].get("ndcg_at_10")
    elif number == 6:
        report["primary_metric"], report["primary_value"] = "selected quality (quality objective)", report["custom_metrics"].get("quality_quality")
    elif number == 48:
        report["primary_metric"], report["primary_value"] = "Brier (lower is better)", report["custom_metrics"].get("brier")
    elif number in CASE_EXACT_BENCHMARKS:
        report["primary_metric"], report["primary_value"] = "case exact accuracy", report["case_exact_accuracy"]
    elif number in F1_BENCHMARKS and number != 40:
        report["primary_metric"], report["primary_value"] = "macro-F1", report.get("macro_f1")
    elif number == 40:
        report["primary_metric"], report["primary_value"] = "see separate subtask tracks", None
    else:
        report["primary_metric"], report["primary_value"] = "accuracy", report["field_accuracy"]
    return report


def benchmark_summary(suite, results, engine, reference=None):
    groups = collections.defaultdict(list)
    for r in suite.rows(apply_exclusions=True):
        groups[r["_evaluation"]["catalog_id"]].append(r)
    reports = []
    for n, rows in groups.items():
        cases = collections.defaultdict(list)
        for r in rows:
            cases[r["_evaluation"]["group_id"]].append(r)
        complete = [r for rs in cases.values() if all(results.get(r["_evaluation"]["run_id"], {}).get("status") == "ok" for r in rs) for r in rs]
        counts = collections.Counter(results.get(r["_evaluation"]["run_id"], {}).get("status", "pending") for r in rows)
        v = score(complete, results) if complete else {}
        j = score(complete, reference) if complete and reference else {}
        ts = [results[r["_evaluation"]["run_id"]]["total_wall_ms"] for r in rows if results.get(r["_evaluation"]["run_id"], {}).get("status") == "ok"]
        tracks = collections.defaultdict(list)
        for r in complete:
            tracks[r["_evaluation"]["track"]].append(r)
        entry = dict(
            catalog_id=n,
            dataset=rows[0]["_evaluation"]["dataset"],
            requests=len(rows),
            answered=counts["ok"],
            unsupported=counts["unsupported"],
            errors=counts["error"],
            abstained=counts["abstained"],
            pending=counts["pending"],
            scored_requests=len(complete),
            metric=v.get("primary_metric"),
            score=v.get("primary_value"),
            reference_same_cases=j.get("primary_value"),
            median_ms=statistics.median(ts) if ts else None,
        )
        if len(tracks) > 1:
            entry["tracks"] = {k: {"metric": t.get("primary_metric"), "score": t.get("primary_value"), "scored_requests": len(rs)} for k, rs in sorted(tracks.items()) for t in [score(rs, results)]}
        if v:
            entry["detail"] = {k: v[k] for k in ("field_accuracy", "case_exact_accuracy", "macro_f1", "cluster_macro_accuracy", "custom_metrics", "subgroups", "positive_f1_by_field", "category_macro_f1", "positive_micro_f1") if k in v}
        reports.append(entry)
    ts = sorted(r["total_wall_ms"] for r in results.values() if r["status"] == "ok" and r.get("total_wall_ms") is not None)
    latency = {"median": statistics.median(ts), "p95": ts[int(0.95 * (len(ts) - 1))], "mean": statistics.mean(ts)} if ts else {"median": None, "p95": None, "mean": None}
    return {
        "engine": engine,
        "counts": dict(collections.Counter(r["status"] for r in results.values())),
        "successful_request_latency_ms": latency,
        "benchmarks": reports,
        "note": "Native benchmark metrics on complete supported case groups. Unsupported cases excluded from accuracy but retained in coverage.",
    }
