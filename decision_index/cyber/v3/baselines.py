"""Development-fitted baselines and exact visible-evidence reference solvers."""
import re
from collections import defaultdict

from ..common import write_json, write_rows, file_hash
from ..scoring import baseline
from ..endpoint import EFFECTS, UNKNOWN as OLD_UNKNOWN
from .endpoint import solve
from .network_rules import decide as network_decide, visible_rule
from .policy import decide as policy_decide, dependency, approval
from .workplace import decide as workplace_decide
from .scoring import score


def prediction(row, distributions, engine):
    answers = {}
    for field, q in row["questions"].items():
        d = distributions[field]
        total = sum(d.values())
        probabilities = {k: float(d.get(k, 0)) / total for k in q["criteria"]}
        answers[field] = {"type": "choice", "choice": max(probabilities, key=probabilities.get), "probabilities": probabilities}
    return {**row["_evaluation"], "status": "ok", "engine": engine, "response": {"answers": answers}}


def deterministic(row):
    family, m, state = row["family"], row["metadata"], row["state"]
    answers = {}
    if family == "authorization_policy":
        if m["task"] == "program_approval":
            answers = {"authorization": policy_decide(state, row["questions"]["authorization"]["instructions"]),
                       "data_dependency": dependency(state)}
        else:
            answers = {"authorization": workplace_decide(state)}
            if "data_flow" in state:
                answers["data_classification"] = state["data_flow"]["classification"]
            else:
                answers["role_scope"] = "covered" if state["access_request"]["role"] in state["policy"]["approval_roles"] else "outside"
    elif family == "endpoint_investigation":
        if not m["diagnostic_only"]:
            app, refs = solve(state, m["task"])
            answers = {"application": app, "supporting_events": refs}
        else:
            event = state["events"][0]
            answers = {"observed_effect": EFFECTS.get(event["EventID"], "insufficient_evidence")}
            if "application" in row["questions"]:
                answers["application"] = event.get("Image") or "insufficient_evidence"
            if "user_account" in row["questions"]:
                answers["user_account"] = event.get("User") or "insufficient_evidence"
    elif m["task"] == "native_network_boundary_rule":
        answers["boundary_action"] = network_decide(state, visible_rule(row["questions"]["boundary_action"]["instructions"]))
    else:
        raise ValueError("Reference rule solver does not classify publisher incident/flow outcomes")
    return prediction(row, {field: {y: 1.0} for field, y in answers.items()}, "visible_rule_reference")


def guide_features(row):
    state = row["state"]
    features = {"alerts": state["alerts"], "evidence_rows": state["evidence_rows"]}
    for field, counts in state["counts"].items():
        for value, count in counts.items():
            features[field + "=" + value] = count
    return features


def flow_features(row, addresses=False):
    numeric = {"Dur", "TotPkts", "TotBytes", "SrcBytes", "sTos", "dTos", "duration", "orig_bytes", "resp_bytes",
               "missed_bytes", "orig_pkts", "orig_ip_bytes", "resp_pkts", "resp_ip_bytes"}
    omit = {"StartTime", "ts", "uid"} | (set() if addresses else {"SrcAddr", "DstAddr", "id.orig_h", "id.resp_h"})
    features = {}
    for field, value in row["state"]["flow"].items():
        if field in omit:
            continue
        if field in numeric:
            try:
                import math
                features[field] = math.log1p(max(0, float(value)))
            except ValueError:
                features[field + "=missing"] = 1
        else:
            features[field + "=" + str(value)] = 1
    return features


def transfer_features(row):
    """Comparable packet/byte/duration features with source units made explicit."""
    import math
    flow, source = row["state"]["flow"], row["metadata"]["source_dataset"]
    def number(name):
        try:
            value = float(flow.get(name, 0))
            return value if math.isfinite(value) and value >= 0 else 0
        except (ValueError, TypeError):
            return 0
    if source == "CTU-13":
        duration, packets, size, port, proto = number("Dur"), number("TotPkts"), number("TotBytes"), str(flow.get("Dport", "")), str(flow.get("Proto", ""))
    elif source == "IoT-23":
        duration, packets, size = number("duration"), number("orig_pkts") + number("resp_pkts"), number("orig_bytes") + number("resp_bytes")
        port, proto = str(flow.get("id.resp_p", "")), str(flow.get("proto", ""))
    else:
        duration, packets, size = number("Flow Duration") / 1e6, number("Tot Fwd Pkts") + number("Tot Bwd Pkts"), number("TotLen Fwd Pkts") + number("TotLen Bwd Pkts")
        port, proto = str(flow.get("Dst Port", "")), {"6": "tcp", "17": "udp", "0": "other"}.get(str(flow.get("Protocol", "")), "other")
    return {"duration": math.log1p(duration), "packets": math.log1p(packets), "bytes": math.log1p(size),
            "destination_port=" + port: 1, "protocol=" + proto: 1}


def conventional(development, directory):
    """GUIDE LR/RF chosen with organization-held-out CV inside development.

    Network LR is fit by source dataset, with/without address fingerprints.
    Calibration is never used to fit model weights or select these settings.
    """
    from pathlib import Path
    import joblib
    import numpy as np
    import sklearn
    from sklearn.feature_extraction import DictVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import GroupKFold
    from sklearn.pipeline import make_pipeline
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    guide = [r for r in development if r["family"] == "incident_triage"]
    xs = [guide_features(r) for r in guide]
    ys = np.array([r["expected"]["incident_grade"] for r in guide])
    groups = np.array([r["metadata"]["organization"] for r in guide])
    cv = list(GroupKFold(3).split(xs, ys, groups))
    models, recipes = {}, {}
    candidates = {"guide_logistic": [LogisticRegression(C=c, max_iter=2000, class_weight="balanced", random_state=42) for c in (0.1, 1, 10)],
                  "guide_random_forest": [RandomForestClassifier(n_estimators=200, min_samples_leaf=2, max_features="sqrt", class_weight="balanced", random_state=42, n_jobs=4)]}
    from sklearn.base import clone
    for name, classifiers in candidates.items():
        results = []
        for classifier in classifiers:
            guesses = np.full(len(guide), "", dtype=object)
            for train, heldout in cv:
                fitted = make_pipeline(DictVectorizer(), clone(classifier))
                fitted.fit([xs[i] for i in train], ys[train])
                guesses[heldout] = fitted.predict([xs[i] for i in heldout])
            org_means = [float(np.mean(guesses[groups == org] == ys[groups == org])) for org in sorted(set(groups))]
            results.append({"parameters": classifier.get_params(), "development_organization_macro_cv_accuracy": float(np.mean(org_means))})
        best = max(range(len(results)), key=lambda i: results[i]["development_organization_macro_cv_accuracy"])
        model = make_pipeline(DictVectorizer(), clone(classifiers[best])).fit(xs, ys)
        models[name] = model
        recipes[name] = {"selected": best, "candidates": results, "features": "allowlisted native count features; no organization, incident or outcome metadata"}
    network = [r for r in development if r["family"] == "network_defense" and "flow_class" in r["expected"]]
    for addresses in (False, True):
        name = "network_logistic_address" if addresses else "network_logistic_address_blind"
        models[name] = {}
        for dataset in sorted({r["metadata"]["source_dataset"] for r in network}):
            subset = [r for r in network if r["metadata"]["source_dataset"] == dataset]
            labels = [r["expected"]["flow_class"] for r in subset]
            if len(set(labels)) < 2:
                models[name][dataset] = {"constant": labels[0]}
            else:
                models[name][dataset] = make_pipeline(DictVectorizer(), LogisticRegression(C=1, max_iter=2000, class_weight="balanced", random_state=42)).fit(
                    [flow_features(r, addresses) for r in subset], labels)
        recipes[name] = {"C": 1, "class_weight": "balanced", "fit": "development only, separately per publisher dataset", "addresses": addresses, "timestamps": False}
    judged = [r for r in network if r["expected"]["flow_class"] in ("Botnet", "Normal", "Malicious", "Benign")]
    models["network_publisher_transfer_logistic"] = make_pipeline(DictVectorizer(), LogisticRegression(C=1, max_iter=2000, class_weight="balanced", random_state=42)).fit(
        [transfer_features(r) for r in judged], ["Malicious" if r["expected"]["flow_class"] in ("Botnet", "Malicious") else "Benign" for r in judged])
    recipes["network_publisher_transfer_logistic"] = {"fit": "Judged CTU/IoT development only; no CSE-CIC source used in fitting or hyperparameter selection",
        "features": "address-free comparable duration, packets, bytes, destination port and protocol", "C": 1, "class_weight": "balanced"}
    for name, model in models.items():
        joblib.dump(model, directory / f"{name}.joblib")
    write_json(directory / "recipes.json", {"sklearn": sklearn.__version__, "selection": "3-fold organization CV within development; calibration not read for weight selection",
        "models": recipes, "files": {p.name: file_hash(p) for p in directory.glob("*.joblib")}})
    return models


def classify(values, models, name):
    output = []
    for row in values:
        if name == "network_publisher_transfer_logistic":
            if row["family"] != "network_defense" or row["metadata"]["source_dataset"] != "CSE-CIC-IDS2018":
                continue
            model, features, field = models[name], transfer_features(row), "flow_class"
        elif name.startswith("guide_") and row["family"] == "incident_triage":
            model, features, field = models[name], guide_features(row), "incident_grade"
        elif name.startswith("network_") and row["family"] == "network_defense" and "flow_class" in row["expected"]:
            model = models[name].get(row["metadata"]["source_dataset"])
            features, field = flow_features(row, name == "network_logistic_address"), "flow_class"
            if model is None:
                output.append({**row["_evaluation"], "status": "unsupported_source", "engine": name})
                continue
            if isinstance(model, dict):
                output.append(prediction(row, {field: {model["constant"]: 1}}, name))
                continue
        else:
            continue
        probabilities = model.predict_proba([features])[0]
        output.append(prediction(row, {field: dict(zip(model.classes_, probabilities))}, name))
    return output


def run(development, calibration, out):
    from pathlib import Path
    out = Path(out)
    if out.exists():
        raise FileExistsError("Baseline run is immutable")
    out.mkdir(parents=True)
    models = conventional(development, out / "models")
    reports = {}
    for split, values in (("development", development), ("calibration", calibration)):
        reports[split] = {}
        engines = {mode: baseline(values, development, mode) for mode in ("uniform", "development_majority")}
        rule_rows = [r for r in values if r["family"] in ("authorization_policy", "endpoint_investigation") or r["metadata"]["task"] == "native_network_boundary_rule"]
        engines["visible_rule_reference"] = [deterministic(r) for r in rule_rows]
        engines["program_blind_policy"] = [prediction(r, {"authorization": {approval(r["state"], r["questions"]["authorization"]["instructions"], True): 1},
                "data_dependency": {"response_body": 1}}, "program_blind_policy") for r in rule_rows if r["metadata"]["task"] == "program_approval"]
        engines.update({name: classify(values, models, name) for name in models})
        for name, predictions in engines.items():
            run_ids = {p["run_id"] for p in predictions}
            subset = [r for r in values if r["_evaluation"]["run_id"] in run_ids]
            write_rows(out / f"{split}/{name}.jsonl", predictions)
            reports[split][name] = score(subset, predictions)
            write_json(out / f"{split}/{name}.json", reports[split][name])
    write_json(out / "reports.json", reports)
    return {split: {name: {track: {k: v[k] for k in ("accuracy", "case_exact_accuracy", "group_macro_accuracy", "coverage")}
                   for track, v in report["ranking_tracks"].items()} for name, report in group.items()} for split, group in reports.items()}
