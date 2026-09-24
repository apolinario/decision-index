import math
import statistics

from decision_index import constants as C
from decision_index.scoring.index import chance_baselines, headline, load_data, rnd, score_panel, track_list


def spec():
    return load_data("index-0.2.json")


def clip(x):
    return min(1.0, max(0.0, x))


def chance_of(n, s=None):
    s = s or spec()
    return s["chance"][str(n)]["chance"]


def share(answered, requests):
    return answered / requests if requests else 1.0


def track_value(b):
    h = headline(b)
    random = next((t.get("random") for t in b.get("tracks", []) if t["track"] == C.HEADLINE.get(b["catalog_id"], t["track"]) and t.get("random") is not None), None)
    return dict(raw=rnd(h["raw"]), skill=rnd(h["skill"]), coverage=rnd(h["coverage"]), random=random, rule="track")


def loss_value(score, answered, requests, rule):
    cov = share(answered, requests)
    value = 0.0 if score is None else clip((rule["baseline"] - rnd(score)) / (rule["baseline"] - rule["best"])) * cov
    return dict(raw=value, skill=value, coverage=cov, random=rule["baseline"], rule="vs baseline")


def added_value(score, answered, requests, chance):
    cov = share(answered, requests)
    raw = rnd(0.0 if score is None else score * cov)
    return dict(raw=raw, skill=rnd(clip((raw - chance) / (1 - chance)) if chance < 1 else raw), coverage=cov, random=chance, rule="chance")


def display_value(score, answered, requests, chance):
    cov = share(answered, requests)
    raw = 0.0 if score is None else rnd(score) * cov
    return dict(raw=raw, skill=clip((raw - chance) / (1 - chance)) if chance < 1 else raw, coverage=cov, random=chance, rule="chance")


def benchmark_value(n, s, track=None, native=None):
    key = str(n)
    if n in s["track_scored"]:
        return track_value(track)
    native = native or {}
    score, answered, requests = native.get("score"), native.get("answered", 0), native.get("requests", 0)
    if key in s["loss_rules"]:
        return loss_value(score, answered, requests, s["loss_rules"][key])
    if key in s["added"]:
        return added_value(score, answered, requests, chance_of(n, s))
    return display_value(score, answered, requests, chance_of(n, s))


def aggregate(values, s=None):
    s = s or spec()
    areas = []
    for a in s["areas"]:
        ids = a["benchmarks"]
        areas.append(dict(id=a["id"], label=a["label"], raw=statistics.mean(values[n]["raw"] for n in ids), skill=statistics.mean(values[n]["skill"] for n in ids), coverage=statistics.mean(values[n]["coverage"] for n in ids), n=len(ids), benchmarks=list(ids)))
    skills = [a["skill"] for a in areas]
    scores = dict(
        balanced_skill=100 * statistics.mean(skills),
        balanced_raw=100 * statistics.mean(a["raw"] for a in areas),
        breadth_skill=100 * (math.prod((0.1 + 0.9 * x) ** (1 / len(skills)) for x in skills) - 0.1) / 0.9,
    )
    return scores, areas


def ranks(entries, tie=None):
    tie = spec()["tie"] if tie is None else tie
    ordered = sorted(((k, v) for k, v in entries.items() if v is not None), key=lambda kv: -kv[1])
    out, start = {}, 0
    for i, (k, v) in enumerate(ordered):
        if i and ordered[i - 1][1] - v > tie:
            start = i
        out[k] = start + 1
    counts = {}
    for r in out.values():
        counts[r] = counts.get(r, 0) + 1
    return {k: {"rank": r, "tied": counts[r] > 1} for k, r in out.items()}


def index_entry(suite, results, summary, added_reports):
    s = spec()
    scored = score_panel(suite, results)
    native = {b["catalog_id"]: b for b in summary["benchmarks"]}
    native.update(added_reports)
    values = {}
    for a in s["areas"]:
        for n in a["benchmarks"]:
            values[n] = benchmark_value(n, s, scored.get(n), native.get(n))
    scores, areas = aggregate(values, s)
    benchmarks = {}
    for n, v in sorted(values.items()):
        entry = {k: rnd(v[k]) for k in ("raw", "skill", "coverage")}
        entry.update(random=rnd(v.get("random")), rule=v["rule"], in_index=True)
        if n in scored:
            entry["tracks"] = track_list(scored[n])
        benchmarks[str(n)] = entry
    for item in s["not_in_index"]:
        n = item["id"]
        b = native.get(n) or {}
        chance = rnd(chance_baselines()[str(n)]["primary"]["value"])
        v = display_value(b.get("score"), b.get("answered", 0), b.get("requests", 0), chance)
        benchmarks[str(n)] = dict(raw=rnd(v["raw"]), skill=rnd(v["skill"]), coverage=rnd(v["coverage"]), random=chance, rule="shown, not counted", in_index=False)
    return {
        "edition": s["edition"],
        "panel_id": s["panel_id"],
        "index": rnd(scores[s["headline"]], 2),
        "raw_index": rnd(scores[s["secondary"]], 2),
        "scores": {k: rnd(v, 2) for k, v in scores.items()},
        "areas": [{k: (rnd(v) if isinstance(v, float) else v) for k, v in a.items()} for a in areas],
        "benchmarks": benchmarks,
        "coverage": rnd(statistics.mean(a["coverage"] for a in areas)),
        "note": s["index_note"],
    }
