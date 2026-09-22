import ast
import collections
import json
import math
import re
import sqlite3

from decision_index.scoring.metrics import score_query
from decision_index.suite.build.layout import REFERENCE_MODEL, sha256

STOP = set("a an the is are was were be been being do does did to of on in for by with and or as at from that this these those it its i you we they he she have has had can could should would will what which how when where why who please me my your".split())


def canonical(x):
    return json.dumps(x, ensure_ascii=True, sort_keys=True, separators=(",", ":"))


def parquet_rows(path):
    import pyarrow.parquet as pq

    for batch in pq.ParquetFile(path).iter_batches(batch_size=1024):
        yield from batch.to_pylist()


def make_index(layout, name, paths, content_key):
    import pyarrow.parquet as pq

    directory = layout.suite / "retrieval-indexes-v2"
    directory.mkdir(exist_ok=True)
    fingerprint = canonical({p.name: sha256(p) for p in paths})
    conn = sqlite3.connect(directory / (name + ".sqlite"))
    conn.execute("CREATE TABLE IF NOT EXISTS metadata(key TEXT PRIMARY KEY,value TEXT)")
    old = conn.execute("SELECT value FROM metadata WHERE key='source'").fetchone()
    if old and old[0] == fingerprint:
        return conn, json.loads(fingerprint)
    conn.execute("DROP TABLE IF EXISTS docs")
    conn.execute('CREATE VIRTUAL TABLE docs USING fts5(docid UNINDEXED,content,tokenize="porter unicode61")')
    seen = set()
    for p in paths:
        for batch in pq.ParquetFile(p).iter_batches(batch_size=1024):
            values = []
            for r in batch.to_pylist():
                did = str(r["id"])
                assert did not in seen, (name, did)
                seen.add(did)
                content = r[content_key]
                assert isinstance(content, str)
                values.append((did, content))
            conn.executemany("INSERT INTO docs(docid,content) VALUES (?,?)", values)
    conn.execute("INSERT INTO docs(docs) VALUES('optimize')")
    conn.execute("INSERT OR REPLACE INTO metadata VALUES (?,?)", ("source", fingerprint))
    conn.commit()
    return conn, json.loads(fingerprint)


def retrieve(conn, query, excluded=(), k=32):
    words = list(dict.fromkeys(w.casefold() for w in re.findall(r"[^\W_]+", query, flags=re.UNICODE) if w.casefold() not in STOP))
    excluded = sorted(set(str(x) for x in excluded))
    where = " AND docid NOT IN (" + ",".join("?" for _ in excluded) + ")" if excluded else ""
    match = " OR ".join('"' + w.replace('"', '""') + '"' for w in words)
    hits = []
    if match:
        hits = conn.execute("SELECT docid,content FROM docs WHERE docs MATCH ?" + where + " ORDER BY bm25(docs),docid LIMIT ?", [match, *excluded, k]).fetchall()
    if len(hits) < k:
        blocked = excluded + [x[0] for x in hits]
        condition = " WHERE docid NOT IN (" + ",".join("?" for _ in blocked) + ")" if blocked else ""
        hits += conn.execute("SELECT docid,content FROM docs" + condition + " ORDER BY docid LIMIT ?", [*blocked, k - len(hits)]).fetchall()
    assert len({x[0] for x in hits}) == len(hits) and not set(x[0] for x in hits) & set(excluded)
    return hits


def project(family, rid, query, hits, qrels, metadata, enc):
    state = {"query": query, "task": "Rank candidate documents/tools by relevance to this query."}
    st = len(enc.encode(canonical(state), disallowed_special=()))
    fields = []
    oversized = []
    for i, (did, content) in enumerate(hits):
        q = {"type": "choice", "instructions": {"task": "Assess whether this candidate is relevant/useful to the query. Use the full text below; return the probability of relevance.", "candidate": content}, "criteria": {"yes": "Relevant/useful to the query.", "no": "Not relevant/useful to the query."}}
        size = len(enc.encode(canonical(q), disallowed_special=()))
        if st + size > 31000:
            oversized.append(did)
        else:
            fields.append((f"doc_{i}", did, q, size))
    chunks = []
    chunk = []
    size = st
    for field in fields:
        if chunk and size + field[3] > 60000:
            chunks.append(chunk)
            chunk = []
            size = st
        chunk.append(field)
        size += field[3]
    if chunk:
        chunks.append(chunk)
    retrieved = [d for d, _ in hits]
    positive = sum(v > 0 for v in qrels.values())
    recall = sum(qrels.get(d, 0) > 0 for d in retrieved) / positive if positive else None
    record = {"id": rid, "family": family, "qrels": qrels, "retrieved_ids": retrieved, "scorable_ids": [x[1] for x in fields], "candidate_recall": recall, "oversized_ids": oversized, "request_count": len(chunks), "metadata": metadata}
    rows = []
    for ix, chunk in enumerate(chunks):
        rows.append({"id": f"{family}:{rid}:chunk{ix}", "family": family, "split": "evaluation-only", "state": state, "questions": {f: q for f, d, q, s in chunk}, "expected": {f: "yes" if qrels.get(d, 0) > 0 else None for f, d, q, s in chunk}, "scoring": {"type": "retrieval_ranking", "field_to_document": {f: d for f, d, q, s in chunk}, "qrels": qrels, "retrieved_ids": retrieved, "scorable_ids": record["scorable_ids"], "candidate_recall": recall, "scorer": "evaluation.adapters.retrieval_ranking:score_query", "combine_chunks_by": "metadata.group_id"}, "metadata": {**metadata, "group_id": rid, "candidate_count": len(hits), "context_omitted_ids": oversized, "context_estimator": "cl100k_base;31Kstate+field/60Ktotal; fulltext only", "protocol": "Null means unjudged, never score as verified-negative classification. For conventional ranking metrics only, unjudged relevance=0. No gold insertion."}})
    return rows, record


def retrieval(layout, candidates=32, families=("ToolRet-retrieval", "BRIGHT-retrieval"), log=print):
    import tiktoken

    assert 2 <= candidates <= 255
    enc = tiktoken.get_encoding("cl100k_base")
    out = layout.normalized
    req = layout.requests
    side = layout.suite / "retrieval-queries"
    side.mkdir(exist_ok=True)
    cfg = ast.parse((layout.repos / "toolret/toolret/config.py").read_text())
    mapping = next(ast.literal_eval(n.value) for n in cfg.body if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "_TASK_2_CATEGORY" for t in n.targets))
    indexes = {}
    source_hashes = {}
    report = {}
    for family in families:
        stats = collections.Counter()
        recalls = []
        perdomain = collections.Counter()
        with (out / (family + ".jsonl")).open("w") as f, (req / (family + ".jsonl")).open("w") as g, (side / (family + ".jsonl")).open("w") as h:
            paths = sorted((layout.raw / "toolret_queries").glob("*/*.parquet")) if family.startswith("ToolRet") else sorted((layout.raw / "bright/examples").glob("*.parquet"))
            for p in paths:
                domain = p.parent.name if family.startswith("ToolRet") else p.stem.replace("-00000-of-00001", "")
                category = mapping[domain] if family.startswith("ToolRet") else domain
                index_name = ("toolret_" if family.startswith("ToolRet") else "bright_") + category
                if index_name not in indexes:
                    corpora = sorted((layout.raw / "toolret_tools" / category).glob("*.parquet")) if family.startswith("ToolRet") else [layout.raw / "bright/documents" / p.name]
                    assert corpora and all(x.exists() for x in corpora)
                    indexes[index_name], source_hashes[index_name] = make_index(layout, index_name, corpora, "documentation" if family.startswith("ToolRet") else "content")
                conn = indexes[index_name]
                checksum = sha256(p)
                for q in parquet_rows(p):
                    excluded = set(str(x) for x in q.get("excluded_ids", []) or [])
                    qrels = {str(x["id"]): 1 for x in json.loads(q["labels"])} if family.startswith("ToolRet") else {str(x): 1 for x in q["gold_ids"]}
                    assert not excluded & set(qrels), "Official positives/exclusion conflict must be reviewed"
                    hits = retrieve(conn, q["query"], excluded, candidates)
                    rid = domain + ":" + str(q["id"])
                    metadata = {"domain": domain, "category": category, "source": layout.rel(p), "source_sha256": checksum, "corpus_index": index_name}
                    rows, record = project(family, rid, q["query"], hits, qrels, metadata, enc)
                    record["lexical_baseline"] = score_query(record, {d: 1 - i / (len(hits) + 1) for i, (d, c) in enumerate(hits)})
                    h.write(canonical(record) + "\n")
                    for r in rows:
                        f.write(canonical(r) + "\n")
                        g.write(canonical({"model": REFERENCE_MODEL, "state": r["state"], "questions": r["questions"]}) + "\n")
                    stats["queries"] += 1
                    stats["requests"] += len(rows)
                    stats["fields"] += sum(len(r["questions"]) for r in rows)
                    stats["oversized_candidates"] += len(record["oversized_ids"])
                    stats["queries_without_requests"] += not rows
                    perdomain[domain] += 1
                    if record["candidate_recall"] is not None:
                        recalls.append(record["candidate_recall"])
                    if stats["queries"] % 200 == 0:
                        log(json.dumps({"family": family, "queries": stats["queries"], "requests": stats["requests"]}))
                if family.startswith("BRIGHT"):
                    indexes.pop(index_name).close()
            report[family] = {**dict(stats), "queries_by_domain": dict(perdomain), "mean_candidate_recall": sum(recalls) / len(recalls) if recalls else None}
        log(json.dumps({"family_complete": family, **report[family]}))
    for conn in indexes.values():
        conn.close()
    report.update(candidate_policy=f"Query-only SQLiteFTS5(porter unicode61) BM25 top{candidates}; stopwords/repetition removed; excludedIDs filtered before ranking; deterministicIDfallback; no gold insertion.", corpus_hashes=source_hashes, context_policy="No truncation. Unscorable complete documents retained in query sidecar coverage, omitted from requests. cl100k estimate, not certified Jev fit.", scoring="Probability P(yes) rankings combined across chunks. nDCG@10/MRR/Recall@10 plus candidate recall/coverage. Unjudged=0 for ranking convention only, not binary classification gold.")
    (layout.suite / "retrieval-ranking-audit.json").write_text(json.dumps(report, indent=2) + "\n")
    return {k: v for k, v in report.items() if k in families}


def toolret(layout):
    return retrieval(layout, families=("ToolRet-retrieval",))


def bright(layout):
    return retrieval(layout, families=("BRIGHT-retrieval",))


BUILDERS = {2: toolret, 36: bright}
