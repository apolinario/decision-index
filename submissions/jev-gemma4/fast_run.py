"""Batched ("64 concurrent requests") driver for the JEV engine over the Decision Index 0.2.1 suite.

The kit's runner sends one request at a time. This driver takes the next N requests (default 64), renders every
question of all of them with the same engine code (jev_engine.JevEngine: template, option rendering, noul phrasing,
>16-option groups + final), runs all prompts through the backbone together in length-sorted micro-batches, and writes
one result row per request in the kit's results.jsonl format (compact: no payload / raw_output). The response of
every request is checked with decision_index.engines.validate. Scores come from the kit's own `score` command.

Differences from the sequential runner, by design: prompts of different requests share micro-batches (bf16 kernels
can differ in the last bits with batch shape), and total_wall_ms is the batch wall time divided by the batch size.
"""
import argparse, collections, json, os, sys, time, traceback
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # jev_engine.py next to this file
import torch  # noqa: E402

from decision_index.engines.base import Unsupported, validate  # noqa: E402
from decision_index.runner import iter_rows  # noqa: E402
from decision_index.suite.io import Suite, atomic_json, dumps, read_jsonl  # noqa: E402
from jev_engine import (MAX_CHOICE, NEEDED, SLOTS, JevEngine, as_text, combine, noul_question, option_text, pick_finalists,  # noqa: E402
                        plan_groups, render)


def stamp():
    return datetime.now(timezone.utc).isoformat()


def plan_request(state, questions):
    st = as_text(state) if state is not None else ""
    jobs, plans = [], {}
    for key, q in questions.items():
        qtext = as_text(q.get("instructions", ""))
        if q["type"] == "noul":
            jobs.append((key, "noul", render("noul", st, noul_question(q.get("instructions", "")), []), None))
        elif q["type"] == "choice":
            keys = list(q["criteria"])
            if len(keys) < 2:
                raise Unsupported("choice needs at least 2 options")
            texts = [option_text(k, q["criteria"][k]) for k in keys]
            groups = [list(range(len(keys)))] if len(keys) <= MAX_CHOICE else plan_groups(len(keys))
            plans[key] = (keys, texts, groups, qtext)
            for gi, g in enumerate(groups):
                jobs.append((key, "choice", render("choice", st, qtext, [texts[o] for o in g]), (gi, g)))
        else:
            raise Unsupported(f"unsupported question type {q['type']!r}")
    return st, jobs, plans


LONG = 16384

class VllmBackend:
    """Same prompts, head read-out and temperatures as JevEngine, with the forward passes run by vLLM on the published
    adapter_vllm/ (backbone LoRA + the 24-slot head as an lm_head LoRA). Each prompt is a 1-token completion restricted
    to its option tokens; logprob + head bias gives the head logits up to a per-prompt constant, which cancels in the
    softmax over the active slots (same as the model card's vLLM recipe)."""

    def __init__(self, model, revision, gpu_mem=0.85, max_model_len=131072, prefix_caching=True, max_num_seqs=256):
        from huggingface_hub import HfApi, snapshot_download
        from transformers import AutoTokenizer
        from vllm import LLM
        from vllm.lora.request import LoRARequest
        sha = HfApi().model_info(model, revision=revision).sha
        d = snapshot_download(model, revision=sha, allow_patterns=NEEDED + ["adapter_vllm/*"])
        self.tok = AutoTokenizer.from_pretrained(d)
        dh = json.load(open(f"{d}/adapter_vllm/decision_head.json"))
        self.bias, self.vids = dh["bias"], dh["verbalizer_ids"]
        self.T = json.load(open(f"{d}/calibration.json"))["per_kind"]
        r = json.load(open(f"{d}/adapter_vllm/adapter_config.json"))["r"]
        kw = dict(enable_prefix_caching=True, mamba_cache_mode="align") if prefix_caching else dict(enable_prefix_caching=False)
        self.llm = LLM(model=d, dtype="bfloat16", enable_lora=True, max_lora_rank=r, max_loras=1,
                       logprobs_mode="processed_logprobs", gpu_memory_utilization=gpu_mem, max_model_len=max_model_len,
                       max_num_seqs=max_num_seqs, **kw)
        self.lora = LoRARequest("jev-decision", 1, f"{d}/adapter_vllm")
        self.max_len = max_model_len - 8
        self.provenance = {"kind": "trained", "repo": model, "revision": sha, "local_dir": d,
                           "path": "System 1 via vLLM: bf16 backbone + adapter_vllm (LoRA + head as lm_head LoRA) + calibration.json",
                           "prefix_caching": prefix_caching, "max_model_len": max_model_len}

    def logits(self, ids, specs):
        from vllm import SamplingParams, TokensPrompt
        prompts, params = [], []
        for x, (kind, n) in zip(ids, specs):
            s = SLOTS[kind][0]
            prompts.append(TokensPrompt(prompt_token_ids=x))
            params.append(SamplingParams(max_tokens=1, temperature=1.0, allowed_token_ids=self.vids[s:s + n], logprobs=n))
        outs = self.llm.generate(prompts, params, lora_request=self.lora, use_tqdm=False)
        res = []
        for o, (kind, n) in zip(outs, specs):
            s = SLOTS[kind][0]
            lp = o.outputs[0].logprobs[0]
            z = torch.full((24,), float("-inf"))
            for i, t in enumerate(self.vids[s:s + n]):
                z[s + i] = (lp[t].logprob if t in lp else -1e9) + self.bias[s + i]
            res.append(z)
        return res

    def _probs(self, kind, z, n):
        s = SLOTS[kind][0]
        return torch.softmax(z[s:s + n] / float(self.T[kind]), dim=0).tolist()

    def runtime(self):
        import vllm
        return {"torch": torch.__version__, "vllm": vllm.__version__, "gpu": torch.cuda.get_device_name(0)}

    def synchronize(self):
        pass

    def warmup(self):
        pass


def forward(eng, ids, specs):
    if isinstance(eng, VllmBackend):
        return eng.logits(ids, specs)
    out = [None] * len(ids)
    short = [i for i in range(len(ids)) if len(ids[i]) <= LONG]
    for i in range(len(ids)):  # long prompts alone: no padding (as the sequential engine runs them)
        if len(ids[i]) > LONG:
            out[i] = eng._logits_ids([ids[i]])[0]
    for i, z in zip(short, eng._logits_ids([ids[i] for i in short]) if short else []):
        out[i] = z
    return out



def run_batch(eng, rows):
    """Returns [(status, response_or_error)] for rows, answering all of them with shared forward passes."""
    out = [None] * len(rows)
    live = []  # (row index, state text, jobs, plans, token ids per job)
    for i, row in enumerate(rows):
        try:
            st, jobs, plans = plan_request(row["state"], row["questions"])
            ids = [eng.tok.encode(j[2], add_special_tokens=False) for j in jobs]
            if any(len(x) > eng.max_len for x in ids):
                raise Unsupported(f"prompt of {max(map(len, ids))} tokens exceeds the {eng.max_len}-token context window")
            live.append((i, st, jobs, plans, ids))
        except Unsupported as e:
            out[i] = ("unsupported", str(e))
    flat = [(li, j) for li, rec in enumerate(live) for j in range(len(rec[2]))]
    per = collections.defaultdict(dict)
    spec = lambda job: ("noul", 2) if job[1] == "noul" else ("choice", len(job[3][1]))
    zs = forward(eng, [live[li][4][j] for li, j in flat], [spec(live[li][2][j]) for li, j in flat])
    for (li, j), z in zip(flat, zs):
        per[li][j] = z
    finals = []  # (li, key, finalists, token ids)
    state = {}
    for li, (i, st, jobs, plans, ids) in enumerate(live):
        answers, in_group = {}, {k: {} for k in plans}
        for j, (key, kind, _, extra) in enumerate(jobs):
            z = per[li][j]
            if kind == "noul":
                answers[key] = {"type": "noul", "noul": eng._probs("noul", z, 2)[1]}
            else:
                _, g = extra
                for o, p in zip(g, eng._probs("choice", z, len(g))):
                    in_group[key][o] = p
        for key, (keys, texts, groups, qtext) in plans.items():
            if len(groups) > 1:
                fin = pick_finalists(groups, in_group[key])
                finals.append((li, key, fin, eng.tok.encode(render("choice", st, qtext, [texts[o] for o in fin]), add_special_tokens=False)))
        state[li] = (answers, in_group)
    final_probs = {}
    if finals:
        for (li, key, fin, _), z in zip(finals, forward(eng, [f[3] for f in finals], [("choice", len(f[2])) for f in finals])):
            final_probs[(li, key)] = (fin, dict(zip(fin, eng._probs("choice", z, len(fin)))))
    for li, (i, st, jobs, plans, ids) in enumerate(live):
        answers, in_group = state[li]
        for key, (keys, texts, groups, qtext) in plans.items():
            if len(groups) == 1:
                p = in_group[key]
            else:
                fin, fp = final_probs[(li, key)]
                p = combine(groups, in_group[key], fin, fp)
            probs = {keys[o]: float(v) for o, v in p.items()}
            tot = sum(probs.values())
            probs = {k: v / tot for k, v in probs.items()}
            answers[key] = {"type": "choice", "choice": max(probs, key=probs.get), "probabilities": probs}
        out[i] = ("ok", {"model": eng.provenance["repo"], "answers": {k: answers[k] for k in rows[i]["questions"]}})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--revision", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--token-budget", type=int, default=131072)
    ap.add_argument("--suite-dir", default="suite-0.2")
    ap.add_argument("--rows", help="optional rows file instead of the suite (for tests)")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--backend", choices=["hf", "vllm"], default="hf")
    ap.add_argument("--gpu-mem", type=float, default=0.85)
    ap.add_argument("--max-model-len", type=int, default=131072)
    ap.add_argument("--no-prefix-caching", action="store_true")
    ap.add_argument("--max-num-seqs", type=int, default=256)
    ap.add_argument("--local-dir", default=None, help="load the bundle from this directory instead of the Hub")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    def event(**kw):
        rec = {"time": stamp(), **kw}
        print(dumps(rec), flush=True)
        atomic_json(out / "status.json", rec, indent=None)

    torch.manual_seed(20260919)
    t0 = time.perf_counter()
    event(event="loading", engine="jev_engine:JevEngine (batched driver)")
    if a.backend == "vllm":
        eng = VllmBackend(a.model, a.revision, a.gpu_mem, a.max_model_len, not a.no_prefix_caching, a.max_num_seqs)
    else:
        eng = JevEngine(model=a.model, revision=a.revision, local_dir=a.local_dir, token_budget=a.token_budget, pad_mask=False)
    if a.rows:
        rows_path, keep, corpus = a.rows, None, None
    else:
        suite = Suite(a.suite_dir, "0.2.1")
        rows_path, keep = suite.row_paths, suite.in_edition
        corpus = suite.verify(strict=True)["sha256"]
    atomic_json(out / "environment.json", {
        "engine": "jev_engine:JevEngine", "engine_options": {"model": a.model, "revision": a.revision, "backend": a.backend, "token_budget": a.token_budget, "pad_mask": False, "gpu_mem": a.gpu_mem, "max_model_len": a.max_model_len, "prefix_caching": not a.no_prefix_caching},
        "driver": f"di/fast_run.py batched driver: {a.batch} requests per batch, prompts of all of them in shared length-sorted micro-batches",
        "model_source": eng.provenance, **eng.runtime(), "loaded_seconds": time.perf_counter() - t0,
        "frozen_corpus_sha256": corpus, "latency": "batch wall time / batch size (not a per-request latency)"})
    eng.warmup()
    event(event="ready", engine="batched")
    done = set()
    res_path = out / "results.jsonl"
    if res_path.exists():
        done = {r["run_id"] for r in read_jsonl(res_path, complete_lines_only=True) if r["status"] != "error"}
    counts = collections.Counter()
    start = time.perf_counter()
    pending = []

    def flush(logf):
        if not pending:
            return
        t = time.perf_counter()
        started = stamp()
        try:
            results = run_batch(eng, pending)
        except Exception:  # e.g. an unsplittable OOM: fall back to one request at a time with the plain engine
            if isinstance(eng, VllmBackend):
                raise
            results = []
            for row in pending:
                try:
                    results.append(("ok", eng(row["state"], row["questions"])[0]))
                except Unsupported as e:
                    results.append(("unsupported", str(e)))
                except Exception as e:
                    results.append(("error", f"{type(e).__name__}: {e}"))
        eng.synchronize()
        per_ms = (time.perf_counter() - t) * 1000 / len(pending)
        for row, (status, payload) in zip(pending, results):
            rec = {**row["_evaluation"], "started_utc": started, "engine": "jev_engine:JevEngine"}
            if status == "ok":
                try:
                    validate(row["questions"], payload)
                    rec.update(status="ok", response=payload)
                except Exception as e:
                    rec.update(status="error", error=str(e), exception=type(e).__name__, traceback=traceback.format_exc())
            else:
                rec.update(status=status, error=payload)
            rec.update(completed_utc=stamp(), total_wall_ms=per_ms, model_request_wall_ms=per_ms)
            logf.write(dumps(rec) + "\n")
            counts[rec["status"]] += 1
        logf.flush()
        done.update(r["_evaluation"]["run_id"] for r in pending)
        pending.clear()
        event(event="progress", completed=len(done), counts=dict(counts), elapsed_seconds=round(time.perf_counter() - start, 1))

    with res_path.open("a", encoding="utf-8") as logf:
        n = 0
        for row in iter_rows(rows_path, keep):
            if row["_evaluation"]["run_id"] in done:
                continue
            pending.append(row)
            n += 1
            if len(pending) >= a.batch:
                flush(logf)
            if a.limit and n >= a.limit:
                break
        flush(logf)
    event(event="complete", completed=len(done), counts=dict(counts), elapsed_seconds=round(time.perf_counter() - start, 1))


if __name__ == "__main__":
    main()
