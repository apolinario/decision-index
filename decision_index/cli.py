import argparse
import json
import os
import sys
from pathlib import Path

from decision_index import constants as C


def parse_value(v):
    try:
        return json.loads(v)
    except ValueError:
        return v


def engine_options(args):
    opts = {}
    if getattr(args, "model", None):
        opts["model"] = args.model
    for item in args.option or []:
        k, _, v = item.partition("=")
        opts[k] = parse_value(v)
    return opts


def ensure_suite(args):
    from decision_index.suite.download import download
    from decision_index.suite.io import Suite

    directory = Path(args.suite_dir)
    if not (directory / C.SUITE_ROWS_FILE).exists():
        print(json.dumps(download(directory, dataset=args.suite_dataset)), flush=True)
    return Suite(directory)


def cmd_suite(args):
    from decision_index.suite.io import Suite

    if args.suite_command == "download":
        from decision_index.suite.download import download

        print(json.dumps(download(args.dir, dataset=args.dataset, revision=args.revision, verify=not args.no_verify), indent=2))
    elif args.suite_command == "verify":
        print(json.dumps(Suite(args.dir).verify(strict=False), indent=2))
    elif args.suite_command == "import":
        from decision_index.suite.download import from_local

        print(json.dumps(from_local(args.dir, args.rows, args.exclusions, args.manifest, verify=not args.no_verify), indent=2))
    elif args.suite_command == "sample":
        from decision_index.suite.sample import stratified_sample

        print(json.dumps(stratified_sample(Suite(args.dir), args.n, args.out, seed=args.seed, complete_groups=not args.single_requests), indent=2))
    elif args.suite_command == "rebuild":
        from decision_index.suite.build import rebuild

        print(json.dumps(rebuild.main(args.work, only=args.only, skip_download=args.skip_download, skip_normalize=args.skip_normalize, compare=args.compare), indent=2))
    else:
        raise SystemExit("unknown suite command")


def cmd_run(args):
    from decision_index.runner import run
    from decision_index.suite.io import sha256_file

    if args.rows:
        rows = Path(args.rows)
        corpus = None
    else:
        suite = ensure_suite(args)
        rows = suite.rows_path
        corpus = suite.verify(strict=not args.no_verify)["sha256"]
    out = Path(args.out or ("runs/" + args.engine))
    print(json.dumps(run(args.engine, engine_options(args), rows, out, limit=args.limit, compact=args.compact, resume=not args.fresh, corpus_sha256=corpus)))


def cmd_score(args):
    from decision_index.pipeline import score_run
    from decision_index.suite.io import Suite

    suite = Suite(args.suite_dir)
    results = Path(args.results)
    out = Path(args.out or results.parent)
    scores = score_run(suite, results, args.engine or results.parent.name, out, reference_results=args.reference)
    print(json.dumps({"decision_index": scores["decision_index"], "scores": scores["scores"], "areas": [{k: a[k] for k in ("id", "skill", "raw", "coverage", "n")} for a in scores["areas"]], "completed": scores["completed"], "complete": scores["complete"], "out": str(out)}, indent=2))


def cmd_pipeline(args):
    from decision_index.pipeline import score_run, upload_run
    from decision_index.runner import run

    suite = ensure_suite(args)
    corpus = suite.verify(strict=not args.no_verify)["sha256"]
    out = Path(args.out or ("runs/" + args.engine))
    rows = Path(args.rows) if args.rows else suite.rows_path
    run(args.engine, engine_options(args), rows, out, limit=args.limit, compact=args.compact, resume=not args.fresh, corpus_sha256=corpus)
    scores = score_run(suite, out / "results.jsonl", args.engine, out)
    print(json.dumps({"decision_index": scores["decision_index"], "scores": scores["scores"], "complete": scores["complete"]}, indent=2))
    if args.upload:
        print(upload_run(args.upload, out, path_in_repo=args.upload_path, private=not args.public))


def cmd_hf_job(args):
    from decision_index.hf_job import submit

    engine_args = []
    if args.model:
        engine_args += ["--model", args.model]
    for o in args.option or []:
        engine_args += ["--option", o]
    extra = []
    if args.limit:
        extra += ["--limit", str(args.limit)]
    if args.compact:
        extra += ["--compact"]
    print(json.dumps(submit(args.results_repo, args.engine, engine_args, args.suite_dataset, args.run_name or args.engine, extra, flavor=args.flavor, image=args.image, timeout=args.timeout, git_url=args.git_url, namespace=args.namespace, private=not args.public, dry_run=args.dry_run), indent=2))


def add_engine_args(p):
    p.add_argument("--engine", required=True, help="engine name (http, transformers, random) or module:Class")
    p.add_argument("--model", help="model id for the transformers engine")
    p.add_argument("--option", action="append", help="engine option key=value (JSON values accepted), repeatable")


def add_suite_args(p):
    p.add_argument("--suite-dir", default="suite")
    p.add_argument("--suite-dataset", default=C.SUITE_DATASET)
    p.add_argument("--no-verify", action="store_true")


def build_parser():
    ap = argparse.ArgumentParser(prog="decision-index", description="Decision Index reproduction kit")
    sub = ap.add_subparsers(dest="command", required=True)

    s = sub.add_parser("suite", help="obtain, verify, sample or rebuild the frozen suite")
    ss = s.add_subparsers(dest="suite_command", required=True)
    d = ss.add_parser("download")
    d.add_argument("--dir", default="suite")
    d.add_argument("--dataset", default=C.SUITE_DATASET)
    d.add_argument("--revision")
    d.add_argument("--no-verify", action="store_true")
    v = ss.add_parser("verify")
    v.add_argument("--dir", default="suite")
    i = ss.add_parser("import")
    i.add_argument("--dir", default="suite")
    i.add_argument("--rows", required=True)
    i.add_argument("--exclusions")
    i.add_argument("--manifest")
    i.add_argument("--no-verify", action="store_true")
    m = ss.add_parser("sample")
    m.add_argument("--dir", default="suite")
    m.add_argument("--n", type=int, default=100)
    m.add_argument("--out", default="sample-rows.jsonl.gz")
    m.add_argument("--seed", type=int, default=C.FREEZE_SEED)
    m.add_argument("--single-requests", action="store_true")
    r = ss.add_parser("rebuild")
    r.add_argument("--work", default="work")
    r.add_argument("--only", type=int, nargs="*")
    r.add_argument("--skip-download", action="store_true")
    r.add_argument("--skip-normalize", action="store_true")
    r.add_argument("--compare", help="path to a reference selected-rows.jsonl(.gz) to compare run ids and payload hashes")
    s.set_defaults(func=cmd_suite)

    p = sub.add_parser("run", help="run an engine over the suite with checkpoint/resume")
    add_engine_args(p)
    add_suite_args(p)
    p.add_argument("--rows", help="run over this rows file instead of the full suite")
    p.add_argument("--out")
    p.add_argument("--limit", type=int)
    p.add_argument("--compact", action="store_true", help="omit payload and raw_output from results rows")
    p.add_argument("--fresh", action="store_true", help="ignore an existing results.jsonl")
    p.set_defaults(func=cmd_run)

    p = sub.add_parser("score", help="score a results.jsonl and compute the Decision Index")
    p.add_argument("--results", required=True)
    p.add_argument("--suite-dir", default="suite")
    p.add_argument("--engine")
    p.add_argument("--out")
    p.add_argument("--reference", help="optional reference results.jsonl scored on the same complete cases")
    p.set_defaults(func=cmd_score)

    p = sub.add_parser("pipeline", help="suite -> run -> score -> index (-> upload) in one go")
    add_engine_args(p)
    add_suite_args(p)
    p.add_argument("--rows")
    p.add_argument("--out")
    p.add_argument("--limit", type=int)
    p.add_argument("--compact", action="store_true")
    p.add_argument("--fresh", action="store_true")
    p.add_argument("--upload", help="Hub dataset repo to receive results and scores")
    p.add_argument("--upload-path", default="")
    p.add_argument("--public", action="store_true")
    p.set_defaults(func=cmd_pipeline)

    p = sub.add_parser("hf-job", help="submit one Hugging Face Job running the whole pipeline")
    add_engine_args(p)
    p.add_argument("--results-repo", required=True, help="Hub dataset repo that receives code snapshot, results and scores")
    p.add_argument("--suite-dataset", default=C.SUITE_DATASET)
    p.add_argument("--run-name")
    p.add_argument("--flavor", default="rtx-pro-6000")
    p.add_argument("--image", default="pytorch/pytorch:2.8.0-cuda12.8-cudnn9-runtime")
    p.add_argument("--timeout", default="72h")
    p.add_argument("--git-url", help="install the kit from this git URL instead of uploading a source snapshot")
    p.add_argument("--namespace")
    p.add_argument("--limit", type=int)
    p.add_argument("--compact", action="store_true")
    p.add_argument("--public", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    p.set_defaults(func=cmd_hf_job)
    return ap


def main(argv=None):
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    args = build_parser().parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
