import os
import shutil
import subprocess
import urllib.request
import zipfile
from pathlib import Path

from decision_index.suite.build.layout import sha256

GIT = {
    "bfcl": ("https://github.com/gorilla-llm/gorilla.git", "916260dfc116bf06793a1af79b4ec8195b0453b6", ["berkeley-function-call-leaderboard/data"]),
    "toolret": ("https://github.com/mangopy/tool-retrieval-benchmark.git", "c4181d914a227134705ecb6bab13fbd92ccd2938", None),
    "apibank": ("https://github.com/AlibabaResearch/DAMO-ConvAI.git", "188835d4f9948563a6b9c8ac50cd0f3ae4021ed6", ["api-bank"]),
    "sgd": ("https://github.com/google-research-datasets/dstc8-schema-guided-dialogue.git", "e852981ae34990f4358979625854259302feaa78", ["test"]),
    "contractnli": ("https://github.com/stanfordnlp/contract-nli.git", "eced6528dd3c1d14d73f9a87df8f7bdbc03126f9", None),
    "pop909cl": ("https://github.com/AndyWeasley2004/POP909-CL-Dataset.git", "be9094392903c471a930519e1c0bacf8b6be5d62", ["POP909_processed"]),
    "searchless_chess": ("https://github.com/google-deepmind/searchless_chess.git", "90ae0e6b121673fc3079aaeffa047580bb600c0a", None),
    "esci": ("https://github.com/amazon-science/esci-data.git", "7916cdf6ab75a462e77f20ab40428a10923998d5", None),
    "acos": ("https://github.com/NUSTM/ACOS.git", "45d179a3dcc6a3dedd848d81b16f2552454805fe", ["data"]),
    "finentity": ("https://github.com/yixuantt/FinEntity.git", "3b6cedc5485b669c2ed168f1d949f517636eb7b8", ["data"]),
    "isarcasm": ("https://github.com/iabufarha/iSarcasmEval.git", "dfc708b53bde1bb571abfb5692f63231c2232195", None),
    "nli4ct": ("https://github.com/ai-systems/Task-2-SemEval-2024.git", "7f32fa6c7db43e577f22e9fc6c28eef1cc6d223e", None),
    "cruxeval": ("https://github.com/facebookresearch/cruxeval.git", "190faf16d175b5847b0af05d937872b1fb395942", ["data", "samples"]),
    "forecastbench-datasets": ("https://github.com/forecastingresearch/forecastbench-datasets.git", "da48cfb37db3e674fd57030266c92009873be4b6", ["datasets/question_sets", "datasets/resolution_sets"]),
    "habermas_machine": ("https://github.com/google-deepmind/habermas_machine.git", "7923b71966c14136077c78d7841bc9e1a182dfe0", None),
    "sata": ("https://github.com/sata-bench/sata-bench.git", "371dd0c18fe75a96fbbcf2d1507ceeaf0d5263c5", None),
}
GIT_FLAT = {
    "musr": ("https://github.com/Zayne-sprague/MuSR.git", "b1f4d4168a9cfc6760e8b74d728e4516023dfaa5", ["datasets"]),
    "simplebench": ("https://github.com/simple-bench/SimpleBench.git", "fbc2e429085bdedad7d1a236d2bc9bc18c95f16e", None),
    "cladder": ("https://github.com/causalNLP/cladder.git", "3d2d1169b4b939a09048a6a75956c8972a93cc38", ["data"]),
    "vast": ("https://github.com/emilyallaway/zero-shot-stance.git", "e7c4775182b184730f350995f8260579c9e066fe", ["data/VAST"]),
}
HF = {
    "raw/banking77": ("PolyAI/banking77", "90d4e2ee5521c04fc1488f065b8b083658768c57", ["test.csv", "categories.json"]),
    "raw/routerbench": ("withmartian/routerbench", "784021482c3f320c6619ed4b3bb3b41a21424fcb", ["routerbench_0shot.pkl", "routerbench_5shot.pkl"]),
    "raw/bright": ("xlangai/BRIGHT", "3066d29c9651a576c8aba4832d249807b181ecae", ["examples/*.parquet", "documents/*.parquet"]),
    "raw/toolret_queries": ("mangopy/ToolRet-Queries", "b8c76ad3349ff17497b6bdb28bb5b8f61a0f6445", ["*/*.parquet"]),
    "raw/toolret_tools": ("mangopy/ToolRet-Tools", "e06c38c75612b6536bd959e08cdd345894aba6a7", ["*/*.parquet"]),
    "raw/hle": ("cais/hle", "5a81a4c7271a2a2a312b9a690f0c2fde837e4c29", ["data/test-00000-of-00001.parquet"]),
    "raw/anli": ("facebook/anli", "8e4813d81f46d313dac7892e1c28076917cfcdf9", ["plain_text/test_*.parquet"]),
    "data/sources/mmlu": ("cais/mmlu", "c30699e8356da336a370243923dbaf21066bb9fe", ["all/test-00000-of-00001.parquet"]),
    "data/sources/arc": ("allenai/ai2_arc", None, ["ARC-Easy/test-00000-of-00001.parquet", "ARC-Challenge/test-00000-of-00001.parquet"]),
    "data/sources/winogrande": ("allenai/winogrande", "01e74176c63542e6b0bcb004dcdea22d94fb67b5", ["winogrande_xl/validation-00000-of-00001.parquet"]),
    "data/sources/hellaswag": ("Rowan/hellaswag", "218ec52e09a7e7462a5400043bb9a69a41d06b76", ["data/validation-00000-of-00001.parquet"]),
    "data/sources/gsm8k": ("openai/gsm8k", "740312add88f781978c0658806c59bc2815b9866", ["main/test-00000-of-00001.parquet"]),
}
HTTP = {
    "raw/downloads/BPoMP_p1.json": ("https://zenodo.org/api/records/7299879/files/BPoMP_datasets_p1_out_of_3.json/content", "2500e828fd3b38c43f869ce3c1d2defd991adf340ecb5757c4e9097d7d7a2925"),
    "raw/downloads/BPoMP_p2.json": ("https://zenodo.org/api/records/7299879/files/BPoMP_datasets_p2_out_of_3.json/content", "ee17bc0fad97f58da19186d21773bb6b85472aacedf1400865358251140270b5"),
    "raw/downloads/BPoMP_p3.json": ("https://zenodo.org/api/records/7299879/files/BPoMP_datasets_p3_out_of_3.json/content", "cc0b75645a28cce5a6fe15bd2dfe7878a6da4a0bc6b5842dfc0a1b677ec99f3b"),
    "raw/downloads/cfcolor.zip": ("https://www.dgp.toronto.edu/~donovan/cfcolor/cfcolor.zip", "47c07095642cfab3c2eeab366a5d152b07783cbb5af390cbfda7d7c13db4b54c"),
    "raw/downloads/chessbench-test-action-value.bag": ("https://storage.googleapis.com/searchless_chess/data/test/action_value_data.bag", "5f73aac8f60e31734cdbf276ba3fca8d5ba5cb6171ba600e31af4f36327986b0"),
    "raw/downloads/humicroedit-full.zip": ("https://cs.rochester.edu/u/nhossain/semeval-2020-task-7-dataset.zip", "12a6cbf28c8b698ad80be42a65ac867b57e4c71662eedab607805e167ba791ab"),
    "raw/clinc150/data_full.json": ("https://raw.githubusercontent.com/clinc/oos-eval/master/data/data_full.json", "36923c3705a59e08fe9c3883d8bc2dd966ef93e22cb78ac41171782a698d56e0"),
    "data/sources/gpqa/dataset.zip": ("https://github.com/idavidrein/gpqa/raw/main/dataset.zip", "461ae7329f15a3e35f8184d2dac24b990f34fdf12f366ca4062d8e6638cd08dc"),
}
ESCI_LFS = {
    "shopping_queries_dataset/shopping_queries_dataset_examples.parquet": "4a735b693b4a424a6fc67f5be6e4c811495c488bbf66d02a602d308b2744263a",
    "shopping_queries_dataset/shopping_queries_dataset_products.parquet": "25124442d064d64b26f74082d6fa09438d679efc0c18",
}
NEEDS = {
    1: ["git:bfcl"], 2: ["git:toolret", "hf:raw/toolret_queries", "hf:raw/toolret_tools"], 3: ["git:apibank"], 4: ["hf:raw/banking77"],
    5: ["http:raw/clinc150/data_full.json"], 6: ["hf:raw/routerbench"], 9: [], 10: ["git:sgd"], 11: ["git:contractnli"], 12: ["hf:raw/anli"],
    20: ["http:raw/downloads/BPoMP_p1.json", "http:raw/downloads/BPoMP_p2.json", "http:raw/downloads/BPoMP_p3.json"],
    21: ["http:raw/downloads/humicroedit-full.zip"], 22: ["git:pop909cl"], 23: ["http:raw/downloads/cfcolor.zip"],
    24: ["hf:data/sources/mmlu"], 25: ["http:data/sources/gpqa/dataset.zip"], 26: ["hf:data/sources/arc"], 27: ["hf:data/sources/arc"],
    28: ["hf:data/sources/winogrande"], 29: ["hf:data/sources/hellaswag"], 30: ["hf:data/sources/gsm8k"],
    31: ["git:searchless_chess", "http:raw/downloads/chessbench-test-action-value.bag"], 32: ["gitflat:musr"], 33: ["git:sata"],
    34: ["gitflat:simplebench"], 36: ["hf:raw/bright"], 37: ["git:esci"], 38: ["git:acos"], 39: ["git:finentity"], 40: ["git:isarcasm"],
    41: ["gitflat:vast"], 42: ["git:nli4ct"], 43: ["git:cruxeval"], 44: ["gitflat:cladder"], 45: ["hf:raw/hle"],
    48: ["git:forecastbench-datasets"], 50: ["git:habermas_machine"],
}


def git_fetch(dest, url, revision, sparse=None, log=print):
    dest = Path(dest)
    if (dest / ".git").exists():
        head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=dest, text=True).strip()
        if head == revision:
            return
    dest.mkdir(parents=True, exist_ok=True)
    if not (dest / ".git").exists():
        subprocess.run(["git", "init", "-q"], cwd=dest, check=True)
        subprocess.run(["git", "remote", "add", "origin", url], cwd=dest, check=True)
    if sparse:
        subprocess.run(["git", "sparse-checkout", "set", "--no-cone", *sparse, "LICENSE*", "README*", "*.md", "*.txt", "*.py"], cwd=dest, check=True)
    log(f"fetching {url} @ {revision[:10]}")
    subprocess.run(["git", "fetch", "-q", "--depth", "1", "origin", revision], cwd=dest, check=True)
    subprocess.run(["git", "checkout", "-q", "FETCH_HEAD"], cwd=dest, check=True)


def http_fetch(dest, url, expected_sha, log=print):
    dest = Path(dest)
    if dest.exists() and sha256(dest) == expected_sha:
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    log(f"downloading {url}")
    tmp = dest.with_suffix(dest.suffix + ".part")
    with urllib.request.urlopen(url, timeout=600) as r, tmp.open("wb") as f:
        shutil.copyfileobj(r, f)
    actual = sha256(tmp)
    if actual != expected_sha:
        raise ValueError(f"{url}: sha256 {actual} differs from the pinned {expected_sha}")
    tmp.replace(dest)


def hf_fetch(dest, repo, revision, patterns, log=print):
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    from huggingface_hub import snapshot_download

    log(f"snapshot {repo} @ {revision or 'main'}")
    snapshot_download(repo, repo_type="dataset", revision=revision, allow_patterns=patterns, local_dir=str(dest))


def acquire(layout, numbers=None, log=print):
    numbers = sorted(NEEDS) if numbers is None else numbers
    done = set()
    for n in numbers:
        for spec in NEEDS.get(n, []):
            if spec in done:
                continue
            done.add(spec)
            kind, _, key = spec.partition(":")
            if kind == "git":
                url, rev, sparse = GIT[key]
                git_fetch(layout.repos / key, url, rev, sparse, log)
                if key == "esci":
                    for rel, sha in ESCI_LFS.items():
                        target = layout.repos / key / rel
                        if target.exists() and sha256(target).startswith(sha[:40]):
                            continue
                        http_fetch_prefix(target, f"https://media.githubusercontent.com/media/amazon-science/esci-data/{rev}/{rel}", sha, log)
            elif kind == "gitflat":
                url, rev, sparse = GIT_FLAT[key]
                git_fetch(layout.raw / key, url, rev, sparse, log)
            elif kind == "hf":
                repo, rev, patterns = HF[key]
                hf_fetch(layout.root / key, repo, rev, patterns, log)
            elif kind == "http":
                url, sha = HTTP[key]
                http_fetch(layout.root / key, url, sha, log)
    if 23 in numbers:
        release = layout.repos / "cfcolor/release"
        if not (release / "themeData.mat").exists():
            with zipfile.ZipFile(layout.downloads / "cfcolor.zip") as z:
                for member in ("release/allMTurkRatings.mat", "release/themeData.mat"):
                    z.extract(member, layout.repos / "cfcolor")
    if 43 in numbers:
        samples = layout.repos / "cruxeval/samples"
        for name in ("model_generations", "evaluation_results"):
            if not (samples / name).exists() and (samples / (name + ".zip")).exists():
                with zipfile.ZipFile(samples / (name + ".zip")) as z:
                    z.extractall(samples)


def http_fetch_prefix(dest, url, sha_prefix, log=print):
    dest.parent.mkdir(parents=True, exist_ok=True)
    log(f"downloading {url}")
    tmp = dest.with_suffix(dest.suffix + ".part")
    with urllib.request.urlopen(url, timeout=1800) as r, tmp.open("wb") as f:
        shutil.copyfileobj(r, f)
    actual = sha256(tmp)
    if not actual.startswith(sha_prefix[:40]):
        raise ValueError(f"{url}: sha256 {actual} differs from the pinned prefix {sha_prefix}")
    tmp.replace(dest)
