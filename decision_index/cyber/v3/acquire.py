"""Fetch only publisher label files reached from the collection's CTU source.

No binaries or packet captures. Downloads have explicit byte bounds and atomic
completion; receipts pin the actual downloaded publisher bytes.
"""
import html
import re
import urllib.request
from pathlib import Path

from ..common import file_hash, write_json

BASE = "https://mcfp.felk.cvut.cz/publicDatasets/"
CTU = {"46": "Virut", "48": "Sogou", "54": "Virut"}
IOT = {"8-1": "Hakai", "20-1": "Torii", "21-1": "Torii", "42-1": "Trojan", "44-1": "Mirai"}


def fetch(url, path, bound):
    path = Path(path)
    if path.exists():
        return {"url": url, "path": str(path), "bytes": path.stat().st_size, "sha256": file_hash(path)}
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".part")
    req = urllib.request.Request(url, headers={"User-Agent": "ESDB source-replay research"})
    with urllib.request.urlopen(req, timeout=45) as response, temp.open("wb") as out:
        size = 0
        for block in iter(lambda: response.read(1024 * 1024), b""):
            size += len(block)
            if size > bound:
                raise ValueError(f"Publisher file exceeds admitted byte bound: {url}")
            out.write(block)
    temp.rename(path)
    return {"url": url, "path": str(path), "bytes": size, "sha256": file_hash(path)}


def acquire(collection, log=print):
    root = Path(collection).resolve() / "external/esdb-network-v2"
    entries = []
    for number, family in CTU.items():
        prefix = BASE + f"CTU-Malware-Capture-Botnet-{number}/"
        readme = fetch(prefix + "README.md", root / f"ctu13/{number}/README.md", 100000)
        url = prefix + "detailed-bidirectional-flow-labels/"
        req = urllib.request.Request(url, headers={"User-Agent": "ESDB source-replay research"})
        with urllib.request.urlopen(req, timeout=45) as r:
            page = r.read(100000).decode()
        names = sorted(set(html.unescape(x) for x in re.findall(r'href="([^"/]+\.binetflow)"', page)))
        if len(names) != 1:
            raise ValueError(f"Expected one complete bidirectional CSV in {url}: {names}")
        log(f"Fetching publisher CTU-13 {number} ({family}) label file", flush=True)
        item = fetch(url + names[0], root / f"ctu13/{number}/{names[0]}", 600000000)
        entries.append({"dataset": "CTU-13", "capture": number, "family": family, "source": item, "readme": readme})
    for number, family in IOT.items():
        prefix = BASE + f"IoT-23-Dataset/IndividualScenarios/CTU-IoT-Malware-Capture-{number}/"
        readme = fetch(prefix + "README.md", root / f"iot23/{number}/README.md", 100000)
        log(f"Fetching publisher IoT-23 {number} ({family}) label file", flush=True)
        item = fetch(prefix + "bro/conn.log.labeled", root / f"iot23/{number}/conn.log.labeled", 30000000)
        entries.append({"dataset": "IoT-23", "capture": number, "family": family, "source": item, "readme": readme})
    for entry in entries:
        for role in ("source", "readme"):
            entry[role]["path"] = str(Path(entry[role]["path"]).relative_to(Path(collection).resolve()))
    write_json(root / "sources.json", {"publisher": "Stratosphere Laboratory", "entries": entries,
        "discovery": ["jevalin-collect/scripts/download_stratosphere.py", "https://www.stratosphereips.org/datasets-ctu13", "https://www.stratosphereips.org/datasets-iot23"],
        "scope": "Complete publisher annotation files, never malware binaries or PCAPs."})
    return entries


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--collection", default="../jevalin-collect")
    acquire(p.parse_args().collection)
