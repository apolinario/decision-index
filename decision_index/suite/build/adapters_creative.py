import collections
import hashlib
import json

from decision_index.suite.build.layout import REFERENCE_MODEL, dump_ascii, git_revision

ROOTS = "C C# D D# E F F# G G# A A# B".split()
TEMPLATES = {"M": (0, 4, 7), "m": (0, 3, 7), "dim": (0, 3, 6), "aug": (0, 4, 8), "sus2": (0, 2, 7), "sus4": (0, 5, 7), "7": (0, 4, 7, 10), "maj7": (0, 4, 7, 11), "m7": (0, 3, 7, 10), "half-dim7": (0, 3, 6, 10), "dim7": (0, 3, 6, 9), "mMaj7": (0, 3, 7, 11), "aug7": (0, 4, 8, 10)}
_aliases = {}
for _r in range(12):
    for _name, _ints in TEMPLATES.items():
        _aliases.setdefault(tuple(sorted((_r + x) % 12 for x in _ints)), []).append(f"{ROOTS[_r]}{_name}")
VOCAB = {pc: " / ".join(names) for pc, names in _aliases.items()}
VOCAB["NoChord"] = "NoChord"
VOCAB["Other"] = "Other"
EXCLUDED_SONGS = {"518", "620"}


def _track_events(mid, channel):
    out = []
    end = 0
    for tr in mid.tracks:
        tick = 0
        active = {}
        for msg in tr:
            tick += msg.time
            end = max(end, tick)
            if getattr(msg, "channel", None) != channel:
                continue
            if msg.type == "note_on" and msg.velocity > 0:
                active.setdefault(msg.note, []).append(tick)
            elif msg.type in ("note_off", "note_on"):
                starts = active.get(msg.note, [])
                if starts:
                    start = starts.pop(0)
                    out.append((start, tick, msg.note))
                    if not starts:
                        active.pop(msg.note, None)
    return out, end


def parse_piece(path):
    import mido

    mid = mido.MidiFile(path)
    music, end = _track_events(mid, 0)
    ann, _ = _track_events(mid, 1)
    end = max((b for _, b, _ in music), default=end)
    grid = list(range(2, int(end / mid.ticks_per_beat) + 1, 4))
    public = {"ticks_per_beat": mid.ticks_per_beat, "key_signatures": [], "time_signatures": []}
    for tr in mid.tracks:
        tick = 0
        for msg in tr:
            tick += msg.time
            if msg.type in ("key_signature", "time_signature"):
                public["key_signatures" if msg.type == "key_signature" else "time_signatures"].append({"beat": tick / mid.ticks_per_beat, **({"key": msg.key} if msg.type == "key_signature" else {k: getattr(msg, k) for k in ("numerator", "denominator")})})
    rows = []
    for beat in grid:
        t = beat * mid.ticks_per_beat
        notes = sorted({n % 12 for a, b, n in ann if a <= t < b})
        target = "NoChord" if not notes else VOCAB.get(tuple(notes), "Other")
        context = [{"pitch": n, "pitch_class": n % 12, "start_beat": a / mid.ticks_per_beat, "end_beat": b / mid.ticks_per_beat} for a, b, n in music if a < t + 4 * mid.ticks_per_beat and b > t - 4 * mid.ticks_per_beat]
        rows.append({"song_id": path.stem, "group_id": f"{path.stem}:beat-{beat}", "target_time_beats": beat, "context_notes": context, "annotation_pitch_classes": notes, "expected": target, "public_context": public})
    return rows


def _write(layout, name, rows):
    with (layout.normalized / (name + ".jsonl")).open("w") as f, (layout.requests / (name + ".jsonl")).open("w") as g:
        for r in rows:
            assert set(r["expected"]) == set(r["questions"])
            assert all(r["expected"][k] in q["criteria"] for k, q in r["questions"].items())
            f.write(dump_ascii(r) + "\n")
            g.write(dump_ascii({"model": REFERENCE_MODEL, "state": r["state"], "questions": r["questions"]}) + "\n")


def pop909(layout):
    raw = layout.repos / "pop909cl/POP909_processed"
    counts = collections.Counter()
    songs = set()
    revision = git_revision(raw.parent)
    hashes = {p.stem: hashlib.sha256(p.read_bytes()).hexdigest() for p in raw.glob("*.mid")}
    labels = {pc: "chord_" + str(i) for i, pc in enumerate(VOCAB)}
    criteria = {labels[pc]: ("No chord sounding at the target beat." if pc == "NoChord" else "A chord pitch-class set outside this vocabulary." if pc == "Other" else {"names": VOCAB[pc], "pitch_classes": list(pc)}) for pc in VOCAB}
    reverse = {name: pc for pc, name in VOCAB.items()}

    def rows():
        for path in sorted(raw.glob("*.mid")):
            if path.stem in EXCLUDED_SONGS:
                continue
            for r in parse_piece(path):
                songs.add(r["song_id"])
                counts["rows"] += 1
                if r["expected"] in ("Other", "NoChord"):
                    counts[r["expected"]] += 1
                assert all(n["start_beat"] < r["target_time_beats"] + 4 and n["end_beat"] > r["target_time_beats"] - 4 for n in r["context_notes"])
                if r["expected"] not in ("Other", "NoChord"):
                    assert tuple(r["annotation_pitch_classes"]) == reverse[r["expected"]]
                yield {"id": "POP909:" + r["group_id"], "family": "POP909-chord-pitch-class", "split": "evaluation-only", "state": {k: r[k] for k in ("target_time_beats", "context_notes", "public_context")}, "questions": {"chord": {"type": "choice", "instructions": "Infer the chord sounding at the target beat from the score notes and musical context. Choose its pitch-class set; MIDI pitch classes are C=0 through B=11. Equivalent chord names are grouped.", "criteria": criteria}}, "expected": {"chord": labels[reverse[r["expected"]]]}, "metadata": {"group_id": r["group_id"], "song_id": r["song_id"], "source_sha256": hashes[r["song_id"]], "source_revision": revision, "gold_annotation_pitch_classes": r["annotation_pitch_classes"], "gold_class": r["expected"], "uncertainty_cluster": "song_id", "protocol": "Score channel0 only; expert chord channel1 hidden. Fixed four-beat target grid, eight-beat context. Symbolic chord recognition, not arrangement or audio understanding."}}

    _write(layout, "POP909-chord-pitch-class", rows())
    report = {"eligible_songs": len(songs), **dict(counts), "vocabulary_count": len(criteria), "source_revision": revision, "exclusions": {"518": "algorithmic labels per source README", "620": "known score/annotation misalignment"}, "primary": "Exact pitch-class recognition; macro by song. Other and NoChord reported separately.", "source_sha256": hashes, "semantic_checks": "Gold classes checked against hidden annotation pitch classes; score event overlap checked; fixed gold-independent vocabulary."}
    (layout.suite / "pop909-chords-audit.json").write_text(json.dumps(report, indent=2) + "\n")
    return {k: v for k, v in report.items() if k != "source_sha256"}


def _hex(rgb):
    return ["#" + "".join(f"{int(round(float(x) * 255)):02x}" for x in rgb[i:i + 3]) for i in range(0, 15, 3)]


def cfcolor_rows(base):
    import numpy as np
    from scipy.io import loadmat

    d = loadmat(base / "allMTurkRatings.mat", squeeze_me=True)
    t = loadmat(base / "themeData.mat", squeeze_me=True, struct_as_record=False)["datapoints"]
    rgb = np.asarray(t.rgb)
    train = np.asarray(d["train_vec"])
    test = np.asarray(d["test_vec"])
    rows = []
    color_keys = [tuple(_hex(x)) for x in rgb]
    for user in sorted(set(test[:, 0].tolist())):
        hist = train[train[:, 0] == user]
        raw_hist = {}
        for u, p, r in hist:
            raw_hist.setdefault(int(p), set()).add(int(r))
        uniq = {p: next(iter(rs)) for p, rs in raw_hist.items() if len(rs) == 1}
        hist_order = sorted(uniq.items(), key=lambda z: hashlib.sha256(f"{user}:{z[0]}".encode()).hexdigest())
        raw_test = {}
        for u, p, r in test[test[:, 0] == user]:
            raw_test.setdefault(int(p), set()).add(int(r))
        candidates = [(p, next(iter(rs))) for p, rs in raw_test.items() if len(rs) == 1]
        seed = int(hashlib.sha256(f"user:{user}".encode()).hexdigest()[:16], 16)
        rng = np.random.default_rng(seed)
        rng.shuffle(candidates)
        for (pa, ra), (pb, rb) in zip(candidates[::2], candidates[1::2]):
            if abs(ra - rb) < 2 or color_keys[pa - 1] == color_keys[pb - 1]:
                continue
            h = []
            for hp, hr in hist_order:
                if hp not in (pa, pb) and color_keys[hp - 1] not in (color_keys[pa - 1], color_keys[pb - 1]):
                    h.append({"palette_id": hp, "rating": hr, "colors": _hex(rgb[hp - 1])})
                    if len(h) == 8:
                        break
            if len(h) < 4:
                continue
            h = h[:8]
            flip = int(hashlib.sha256(f"pair:{user}:{pa}:{pb}".encode()).hexdigest()[-1], 16) % 2
            a, b = ((pb, rb), (pa, ra)) if flip else ((pa, ra), (pb, rb))
            rows.append({"id": f"cfcolor:{user}:{pa}:{pb}", "family": "CFColor-preference", "split": "test", "group_id": f"user:{user}", "user_id": int(user), "choices": {"A": {"palette_id": a[0], "colors": _hex(rgb[a[0] - 1])}, "B": {"palette_id": b[0], "colors": _hex(rgb[b[0] - 1])}}, "history": h, "expected": "A" if a[1] > b[1] else "B", "rating_gap": abs(ra - rb)})
    return rows


def cfcolor(layout):
    base = layout.repos / "cfcolor/release"
    raw = cfcolor_rows(base)
    seen = collections.defaultdict(set)
    users = set()
    hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in [base / "allMTurkRatings.mat", base / "themeData.mat"]}

    def rows():
        for r in raw:
            assert 4 <= len(r["history"]) <= 8 and r["rating_gap"] >= 2
            users.add(r["user_id"])
            choices = r["choices"]
            assert choices["A"]["colors"] != choices["B"]["colors"]
            for c in choices.values():
                assert c["palette_id"] not in seen[r["user_id"]]
                seen[r["user_id"]].add(c["palette_id"])
                assert all(h["colors"] != c["colors"] for h in r["history"])
            yield {"id": r["id"], "family": "CFColor-preference", "split": "test", "state": {"history": [{"colors": h["colors"], "rating": h["rating"]} for h in r["history"]], "rating_scale": "1 = lowest preference; 5 = highest preference"}, "questions": {"preference": {"type": "choice", "instructions": "Which five-color palette would this user rate more highly, given their earlier ratings?", "criteria": {k: v["colors"] for k, v in choices.items()}}}, "expected": {"preference": r["expected"]}, "metadata": {"group_id": r["id"], "user_id": r["user_id"], "uncertainty_cluster": "user_id", "rating_gap": r["rating_gap"], "target_palette_ids": {k: v["palette_id"] for k, v in choices.items()}, "history_palette_ids": [h["palette_id"] for h in r["history"]], "source_sha256": hashes, "protocol": "Original test targets, original train history; nonoverlapping deterministic pairs, gap>=2. No reroll. Human preference prediction, not objective beauty."}}

    _write(layout, "CFColor-preference", rows())
    report = {"rows": len(raw), "users": len(users), "rating_gap_counts": dict(collections.Counter(r["rating_gap"] for r in raw)), "source_sha256": hashes, "primary": "Pair accuracy and user-macro accuracy; uncertainty clustered by user.", "semantic_checks": ["unique nonoverlapping target palettes per user", "4..8 history entries", "gap>=2", "no visible RGB duplicate targets or history leak"], "split": "Original train history only, original test targets only; probe unused."}
    (layout.suite / "cfcolor-preferences-audit.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


BUILDERS = {22: pop909, 23: cfcolor}
