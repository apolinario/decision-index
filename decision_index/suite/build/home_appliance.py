import copy
import hashlib
import json
import random

SEED = 2026091807
HOUSEHOLDS = ["alder", "birch", "cedar", "dogwood", "elm", "fir", "garden", "harbor", "indigo", "juniper", "kestrel", "linden", "maple", "northstar", "orchard", "prairie", "quartz", "redwood", "sequoia", "terrace", "upland", "violet", "willow", "zephyr"]
ROOM_SETS = [["kitchen", "den", "main bedroom", "garage", "patio"], ["galley", "lounge", "guest room", "study", "porch"], ["kitchen", "family room", "nursery", "workshop", "deck"], ["breakfast nook", "sitting room", "bedroom", "office", "mudroom"]]
SCENARIOS = ("exact_toggle", "room_except", "named_group", "duplicate_name", "already_satisfied", "range_reject", "atomic_block", "future_schedule", "conditional_rule", "power_filter", "context_pronoun", "missing_device", "relative_level", "room_alias_ambiguous", "last_wins", "partial_unavailable", "security_policy", "ordered_sequence", "group_exception", "status_query")


def canon(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def slug(text):
    return "_".join(text.lower().replace("'", "").split())


def make_home(index, rng):
    rooms = ROOM_SETS[index % len(ROOM_SETS)]
    light_names = ["Pendant", "Window Lamp", "Reading Lamp", "Ceiling Spots", "Corner Light"]
    devices = []
    for i, room in enumerate(rooms):
        devices.extend([
            {"id": f"{slug(room)}_light", "name": light_names[i], "room": room, "kind": "light", "available": True, "power": bool((index + i) % 2), "brightness": 20 + 15 * i, "watts": 8 + i * 3},
            {"id": f"{slug(room)}_plug", "name": ["Kettle", "Air Purifier", "Heated Throw", "Workbench Plug", "Patio Warmer"][i], "room": room, "kind": "appliance", "available": True, "power": bool((index + i + 1) % 2), "watts": [1200, 55, 110, 400, 1500][i]},
        ])
    devices[0]["name"] = "Reading Lamp"
    devices[4]["name"] = "Reading Lamp"
    devices.extend([
        {"id": f"{slug(rooms[0])}_island_light", "name": "Island Lights", "room": rooms[0], "kind": "light", "available": True, "power": bool(index % 2), "brightness": 65, "watts": 16},
        {"id": f"{slug(rooms[0])}_thermostat", "name": "Thermostat", "room": rooms[0], "kind": "thermostat", "available": True, "power": True, "temperature": 21, "min_temperature": 16, "max_temperature": 26, "watts": 3},
        {"id": f"{slug(rooms[1])}_speaker", "name": "Bookshelf Speaker", "room": rooms[1], "kind": "speaker", "available": True, "power": False, "volume": 35, "watts": 18},
        {"id": f"{slug(rooms[3])}_lock", "name": "Side Door", "room": rooms[3], "kind": "lock", "available": True, "locked": index % 2 == 0, "watts": 1},
    ])
    return {"home": HOUSEHOLDS[index], "now": "2026-09-18T20:00:00-07:00", "outside_temperature": 12 + index % 12, "rooms": rooms, "room_aliases": {"back room": [rooms[2], rooms[3]]}, "devices": devices, "groups": {"evening": [devices[1]["id"], devices[3]["id"], devices[-2]["id"]], "downstairs": [d["id"] for d in devices if d["room"] in rooms[:2]]}, "device_context": devices[3]["id"], "policy": {"multi_action": "atomic", "conflict": "last_wins", "allow_partial": False, "security_mode": "away" if index % 3 == 0 else "home", "away_unlock": "blocked", "unavailable_action": "blocked"}, "schedules": [], "rules": []}


def dev(home, did):
    return next(d for d in home["devices"] if d["id"] == did)


def scenario(home, name, variant):
    ds, rooms = home["devices"], home["rooms"]
    l0, l1, l2 = ds[0], ds[2], ds[4]
    a0, a1, a2 = ds[1], ds[3], ds[5]
    therm, speaker, lock = ds[-3], ds[-2], ds[-1]
    if name == "exact_toggle":
        value = not a0["power"]
        req = f"Could you switch the {a0['name'].lower()} {'on' if value else 'off'}?" if variant % 2 else f"I need the {a0['name'].lower()} {'running' if value else 'shut down'}. "
        return {"kind": "execute", "ops": [[a0["id"], "power", value]]}, req.strip()
    if name == "room_except":
        targets = [d["id"] for d in ds if d["room"] == rooms[0] and d["kind"] == "light" and d["id"] != l0["id"]]
        return {"kind": "execute", "ops": [[x, "power", False] for x in targets]}, f"Turn off every light in {rooms[0]} except the {l0['name']}."
    if name == "named_group":
        return {"kind": "execute", "ops": [[x, "power", True] for x in home["groups"]["evening"]]}, "Start the evening group, please."
    if name == "duplicate_name":
        return {"kind": "clarify", "candidates": [l0["id"], l2["id"]]}, "Turn on the Reading Lamp."
    if name == "already_satisfied":
        return {"kind": "no_change", "ops": [[speaker["id"], "power", speaker["power"]]]}, f"Leave the {speaker['name'].lower()} {'on' if speaker['power'] else 'off'}."
    if name == "range_reject":
        value = therm["max_temperature"] + 4
        return {"kind": "unsupported", "ops": [[therm["id"], "temperature", value]], "reason": "outside supported range"}, f"Set the thermostat to {value}°C."
    if name == "atomic_block":
        a1["available"] = False
        return {"kind": "blocked", "ops": [[a0["id"], "power", True], [a1["id"], "power", True]], "reason": "atomic request contains unavailable device"}, f"Turn on both the {a0['name'].lower()} and the {a1['name'].lower()}."
    if name == "future_schedule":
        return {"kind": "schedule", "at": "22:15", "ops": [[a2["id"], "power", False]]}, f"At 10:15 tonight, turn off the {a2['name'].lower()}."
    if name == "conditional_rule":
        return {"kind": "rule", "condition": ["outside_temperature", "below", 15], "ops": [[therm["id"], "temperature", 22]]}, "Whenever it gets below 15°C outside, set the thermostat to 22°C."
    if name == "power_filter":
        targets = [d["id"] for d in ds if d["room"] == rooms[0] and d["kind"] == "appliance" and d["watts"] > 100]
        return {"kind": "execute", "ops": [[x, "power", False] for x in targets]}, f"In {rooms[0]}, shut down appliances rated over 100 watts."
    if name == "context_pronoun":
        target = home["device_context"]
        return {"kind": "execute", "ops": [[target, "power", False]]}, "Turn it off too."
    if name == "missing_device":
        return {"kind": "clarify", "candidates": []}, f"Start the dehumidifier in {rooms[2]}."
    if name == "relative_level":
        value = min(100, l1["brightness"] + 20)
        return {"kind": "execute", "ops": [[l1["id"], "brightness", value]]}, f"Make the {l1['name'].lower()} 20 points brighter."
    if name == "room_alias_ambiguous":
        cands = [d["id"] for d in ds if d["room"] in home["room_aliases"]["back room"] and d["kind"] == "light"]
        return {"kind": "clarify", "candidates": cands}, "Switch off the light in the back room."
    if name == "last_wins":
        home["policy"]["multi_action"] = "ordered"
        return {"kind": "execute", "ops": [[a0["id"], "power", True], [a0["id"], "power", False]]}, f"Turn the {a0['name'].lower()} on, then turn it back off."
    if name == "partial_unavailable":
        a1["available"] = False
        home["policy"]["allow_partial"] = True
        home["policy"]["multi_action"] = "best_effort"
        return {"kind": "execute_partial", "ops": [[a0["id"], "power", True], [a1["id"], "power", True]]}, f"Switch on the {a0['name'].lower()} and the {a1['name'].lower()}. Do whatever is available."
    if name == "security_policy":
        home["policy"]["security_mode"] = "away"
        return {"kind": "blocked", "ops": [[lock["id"], "locked", False]], "reason": "away mode blocks unlock"}, f"Unlock the {lock['name'].lower()}."
    if name == "ordered_sequence":
        home["policy"]["multi_action"] = "ordered"
        return {"kind": "execute", "ops": [[l1["id"], "power", False], [l1["id"], "brightness", 40], [l1["id"], "power", True]]}, f"Switch off the {l1['name'].lower()}, set it to 40%, then switch it on."
    if name == "group_exception":
        targets = [x for x in home["groups"]["downstairs"] if x != a0["id"] and "power" in dev(home, x)]
        return {"kind": "execute", "ops": [[x, "power", False] for x in targets]}, f"Power down the downstairs group, but keep the {a0['name'].lower()} as it is."
    return {"kind": "query", "target": therm["id"], "property": "temperature"}, "What's the thermostat set to right now?"


def resolve(home, spec):
    final = copy.deepcopy(home)
    status = spec["kind"]
    changed = []
    if status in {"execute", "execute_partial"}:
        for did, prop, value in spec["ops"]:
            d = dev(final, did)
            if not d.get("available", True):
                if status == "execute_partial" and final["policy"]["allow_partial"]:
                    continue
                return {"status": "blocked", "targets": sorted({x[0] for x in spec["ops"]}), "changed": [], "final": home}
            old = d.get(prop)
            d[prop] = value
            if old != value:
                changed.append([did, prop, value])
        if not changed:
            status = "no_change"
    elif status == "schedule":
        final["schedules"].append({"at": spec["at"], "ops": spec["ops"]})
    elif status == "rule":
        final["rules"].append({"condition": spec["condition"], "ops": spec["ops"]})
    targets = sorted({x[0] for x in spec.get("ops", [])} | set(spec.get("candidates", [])) | ({spec["target"]} if "target" in spec else set()))
    return {"status": status, "targets": targets, "changed": changed, "final": final}


def summarize_updates(result):
    if result["status"] == "schedule":
        return "a future schedule is added; device properties do not change yet"
    if result["status"] == "rule":
        return "an automation rule is saved; device properties do not change immediately"
    if not result["changed"]:
        return "no device property changes"
    return "; ".join(f"{d}.{p} becomes {json.dumps(v)}" for d, p, v in result["changed"])


def choices(rng, values, answer, namespace):
    values = list(dict.fromkeys(values))
    assert answer in values and len(values) >= 2
    rng.shuffle(values)
    criteria, gold = {}, None
    for i, value in enumerate(values):
        key = "k_" + hashlib.sha256(f"{namespace}:{i}:{rng.random()}".encode()).hexdigest()[:8]
        criteria[key] = value
        if value == answer:
            gold = key
    return criteria, gold


def make_row(hidx, offset):
    split = "dev" if hidx < 8 else "test"
    rng = random.Random(f"home-benchmark:{SEED}:{hidx}:{offset}")
    home = make_home(hidx, rng)
    sidx = (hidx * 7 + offset) % len(SCENARIOS)
    scenario_name = SCENARIOS[sidx]
    spec, request = scenario(home, scenario_name, hidx + offset)
    result = resolve(home, spec)
    state = {k: v for k, v in home.items() if k != "home"}
    state["request"] = request
    qdefs = []
    statuses = ["execute now", "execute available parts now", "schedule for later", "save an automation rule", "ask a clarifying question", "make no change because already satisfied", "reject as unsupported", "block under household policy", "answer a state query"]
    status_desc = {"execute": "execute now", "execute_partial": "execute available parts now", "schedule": "schedule for later", "rule": "save an automation rule", "clarify": "ask a clarifying question", "no_change": "make no change because already satisfied", "unsupported": "reject as unsupported", "blocked": "block under household policy", "query": "answer a state query"}[result["status"]]
    qdefs.append(("resolution", "How should the assistant handle the request?", statuses, status_desc))
    executable_targets = set() if result["status"] == "clarify" else set(result["targets"])
    for d in home["devices"]:
        answer = "yes, this device is in the justified target set" if d["id"] in executable_targets else "no, this device is not in the justified target set"
        qdefs.append((f"target_member:{d['id']}", f"Is {d['id']} in the justified device target set for this request?", ["yes, this device is in the justified target set", "no, this device is not in the justified target set"], answer))
    outcome = summarize_updates(result)
    distract = ["no device property changes", "a future schedule is added; device properties do not change yet", "an automation rule is saved; device properties do not change immediately"]
    if result["targets"]:
        did = result["targets"][0]
        distract += [f"{did}.power becomes true", f"{did}.power becomes false"]
    qdefs.append(("outcome", "What is the concrete immediate outcome after applying the correct handling?", [outcome] + [x for x in distract if x != outcome], outcome))
    probe = home["devices"][(hidx + offset) % len(home["devices"])]
    prop = "power" if "power" in probe else "locked" if "locked" in probe else "temperature"
    pval = probe[prop]
    vals = [f"{probe['id']}.{prop} is {json.dumps(pval)}", f"{probe['id']}.{prop} is {json.dumps(not pval) if isinstance(pval, bool) else pval + 1}", "the property is not listed"]
    qdefs.append(("state_read", f"Before handling the request, what is the listed {prop} state of {probe['name']} in {probe['room']}?", vals, vals[0]))
    for n, (did, prop, value) in enumerate(spec.get("ops", [])[:2]):
        actual = dev(result["final"], did).get(prop)
        vals = [json.dumps(actual), json.dumps(value), json.dumps(dev(home, did).get(prop)), "not applicable or not listed"]
        qdefs.append((f"post_{n}", f"Immediately after correct handling, what is {prop} for {did}?", vals, json.dumps(actual)))
    questions, expected, roles = {}, {}, {}
    for pos, (role, instruction, vals, answer) in enumerate(qdefs):
        qid = "q_" + hashlib.sha256(f"qid:{SEED}:{hidx}:{offset}:{pos}".encode()).hexdigest()[:10]
        criteria, gold = choices(rng, vals, answer, qid)
        questions[qid] = {"type": "choice", "instructions": instruction, "criteria": criteria}
        expected[qid] = gold
        roles[qid] = role
    rid = f"home-appliance:{split}:{HOUSEHOLDS[hidx]}:{offset:02d}"
    return {"id": rid, "family": "Home-Appliance", "split": split, "state": state, "questions": questions, "expected": expected, "metadata": {"seed": SEED, "household_family": HOUSEHOLDS[hidx], "scenario_family": scenario_name, "template_family": f"{split}:{scenario_name}:surface-{(hidx + offset) % 2}", "intent_spec": spec, "question_roles": roles, "oracle": {"status": result["status"], "targets": result["targets"], "changed": result["changed"], "outcome": outcome}}}


def home_appliance(layout):
    rows = [make_row(h, i) for h in range(len(HOUSEHOLDS)) for i in range(10)]
    dev_rows = [r for r in rows if r["split"] == "dev"]
    test_rows = [r for r in rows if r["split"] == "test"]
    output = layout.normalized / "Home-Appliance.jsonl"
    output.write_text("".join(canon(r) + "\n" for r in test_rows))
    layout.home.mkdir(parents=True, exist_ok=True)
    (layout.home / "dev.jsonl").write_text("".join(canon(r) + "\n" for r in dev_rows))
    manifest = {"name": "Home-Appliance", "version": 1, "seed": SEED, "generated_rows": len(rows), "dev": len(dev_rows), "test": len(test_rows), "normalized_contains": "test only", "test_sha256": hashlib.sha256(output.read_bytes()).hexdigest(), "dev_sha256": hashlib.sha256((layout.home / "dev.jsonl").read_bytes()).hexdigest(), "generator": "decision_index/suite/build/home_appliance.py"}
    (layout.home / "generation-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


BUILDERS = {9: home_appliance}
