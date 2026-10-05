"""
Evaluation harness for Jevbench-Hard and typed decision benchmarks.
Measures Accuracy, ECE, Brier score, Trap Avoidance, and Per-Family performance.
"""

import json
import time
from pathlib import Path
from typing import Dict, Any, List, Optional, Union
from collections import defaultdict

import torch
from transformers import AutoTokenizer

from djepa.model.djepa import DJEPA


def evaluate_jevbench_hard(
    model: DJEPA,
    tokenizer,
    dataset_path: Union[str, Path] = "data/hard.jsonl",
    max_state_length: int = 4096,
    max_option_length: int = 256,
    device: Optional[torch.device] = None,
) -> Dict[str, Any]:
    """
    Evaluates D-JEPA on Jevbench-Hard records.
    """
    model.eval()
    if device is None:
        device = next(model.parameters()).device

    dataset_path = Path(dataset_path)
    records = []
    with open(dataset_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    total = len(records)
    correct = 0
    traps_total = 0
    traps_avoided = 0

    family_stats = defaultdict(lambda: {"total": 0, "correct": 0})
    type_stats = defaultdict(lambda: {"total": 0, "correct": 0})

    latencies = []
    confidences = []
    accuracies = []
    brier_scores = []

    results = []

    print(f"\n=======================================================")
    print(f"  Evaluating D-JEPA on Jevbench-Hard ({total} items)")
    print(f"=======================================================\n")

    for i, record in enumerate(records):
        start_t = time.perf_counter()
        dec = model.decide(
            record=record,
            tokenizer=tokenizer,
            max_state_length=max_state_length,
            max_option_length=max_option_length,
            device=device,
        )
        elapsed_ms = (time.perf_counter() - start_t) * 1000.0
        latencies.append(elapsed_ms)

        is_corr = dec["is_correct"]
        if is_corr:
            correct += 1

        fam = record.get("family", "unknown")
        qtype = record.get("question", {}).get("type", "unknown")
        family_stats[fam]["total"] += 1
        family_stats[fam]["correct"] += int(is_corr)
        type_stats[qtype]["total"] += 1
        type_stats[qtype]["correct"] += int(is_corr)

        conf = dec["confidence"]
        confidences.append(conf)
        accuracies.append(1.0 if is_corr else 0.0)

        # Brier score for this item
        labels = record.get("labels", [])
        expected = record.get("expected")
        probs_dict = dec["probabilities"]
        item_brier = sum(
            (probs_dict.get(lbl, 0.0) - (1.0 if lbl == expected else 0.0)) ** 2
            for lbl in labels
        )
        brier_scores.append(item_brier)

        # Trap checking
        provenance = record.get("provenance", {}) or {}
        surface_answer = provenance.get("surface_answer")
        if surface_answer and surface_answer != expected and surface_answer in labels:
            traps_total += 1
            if dec["prediction"] != surface_answer:
                traps_avoided += 1

        results.append({
            "id": record.get("id"),
            "family": fam,
            "prediction": dec["prediction"],
            "expected": expected,
            "is_correct": is_corr,
            "confidence": conf,
            "latency_ms": elapsed_ms,
        })

    overall_acc = (correct / total) * 100.0 if total > 0 else 0.0
    trap_avoid_rate = (traps_avoided / traps_total) * 100.0 if traps_total > 0 else 100.0
    mean_latency = sum(latencies) / len(latencies) if latencies else 0.0
    mean_brier = sum(brier_scores) / len(brier_scores) if brier_scores else 0.0

    # 10-bin ECE
    n_bins = 10
    bin_size = 1.0 / n_bins
    ece = 0.0
    for b in range(n_bins):
        b_low = b * bin_size
        b_high = (b + 1) * bin_size
        in_bin = [
            (conf, acc) for conf, acc in zip(confidences, accuracies)
            if b_low < conf <= b_high
        ]
        if in_bin:
            bin_conf = sum(x[0] for x in in_bin) / len(in_bin)
            bin_acc = sum(x[1] for x in in_bin) / len(in_bin)
            prop = len(in_bin) / total
            ece += abs(bin_acc - bin_conf) * prop

    # Print Summary Table
    print(f"{'Metric':<30} | {'Value':<15}")
    print(f"{'-'*30}-+-{'-'*15}")
    print(f"{'Overall Hard Accuracy':<30} | {overall_acc:.2f}% ({correct}/{total})")
    print(f"{'Target Threshold':<30} | 80.00%")
    print(f"{'Status':<30} | {'PASSED (>80)' if overall_acc >= 80.0 else 'IN PROGRESS'}")
    print(f"{'Expected Calibration Error':<30} | {ece:.4f}")
    print(f"{'Mean Brier Score':<30} | {mean_brier:.4f}")
    print(f"{'Trap Avoidance Rate':<30} | {trap_avoid_rate:.2f}% ({traps_avoided}/{traps_total})")
    print(f"{'Average Latency':<30} | {mean_latency:.2f} ms")
    print(f"\n--- Per-Family Breakdown ---")
    for fam, stats in sorted(family_stats.items()):
        fam_acc = (stats["correct"] / stats["total"]) * 100.0
        print(f"  {fam:<25}: {fam_acc:6.2f}% ({stats['correct']}/{stats['total']})")

    print(f"\n--- Per-Question-Type Breakdown ---")
    for qtype, stats in sorted(type_stats.items()):
        t_acc = (stats["correct"] / stats["total"]) * 100.0
        print(f"  {qtype:<25}: {t_acc:6.2f}% ({stats['correct']}/{stats['total']})")

    return {
        "overall_accuracy": overall_acc,
        "correct": correct,
        "total": total,
        "ece": ece,
        "brier_score": mean_brier,
        "trap_avoidance_rate": trap_avoid_rate,
        "traps_avoided": traps_avoided,
        "traps_total": traps_total,
        "mean_latency_ms": mean_latency,
        "family_stats": dict(family_stats),
        "type_stats": dict(type_stats),
        "results": results,
    }
