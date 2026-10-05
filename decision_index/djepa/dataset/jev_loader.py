"""
PyTorch Dataset and DataLoader utilities for Jev-class benchmarks.
"""

import json
from pathlib import Path
from typing import Dict, List, Any, Optional, Union
import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer

from djepa.dataset.formatter import JevFormatter


class JevDataset(Dataset):
    """
    PyTorch Dataset loading Jev-class decision benchmarks (such as Jevbench-Hard).
    """

    def __init__(
        self,
        file_path: Union[str, Path],
        tokenizer: PreTrainedTokenizer,
        max_state_length: int = 4096,
        max_option_length: int = 256,
    ):
        self.file_path = Path(file_path)
        self.tokenizer = tokenizer
        self.max_state_length = max_state_length
        self.max_option_length = max_option_length

        self.records: List[Dict[str, Any]] = []
        with open(self.file_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    self.records.append(json.loads(line))

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        record = self.records[idx]
        state_prompt, option_prompts, target_idx, trap_idx = JevFormatter.format_record(record)

        # Tokenize state
        state_enc = self.tokenizer(
            state_prompt,
            max_length=self.max_state_length,
            truncation=True,
            return_tensors="pt",
        )

        # Tokenize each option
        option_encs = [
            self.tokenizer(
                opt,
                max_length=self.max_option_length,
                truncation=True,
                return_tensors="pt",
            )
            for opt in option_prompts
        ]

        return {
            "id": record.get("id", str(idx)),
            "family": record.get("family", "unknown"),
            "question_type": record.get("question", {}).get("type", "choice"),
            "labels": record.get("labels", []),
            "target_idx": target_idx,
            "trap_idx": trap_idx,
            "state_input_ids": state_enc["input_ids"].squeeze(0),
            "state_attention_mask": state_enc["attention_mask"].squeeze(0),
            "option_input_ids": [o["input_ids"].squeeze(0) for o in option_encs],
            "option_attention_mask": [o["attention_mask"].squeeze(0) for o in option_encs],
        }


def collate_jev_batch(batch: List[Dict[str, Any]], pad_token_id: int = 0) -> Dict[str, Any]:
    """
    Collate variable-length state and option sequences into padded batch tensors.
    """
    batch_size = len(batch)

    # Pad state
    state_ids = [item["state_input_ids"] for item in batch]
    state_masks = [item["state_attention_mask"] for item in batch]
    max_state_len = max(s.size(0) for s in state_ids)

    padded_state_ids = torch.full((batch_size, max_state_len), pad_token_id, dtype=torch.long)
    padded_state_masks = torch.zeros((batch_size, max_state_len), dtype=torch.long)

    for i, (s_id, s_mask) in enumerate(zip(state_ids, state_masks)):
        padded_state_ids[i, : s_id.size(0)] = s_id
        padded_state_masks[i, : s_mask.size(0)] = s_mask

    # Handle options: max K across batch
    max_k = max(len(item["option_input_ids"]) for item in batch)

    # Find max option token length across all options in batch
    all_opt_lens = [
        opt.size(0) for item in batch for opt in item["option_input_ids"]
    ]
    max_opt_len = max(all_opt_lens) if all_opt_lens else 1

    padded_opt_ids = torch.full(
        (batch_size, max_k, max_opt_len), pad_token_id, dtype=torch.long
    )
    padded_opt_masks = torch.zeros(
        (batch_size, max_k, max_opt_len), dtype=torch.long
    )
    # Mask indicating valid options (since some records have fewer than max_k options)
    options_valid_mask = torch.zeros((batch_size, max_k), dtype=torch.bool)

    targets = []
    trap_indices = []

    for i, item in enumerate(batch):
        k_opts = len(item["option_input_ids"])
        options_valid_mask[i, :k_opts] = True
        for k in range(k_opts):
            opt_id = item["option_input_ids"][k]
            opt_mask = item["option_attention_mask"][k]
            padded_opt_ids[i, k, : opt_id.size(0)] = opt_id
            padded_opt_masks[i, k, : opt_mask.size(0)] = opt_mask
        targets.append(item["target_idx"])
        trap_indices.append(item["trap_idx"] if item["trap_idx"] is not None else -100)

    return {
        "ids": [item["id"] for item in batch],
        "families": [item["family"] for item in batch],
        "question_types": [item["question_type"] for item in batch],
        "labels": [item["labels"] for item in batch],
        "state_input_ids": padded_state_ids,
        "state_attention_mask": padded_state_masks,
        "option_input_ids": padded_opt_ids,
        "option_attention_mask": padded_opt_masks,
        "options_valid_mask": options_valid_mask,
        "targets": torch.tensor(targets, dtype=torch.long),
        "trap_indices": torch.tensor(trap_indices, dtype=torch.long),
    }
