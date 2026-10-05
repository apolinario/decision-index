"""
Formatting and serialization logic for Jev-class decision inputs.
Separates the State context, query instructions, and individual typed candidate options with criteria.
"""

import json
from typing import Dict, List, Any, Optional, Tuple, Union


class JevFormatter:
    """
    Formats JevBench records into structured sequences for D-JEPA.
    
    Structure:
    - Context State Prompt:
      <|state|>
      {state_text}
      <|instructions|>
      {instruction_text}
      
    - Option Criteria Prompts (one per candidate option k):
      <|option|> {label_k}: {criteria_k}
    """
    
    STATE_TAG = "<|state|>"
    INSTRUCTIONS_TAG = "<|instructions|>"
    OPTION_TAG = "<|option|>"

    @classmethod
    def format_state_prompt(cls, state: Any, instructions: Any) -> str:
        if isinstance(state, (dict, list)):
            state_clean = json.dumps(state, indent=2) if state else "none"
        else:
            state_clean = str(state).strip() if str(state).strip() else "none"

        if isinstance(instructions, (dict, list)):
            instructions_clean = json.dumps(instructions, indent=2)
        else:
            instructions_clean = str(instructions).strip()

        return f"{cls.STATE_TAG}\n{state_clean}\n{cls.INSTRUCTIONS_TAG}\n{instructions_clean}"

    @classmethod
    def format_option_prompt(cls, label: Any, criteria: Optional[str] = None) -> str:
        label_clean = str(label).strip()
        if criteria and str(criteria).strip():
            crit_clean = str(criteria).strip()
            if label_clean.startswith("k_") or label_clean.startswith("option_") or label_clean.isdigit():
                return f"{cls.OPTION_TAG} {crit_clean}"
            return f"{cls.OPTION_TAG} {label_clean}: {crit_clean}"
        return f"{cls.OPTION_TAG} {label_clean}"

    @classmethod
    def format_record(cls, record: Dict[str, Any]) -> Tuple[str, List[str], int, Optional[int]]:
        """
        Parses a single JevBench record.
        Returns:
            state_prompt: str
            option_prompts: List[str]
            target_idx: int (index of expected label in record['labels'])
            trap_idx: Optional[int] (index of surface_answer if present and != expected)
        """
        state = record.get("state", "")
        question = record.get("question", {})
        instructions = question.get("instructions", "")
        criteria_raw = question.get("criteria", {})
        labels = record.get("labels", [])
        expected = record.get("expected")

        state_prompt = cls.format_state_prompt(state, instructions)

        option_prompts = []
        for idx, label in enumerate(labels):
            crit_text = None
            if isinstance(criteria_raw, dict):
                crit_text = criteria_raw.get(label)
                if crit_text is None and isinstance(label, int):
                    crit_text = criteria_raw.get(str(label))
                elif crit_text is None and isinstance(label, str) and label.isdigit():
                    crit_text = criteria_raw.get(int(label))

                # Handle boolean / noul equivalences (e.g. labels ['no', 'yes'] with criteria {'false': ..., 'true': ...})
                if crit_text is None:
                    norm_label = str(label).lower()
                    if norm_label in ("no", "0"):
                        crit_text = criteria_raw.get("false") or criteria_raw.get("no") or criteria_raw.get("0") or criteria_raw.get(0)
                    elif norm_label in ("yes", "1"):
                        crit_text = criteria_raw.get("true") or criteria_raw.get("yes") or criteria_raw.get("1") or criteria_raw.get(1)
                    elif norm_label == "false":
                        crit_text = criteria_raw.get("no") or criteria_raw.get("0") or criteria_raw.get(0)
                    elif norm_label == "true":
                        crit_text = criteria_raw.get("yes") or criteria_raw.get("1") or criteria_raw.get(1)
            elif isinstance(criteria_raw, list):
                if idx < len(criteria_raw):
                    crit_text = criteria_raw[idx]

            option_prompts.append(cls.format_option_prompt(label, crit_text))

        # Target label index
        if expected in labels:
            target_idx = labels.index(expected)
        elif str(expected) in [str(l) for l in labels]:
            target_idx = [str(l) for l in labels].index(str(expected))
        else:
            target_idx = 0

        # Trap / Surface lure index
        trap_idx = None
        provenance = record.get("provenance", {}) or {}
        surface_answer = provenance.get("surface_answer")
        if surface_answer is not None and surface_answer != expected:
            if surface_answer in labels:
                trap_idx = labels.index(surface_answer)
            elif str(surface_answer) in [str(l) for l in labels]:
                trap_idx = [str(l) for l in labels].index(str(surface_answer))

        return state_prompt, option_prompts, target_idx, trap_idx
