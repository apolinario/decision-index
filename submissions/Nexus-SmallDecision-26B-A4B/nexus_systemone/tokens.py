"""The reserved tokens the model adds to its base vocabulary."""

from __future__ import annotations

STATE_BEGIN = "<state_begin>"
STATE_END = "<state_end>"
Q_BEGIN = "<q_begin>"
Q_END = "<q_end>"
ANS = "<ans>"
PAUSE = "<pause>"
YES = "<yes>"
NO = "<no>"

MAX_OPTIONS = 255
MAX_LEVELS = 10

OPTION_TOKENS = tuple(f"<opt_{i}>" for i in range(MAX_OPTIONS))
LEVEL_TOKENS = tuple(f"<lvl_{i}>" for i in range(MAX_LEVELS))

STRUCTURAL_TOKENS = (STATE_BEGIN, STATE_END, Q_BEGIN, Q_END, ANS, PAUSE)
RESERVED_TOKENS = (*STRUCTURAL_TOKENS, YES, NO, *OPTION_TOKENS, *LEVEL_TOKENS)
