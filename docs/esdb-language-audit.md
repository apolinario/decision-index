# esdb language audit

the script checks the frozen v0.3 blinded review packet: 4,138 cases and 8,276 questions. it runs part-of-speech tagging, dependency parsing and grammar checking locally. it also checks event references, placeholder choices, repeated workplace messages and phrase support in enterprise documentation.

use a separate environment so editorial dependencies do not affect benchmark construction or inference:

```sh
uv venv --python 3.12 work/esdb-language-audit/.venv
uv pip install --python work/esdb-language-audit/.venv/bin/python -r tools/esdb/language-audit-requirements.txt
work/esdb-language-audit/.venv/bin/python scripts/esdb_language_audit.py
work/esdb-language-audit/.venv/bin/python -m pytest -q tests/test_esdb_language_audit.py
```

java 17 or newer is needed for the local grammar server. the first run downloads the english parsing model, language tool 6.6 and the public reference pages. subsequent runs can use verified cached pages:

```sh
work/esdb-language-audit/.venv/bin/python scripts/esdb_language_audit.py --offline
```

the default output is `runs/esdb-v03-language-audit/`:

- `report.md`: readable findings and counts.
- `summary.json`: scope, versions, file hashes, source-page receipts, thresholds and limitations.
- `findings.jsonl`: exact case ids and affected field paths for each finding.
- `annotations.jsonl`: sentence tokens, part-of-speech tags, dependencies and phrase-support urls.

the script checks all 23 frozen files before and after the audit. it rejects output or cache directories inside the frozen dataset. it reads no gold answers or model predictions, and performs no model inference. native programs, event fields, paths and reference-only answer choices are excluded from grammar checking. event references are checked separately. variable destination urls and event references are normalized for template grouping; annotations and grammar offsets refer to that normalized text.

commands can omit a subject. headings, answer labels and colon-labeled fields can be valid fragments. the small parser sometimes mistakes a leading command for a noun; annotations record a command fallback rather than claiming the parser found a verb. controls check both malformed examples and normal questions, commands and labels.

long sentences use a configurable 25-token threshold. that threshold is an editorial heuristic, not a rule imposed by an enterprise vendor. repeating benchmark instructions is expected; the repetition warning applies to workplace message bodies whose wording is presented as chat, tickets, email or handoffs.

phrase support comes from the pages already listed in the dataset's language report, plus the previously selected anthropic, gartner and koch pages. successful downloads are pinned by html and extracted-text hashes. failed downloads provide no phrase support. leading determiners are removed before exact noun-phrase lookup. absence from this limited corpus is informational; it does not establish bad english, ai authorship or lack of use at other companies.

grammar findings are candidates for inspection. sentence structure alone cannot establish that a security answer is correct or that a task reflects actual analyst work. any accepted wording or answer changes belong in a new dataset edition; this audit does not change v0.3.

the parsing components are described in the [spacy documentation](https://spacy.io/usage/linguistic-features). the grammar engine is [language tool](https://languagetool.org/dev). editorial guidance includes [google's tone guide](https://developers.google.com/style/tone) and [microsoft's concise wording guide](https://learn.microsoft.com/en-us/style-guide/word-choice/use-simple-words-concise-sentences).
