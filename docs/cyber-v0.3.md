# esdb benchmark proposal

esdb is the enterprise security decision benchmark. this proposal adds four experimental assessments for a future decision index edition. the current local dataset is v0.3: 9,143 cases and 13,773 questions. the published suite files, benchmark weights and index formula are unchanged.

## assessments and scoring

| assessment | cases | questions | source and task | split unit |
|---|---:|---:|---|---|
| incident classification | 1,487 | 1,487 | complete selected guide incidents; predict the provider's three-class incident grade from anonymized alert metadata | organization; 271 selected organizations |
| endpoint investigation | 669 | 1,575 | native sysmon records; connect process events and supply supporting event references, or select insufficient evidence | connected capture group; 50 groups |
| network classification | 3,263 | 3,263 | publisher flow labels and a separate authored firewall-rule task | capture family or campaign; 12 groups |
| authorization policy | 3,724 | 7,448 | recovered programs and authored access or upload requests; apply the supplied policy to the evidence | source component; 133 components |

the endpoint assessment has 414 investigation cases and 255 field-extraction diagnostics. only the investigation cases belong in its primary results. related cases stay in the same partition.

policy pairs change one part of the data flow, approval, destination or policy. for program cases, credit requires both the approval decision and the data dependency. endpoint pairs have the same event count and field structure. they require host, process identifier and time links; a similar path or process number cannot replace a missing link. credit requires the supporting event references.

report each assessment separately. a case is correct only when all its required answers are correct. also report both-members-correct results for matched pairs, class results, coverage and results by source group. incident results include equal-organization accuracy. network results separate publisher datasets and firewall-rule decisions. background traffic in ctu is unjudged and excluded from malicious-versus-normal detection rates. failed, missing and invalid answers remain in their original denominators. question accuracy and probability metrics are additional diagnostics; probability metrics disclose valid-response coverage.

## current results

jev 1.13.0 and sev were evaluated on the same 6,044 development and calibration requests. jev returned 6,043 valid responses, with one retained incident-classification failure. sev retained eight context failures. evidence, question order and answer options were unchanged. these are calibration results, before independent answer review and final testing.

| assessment | calibration cases | jev case accuracy | sev case accuracy | development-fitted majority |
|---|---:|---:|---:|---:|
| incident classification | 495 | 21.82% | 21.41% | 55.15% |
| endpoint investigation | 148 | 62.16% | 30.41% | 45.95% |
| authorization policy | 1,232 | 84.42% | 47.97% | 28.57% |

random forest reaches 59.39% on incident classification. jev never selects false positive in calibration. its equal-organization accuracy is 17.74%, compared with 65.65% for random forest. guide remains a retrospective classification task, with anonymized detector information; it does not establish live incident-response performance.

jev gets both members correct in 73.21% of policy pairs and 28.38% of endpoint pairs. stronger average accuracy still leaves errors when the evidence or policy changes.

| network source | calibration cases | jev accuracy | sev accuracy | jev malicious flows missed |
|---|---:|---:|---:|---:|
| ctu-13 | 600 | 46.83% | 37.83% | 198 of 200 |
| iot-23 | 356 | 80.06% | 84.55% | 53 of 56 |
| splunk authored firewall rules | 120 | 95.83% | 82.50% | separate rule task |

ctu accuracy includes 200 background rows; detection rates exclude them. always predicting benign reaches 84.27% on iot-23. the independent cse-cic-ids2018 campaign is test-only and remains unscored. ctu and iot share a laboratory. the network assessment still needs more independent campaigns before a broad generalization claim.

[release-summary.json](esdb/release-summary.json) records the counts, source terms as recorded locally, model versions, aggregate calibration results and artifact hashes. it contains no request payloads, source programs, answer keys or individual predictions.

## local reproduction

this is a proposal with a local reproduction path. a clean public checkout cannot rebuild the full dataset yet: it also needs the collected sources and pinned prior-use and policy-analysis records. the source-program redistribution terms remain unresolved. portable source acquisition and publication permissions must be completed before a public dataset release.

install the construction and scoring dependencies and the pinned static parser:

```sh
python3 -m pip install -e '.[cyber,dev]'
npm ci --ignore-scripts --prefix tools/esdb
```

with the existing local source collection and analysis records, build the parent and then v0.3:

```sh
python3 -m decision_index.cyber build \
  --collection ../jevalin-collect \
  --sev-root ../Sev-security-20260925 \
  --typescript tools/esdb/node_modules/typescript/lib/typescript.js
python3 -m decision_index.cyber.v3.acquire --collection ../jevalin-collect
python3 -m decision_index.cyber.v3.cic --collection ../jevalin-collect
python3 -m decision_index.cyber.v3 build
python3 -m decision_index.cyber.v3 verify
python3 -m decision_index.cyber.v3 baselines --out runs/esdb-v03-baselines
python3 -m decision_index.cyber.v3 stage-devcal
```

generated data, review packets and model runs stay outside git. builders refuse to overwrite an existing edition. input files contain no answer keys; labels and source references are separate. the verifier checks generated files, source replay and split boundaries. known local sev training, development and calibration overlap is excluded. public-source pretraining overlap remains unknown.

the hosted jev runner checks the served model version and records request order, failures and run identity:

```sh
export TYPESAFE_API_KEY='<your key>'
python3 scripts/esdb_jev.py --help
python3 scripts/esdb_jev.py
python3 scripts/esdb_compare_v3.py \
  --sev-results runs/sev-v03-devcal-20261004/results.jsonl \
  --jev-results runs/jev-v03-devcal-20261004/results.jsonl \
  --out runs/esdb-v03-jev-comparison.json
```

native sev inference also requires the canonical sev runtime. the included modal scripts reproduce the author's setup; they are optional and require that sibling repository and configured provider credentials. the current settings-freeze and final-test orchestration are specific to sev. support for a model-independent final-test protocol remains integration work.

shared runner changes reject resumes after input, settings, model or runtime changes. shared scorers reject mixed results and invalid probability responses. the suite verifier also requires the frozen exclusions file. these checks can reject old run directories that lack a recorded identity; start a new output directory in that case.

## automated language review

a local editorial audit checked all 4,138 blinded policy and investigation cases, covering 8,276 questions. it used part-of-speech tagging, sentence parsing, a local grammar checker and phrase comparisons with 15 retrieved enterprise documentation pages. gartner returned an access error and provided no phrase support for this run. no grammar-checker errors or broken event references were detected.

the audit found repeated workplace messages, a long instruction sentence in the 414 investigation cases, and four cases with a placeholder process option. these are candidates for inspection; the audit does not certify the answer key. the frozen dataset is unchanged. see [the audit instructions](esdb-language-audit.md) and [aggregate audit results](esdb/language-audit-summary.json).

## review and release requirements

two security reviewers must independently answer every policy and investigation case using the blinded packet. no independent reviews have been completed. disagreements require adjudication; changed answers require a new immutable edition. then freeze model and evaluation settings and record one final test attempt. the public runner and scorer require the review and test protocol. editable local code is not an access-control boundary.

before inclusion in the index, complete independent review, obtain source-program permissions, make source acquisition portable, agree the scoring and reporting with maintainers, and complete the final test. keep authored workplace messages distinct from recovered communications: the upload and access requests were written for this assessment and are not private chat or email records.

the wording review uses [google's writing guidance](https://developers.google.com/style/tone), [anthropic's evaluation documentation](https://platform.claude.com/docs/en/test-and-evaluate/develop-tests), [gartner's cybersecurity assessment terminology](https://www.gartner.com/en/articles/cybersecurity-roadmap), [google's access-management documentation](https://docs.cloud.google.com/iam/docs/overview), and [google's event-search documentation](https://docs.cloud.google.com/chronicle/docs/investigation/udm-search). it checks ordinary technical terms and removes promotional or vague language. it does not establish authorship or reproduce a company's private workplace conversations.
