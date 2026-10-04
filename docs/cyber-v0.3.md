# esdb benchmark proposal

esdb is the enterprise security decision benchmark. this proposal adds four experimental assessments for a future decision index edition. the current local dataset is v0.3: 9,143 cases and 13,773 questions. the published suite files, benchmark weights and index formula are unchanged.

## assessments and scoring

| assessment | cases | questions | source and task | split unit |
|---|---:|---:|---|---|
| incident classification | 1,487 | 1,487 | guide incident records; predict the provider's three-class incident grade from anonymized alert metadata | organization; 271 selected organizations |
| endpoint investigation | 669 | 1,575 | native sysmon records; connect process events and supply supporting event references, or select insufficient evidence | related captures; 50 groups |
| network classification | 3,263 | 3,263 | flow classifications from dataset publishers and a separate firewall-rule task written for this benchmark | capture family or campaign; 12 groups |
| authorization policy | 3,724 | 7,448 | source programs and access or upload requests written for this benchmark; apply the supplied policy to the evidence | source component; 133 components |

the endpoint assessment has 414 investigation cases and 255 field-extraction diagnostics. only the investigation cases belong in its primary results. related cases stay in the same partition.

the policy assessment has 3,192 source-program cases and 532 workplace requests: 266 access requests and 266 file-upload requests. case counts do not represent independent incidents. the workplace requests were written for this benchmark; they are not private chat or email records.

policy pairs change one part of the data flow, approval, destination or policy. for program cases, credit requires both the approval decision and the data dependency. endpoint pairs have the same event count and field structure. they require host, process identifier and time links; a similar path or process number cannot replace a missing link. credit requires the supporting event references.

report each assessment separately. a case is correct only when all its required answers are correct. for matched pairs, also report the percentage for which both cases are correct. include class results, answer coverage and results by source group. incident results include accuracy averaged across organizations. network results separate publisher datasets and firewall-rule decisions. ctu's background classification does not establish maliciousness, so these flows are excluded from malicious-versus-normal detection rates. failed, missing and invalid answers count as incorrect. question accuracy and probability metrics are additional diagnostics; probability metrics include the number of valid answers.

## current results

jev 1.13.0 and sev were evaluated on the same 6,044 development and calibration requests. jev returned 6,043 valid responses; one incident-classification request failed. eight sev requests failed because their full context did not fit. all failures count as incorrect. evidence, question order and answer options were unchanged. the following tables show calibration results, before independent answer review and final testing.

| assessment | calibration cases | jev case accuracy | sev case accuracy | majority baseline |
|---|---:|---:|---:|---:|
| incident classification | 495 | 21.82% | 21.41% | 55.15% |
| endpoint investigation | 148 | 62.16% | 30.41% | 45.95% |
| authorization policy | 1,232 | 84.42% | 47.97% | 28.57% |

the majority baselines use development data only. conventional classifiers are trained and selected using development data, with organization groups kept separate for guide. random forest reaches 59.39% on incident classification. jev never selects false positive in calibration. its accuracy averaged across organizations is 17.74%, compared with 65.65% for random forest. guide remains a retrospective classification task, with anonymized detector information; it does not establish live incident-response performance.

jev gets both cases correct in 73.21% of policy pairs and 28.38% of endpoint pairs. the individual-case scores therefore do not establish reliable answers across matched changes in policy or evidence.

| network source | calibration cases | jev accuracy | sev accuracy | jev malicious flows missed |
|---|---:|---:|---:|---:|
| ctu-13 | 600 | 46.83% | 37.83% | 198 of 200 |
| iot-23 | 356 | 80.06% | 84.55% | 53 of 56 |
| splunk evidence under written firewall rules | 120 | 95.83% | 82.50% | separate rule task |

ctu accuracy includes 200 background rows; detection rates exclude them. a classifier that always predicts benign reaches 84.27% on iot-23. the cse-cic-ids2018 campaign comes from another publisher and is reserved for test. it remains unscored and is excluded from model fitting and selection. its three capture days belong to one campaign. ctu and iot share a laboratory. the network assessment still needs more independent campaigns before a broad generalization claim.

[release-summary.json](esdb/release-summary.json) records the counts, source terms as recorded locally, model versions, aggregate calibration results and file hashes. it contains no request payloads, source programs, answer keys or individual predictions. this pull request includes code and aggregate reports; it does not publish the raw dataset.

## local reproduction

the current build instructions require local files. a clean public checkout cannot rebuild the full dataset yet. it also needs the collected sources and local records of prior training data and source-program analysis. the source-program redistribution terms remain unresolved. download and build instructions that work from a clean checkout, and publication permissions, must be completed before a public dataset release.

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

generated data, review packets and model runs stay outside git. builders refuse to overwrite an existing edition. input files contain no answer keys; labels and source references are separate. the builder compares selected records with their sources and records that check. the verifier checks file hashes, structure and split boundaries; it does not repeat the source comparison. known local sev training, development and calibration overlap is excluded. public-source pretraining overlap remains unknown.

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

native sev inference also requires the sev runtime in the sibling repository. the included modal scripts reproduce the author's setup; they are optional and require that repository and configured provider credentials. the scripts that record model settings and run the final test currently support sev. final-test support for other models remains integration work.

shared runner changes reject resumes after input, settings, model or runtime changes. shared scorers reject mixed results and invalid probability responses. the suite verifier also requires the frozen exclusions file. these checks can reject old run directories that lack a recorded identity; start a new output directory in that case.

## automated language review

a local editorial audit checked all 4,138 blinded policy and investigation cases, covering 8,276 questions. it used part-of-speech tagging, sentence parsing, a local grammar checker and phrase comparisons with 15 retrieved enterprise documentation pages. gartner returned an access error and provided no phrase support for this run. no grammar-checker errors or broken event references were detected.

the audit found repeated workplace messages, a long instruction sentence in the 414 investigation cases, and four cases with a placeholder process option. these are candidates for inspection; the audit does not certify the answer key. the frozen dataset is unchanged. see [the audit instructions](esdb-language-audit.md) and [aggregate audit results](esdb/language-audit-summary.json).

## review and release requirements

the current proposed esdb release gate requires two security reviewers to independently answer all 4,138 policy and investigation cases using the blinded packet. this is an esdb requirement in the proposed code, not a decision index submission rule. no independent reviews have been completed. disagreements require adjudication; changed answers require a new immutable edition. then freeze model and evaluation settings and record one final test attempt. the included runner and scorer require the review and test protocol. editable local code is not an access-control boundary.

before inclusion in the index, complete independent review and obtain source-program permissions. provide download and build instructions that work from a clean checkout. agree the scoring and reporting with maintainers, and complete the final test.

the wording review uses [google's writing guidance](https://developers.google.com/style/tone) and [anthropic's evaluation documentation](https://platform.claude.com/docs/en/test-and-evaluate/develop-tests). technical terms were checked against [gartner's cybersecurity assessment terminology](https://www.gartner.com/en/articles/cybersecurity-roadmap) and [google's access-management documentation](https://docs.cloud.google.com/iam/docs/overview). the review also uses [google's event-search documentation](https://docs.cloud.google.com/chronicle/docs/investigation/udm-search) and [koch's privacy policy](https://privacypolicy.kochinc.com/). gartner was readable in the browser for this separate wording review, although the automated audit could not retrieve it. the review checks ordinary technical terms and removes promotional or vague language. it does not establish authorship or reproduce a company's private workplace conversations.
