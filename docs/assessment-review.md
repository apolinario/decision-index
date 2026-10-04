# assessment design review

the review starts with this repository's benchmark list, source map in [suite.md](suite.md), adapters and scorers. it includes assessments outside cybersecurity. these design comparisons do not establish that esdb outperforms another benchmark.

| existing assessment | design used in esdb |
|---|---|
| contractnli | require supplied evidence and include an explicit insufficient-evidence answer |
| bpomp | compare controlled changes to one task requirement; keep related cases in one split |
| humicroedit | preserve ambiguity rather than invent a decisive answer |
| pop909 and cfcolor | report results by independent source group, not only by row |
| forecastbench | compare probability metrics with a declared baseline and keep settings selection outside test |
| habermas machine | identify whose judgment defines the label; provider grades and authored policies have different meanings |
| ragtruth | keep answer annotations tied to inspectable evidence; source replay does not replace independent review |
| hover | describe the limits of supplied evidence; interpretation does not establish evidence discovery |
| when2call | score recognition of missing information |
| new yorker | state the bounded choice task; the repository's caption-matching adaptation does not test newspaper writing |
| home appliances and bfcl | require all answers in a case to be correct and report failed cases |
| toolret and bright | distinguish unjudged evidence from verified negative examples |
| routerbench and sgd | inspect the actual model payload for answers, future information and source identifiers |

the original esdb policy panel had a 99.1% majority baseline. v0.3 replaces it with matched changes to evidence and policy, strengthens endpoint event reasoning and separates network source results. the [current proposal](cyber-v0.3.md) records the construction, baselines, remaining review and source-permission requirements.
