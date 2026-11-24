# Model submissions

   Model                LemonDecider 4B v3.2 (https://huggingface.co/inclusionAI/LemonDecider)
   Panel                Text
   Public index         64.25
   Model revision       4193a122 (https://huggingface.co/inclusionAI/LemonDecider/
                        tree/4193a122beb01e1ecaf796daf288d5f4b949bae9)
   Results              scores.json (https://huggingface.co/datasets/inclusionAI/lemondecider-decision-
                        index-results/blob/7f7f3b48194be51629a46c6b56940ba30b78d1e6/runs/lemondecider-4b-
                        v32-text/scores.json)
   Engine and hardware  BF16 Transformers, lemondecider-native-direct255, 16 independent PPU-ZW810E
                        workers
  ────────────────────────────────────────────────────────────────────────────────────────────────────────
   Model                LemonDecider 4B v3.2 (https://huggingface.co/inclusionAI/LemonDecider)
   Panel                Vision
   Public index         72.34906582840676
   Model revision       4193a122 (https://huggingface.co/inclusionAI/LemonDecider/
                        tree/4193a122beb01e1ecaf796daf288d5f4b949bae9)
   Results              vision-scores.json (https://huggingface.co/datasets/inclusionAI/lemondecider-
                        decision-index-results/blob/7f7f3b48194be51629a46c6b56940ba30b78d1e6/runs/
                        lemondecider-4b-v32-vision/vision-scores.json)
   Engine and hardware  BF16 Transformers, lemondecider-native-direct255, 16 independent PPU-ZW810E
                        workers

  Both full public runs use the Decision Index 0.3 suite (edition: 0.3; the text suite rows sha256 is
  b2b56d6fb636837ca469e689087bdbf373dda8de7638aa2da6793e6eda0792d5); the kit commit was not recorded in
  the run artifacts. The evaluated engine packs a request's questions into one joint sequence (joint-
  question-inline-readout-v1), reads each answer at that question's <|fim_suffix|> marker, never
  truncates, and recursively bisects an over-limit question list with recursive-question-bisection-v1; the
  checkpoint context is 262,144 tokens and the evaluated serving limit is 8,192 tokens per joint sequence.
  Vision uses between 65,536 and 1,638,400 pixels per image and at most 3,200 image tokens. Temperature is
  0.835655 for choice, noul and score, up to 255 candidates are supported, and rereading is disabled. The
  released joint/no-reread engine, including image tensors, is bundled in the model repository and serves
  both POST /v1/systemone and POST /v1/decisions; the latter is the official Vision decide --option
  protocol=decisions path. Parity checks on a Text suite row and a two-image Vision request reproduced the
  archived evaluator's probabilities exactly. In the published JSONL only engine and response.model were
  normalized; predictions, probabilities, timings and statuses are unchanged. The checkpoint config also
  sets the legacy inference_protocol.read_twice flag to false.
