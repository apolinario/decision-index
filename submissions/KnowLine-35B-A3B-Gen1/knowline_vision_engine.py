"""Decision Index vision-board engine for KnowLine models (kit `decision-index vision run/pipeline --engine
knowline_vision_engine:KnowLineVision`).

KnowLine serves /v1/systemone (knowline_server.py in the model repo). It takes images as OpenAI-style content parts
inside a chat state, not as a top-level `images` list, so this engine reuses the kit's HttpDecide client (retries,
capacity markers, 1.6 MP cap, latency) and only changes the request body: the images go into one user message of the
state, the question and its lettered options stay a normal typed question, and the answer is read from
`answers[<key>].probabilities` as in the kit's `decisions` protocol. The model reads its usual `Answer:` position;
the `prompt.readout` hint ("The answer is (" for R-Bench-M and MMMU-Pro vision) is not used.

  decision-index vision pipeline --dir work/vision --engine knowline_vision_engine:KnowLineVision \
      --option base_url=http://127.0.0.1:8080 --option model=m --workers 16 --out runs/<name>-vision
"""
from decision_index.vision import images as img
from decision_index.vision.engines import HttpDecide, single_choice


class KnowLineVision(HttpDecide):
    name = "knowline-vision"

    def __init__(self, base_url=None, model="m", path="/v1/systemone", **options):
        super().__init__(base_url=base_url, path=path, protocol="decisions", model=model, **options)
        self.provenance.update({"kind": "knowline-systemone", "image_placement": "state: one user message of image parts"})

    def body(self, state, questions, images):
        key, q, keys = single_choice(questions)
        enc = [img.encode(p, self.image_encoding, self.image_root, self.max_pixels) for p in images or []]
        if enc:
            st = {"messages": [{"role": "user", "content": [{"type": "image", "image": u} for u in enc]}]}
        else:
            st = state or {}
        return {"model": self.model, "state": st, "questions": {key: q}, **self.extra}, key, keys
