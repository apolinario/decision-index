"""Pinned Jet v6.2 full BF16 checkpoint, using its published native readout."""
import hashlib
import importlib.util
import json
import math
import sys
from pathlib import Path

from decision_index.engines import Engine, Unsupported

MODEL = 'michaljach/jet'
REVISION = 'fbc3d2daa679e0d4bd9f99c9912b6496d5a41f0a'


def text(value):
    return value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)


def native_question(original):
    kind = original['type']
    instructions = text(original['instructions'])
    if kind == 'choice':
        criteria = original['criteria']
        if not 2 <= len(criteria) <= 255:
            raise Unsupported(f'Choice option count {len(criteria)} outside 2..255')
        return dict(type=kind, instructions=instructions,
                    criteria={k: text(v) for k, v in criteria.items()})
    if kind == 'noul':
        criteria = original.get('criteria')
        if criteria:
            if set(criteria) != {'true', 'false'}:
                raise Unsupported('Boolean criteria must define true and false')
            # Jet represents Boolean semantics in instructions, not a criteria field.
            instructions += '\nyes: ' + text(criteria['true']) + '\nno: ' + text(criteria['false'])
        return dict(type=kind, instructions=instructions)
    raise Unsupported(f'Question type {kind} is not part of this engine contract')


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


class JetEngine(Engine):
    name = 'jet'
    latency = 'Synchronized CUDA wall time including complete native prompts; one request at a time; no prefix cache; model loading excluded.'

    def __init__(self, model=MODEL, revision=REVISION, local_model=None, max_tokens=16384, **options):
        super().__init__(**options)
        import torch
        from huggingface_hub import hf_hub_download, snapshot_download
        if model != MODEL or revision != REVISION:
            raise ValueError('This submission pins the published Jet v6.2 checkpoint')
        self.max_tokens = int(max_tokens)
        if not 0 < self.max_tokens <= 16384:
            raise ValueError('Published complete-prompt limit is 16384 tokens')
        manifest_path = hf_hub_download(model, 'release-manifest.json', revision=revision)
        expected = json.loads(Path(manifest_path).read_text())['files']
        path = Path(local_model) if local_model else Path(snapshot_download(model, revision=revision))
        # A local cache is allowed only if every inference asset matches the Hub release.
        assets = ['config.json','tokenizer.json','tokenizer_config.json','chat_template.jinja',
                  'model.safetensors.index.json','format.py','runtime.py','calibration.json']
        assets += sorted(n for n in expected if n.endswith('.safetensors'))
        for name in assets:
            with (path/name).open('rb') as f:
                if hashlib.file_digest(f, 'sha256').hexdigest() != expected[name]:
                    raise ValueError(f'Local asset differs from pinned release: {name}')
        self.format = load_module('_decision_index_jet_format', path/'format.py')
        self.backend = load_module('_decision_index_jet_runtime', path/'runtime.py')
        torch.set_num_threads(4)
        torch.cuda.set_per_process_memory_fraction(.88)
        self.model, self.tokenizer = self.backend.load_model(str(path))
        self.temperatures = json.loads((path/'calibration.json').read_text())
        if any(not math.isfinite(v) or v<=0 for v in self.temperatures.values()):
            raise ValueError('Invalid published calibration')
        self.provenance = dict(model=model, revision=revision, adapter=None, dtype='bfloat16',
            max_tokens=self.max_tokens, max_choice_options=255, calibration=self.temperatures,
            assets_sha256={n:expected[n] for n in assets},
            engine_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            policy='Complete native Jet prompts; no truncation, option filtering, benchmark-specific prompting or access to gold labels. Boolean true/false descriptions rendered as yes/no instructions.')

    def __call__(self, state, questions):
        import torch
        prepared=[]
        for name, original in questions.items():
            q=self.format.Question.from_dict(native_question(original))
            prompt=self.format.build_prompt(self.tokenizer,self.format.render_state(state),q)
            ids=self.tokenizer.encode(prompt,add_special_tokens=False)
            if len(ids)>self.max_tokens:
                raise Unsupported(f'Complete prompt has {len(ids)} tokens; declared limit {self.max_tokens}')
            prepared.append((name,q,ids,self.format.label_token_ids(self.tokenizer,q)))
        answers={};raw={}
        with torch.no_grad():
            for name,q,ids,labels in prepared:
                z=self.backend.label_logits(self.model,{'ids':ids,'labels':labels})
                probs=(z.double()/self.temperatures[q.type]).softmax(-1).cpu().tolist()
                if q.type=='choice':
                    answers[name]={'type':'choice','choice':q.keys[max(range(len(probs)),key=probs.__getitem__)],
                                   'probabilities':dict(zip(q.keys,probs))}
                else:
                    answers[name]={'type':'noul','noul':probs[q.keys.index('yes')]}
                raw[name]={'prompt_tokens':len(ids),'label_logits':z.cpu().tolist()}
        return {'model':MODEL,'answers':answers,'usage':{'input_tokens':sum(len(x[2]) for x in prepared)}},raw

    def synchronize(self):
        import torch
        torch.cuda.synchronize()

    def runtime(self):
        import platform,torch,transformers
        from importlib.metadata import version
        return dict(platform=platform.platform(),torch=torch.__version__,transformers=transformers.__version__,
                    flash_linear_attention=version('flash-linear-attention'),gpu=torch.cuda.get_device_name(),
                    backend='Published Jet BF16 / SDPA / FLA with native PyTorch convolution')
