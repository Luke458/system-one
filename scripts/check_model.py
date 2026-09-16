"""Full-checkpoint numerical/path checks. Run from the project root."""
import json
from pathlib import Path
import torch
from system_one import DecisionModel
from system_one.engine import candidate_ids
from system_one.qc import CASES, FIELDS

model = DecisionModel.from_pretrained(revision='12a3808a956f869c767195e9266b59c4d21d92e2')
context = CASES[0]['context']
batched = model.decide(context, FIELDS)
checks = {}
for name, field in FIELDS.items():
    single = model.decide(context, {name: field})[name]
    checks[name] = {'batch_value': batched[name].value, 'single_value': single.value,
                    'max_logit_delta': max(abs(a-b) for a,b in zip(batched[name].logits, single.logits))}
prompt, _ = model.field_prompt(context, FIELDS['cause'])
all_labels = candidate_ids(model.tokenizer, prompt, tuple(chr(65+i) for i in range(26)))
inputs = model.encode([prompt])
with torch.inference_mode():
    generated = model.model.generate(**inputs, do_sample=False, max_new_tokens=1, pad_token_id=model.tokenizer.pad_token_id)
first_token = model.tokenizer.decode(generated[0, -1:])
model.model.set_attn_implementation('eager')
eager = model.decide(context, FIELDS)
for name in FIELDS:
    checks[name]['eager_value'] = eager[name].value
    checks[name]['eager_max_logit_delta'] = max(abs(a-b) for a,b in zip(batched[name].logits,eager[name].logits))
report = {'revision': model.revision, 'checks': checks, '26_candidate_ids': all_labels,
          'greedy_first_token_for_cause': first_token,
          'candidate_argmax_label': chr(65 + batched['cause'].values.index(batched['cause'].value))}
Path('reports/model-checks.json').write_text(json.dumps(report, indent=2)+'\n')
print(json.dumps(report, indent=2))
assert first_token == report['candidate_argmax_label']
assert all(v['batch_value'] == v['single_value'] == v['eager_value'] for v in checks.values())
