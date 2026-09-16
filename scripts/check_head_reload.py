"""Load previously fitted local artifacts into a new model instance."""
import json
from pathlib import Path
from system_one import DecisionModel
from system_one.qc import FIELDS
root=Path(__file__).resolve().parents[1]
artifacts=sorted((root/'artifacts').glob('heads-*'))[-1]
model=DecisionModel.from_pretrained(revision='12a3808a956f869c767195e9266b59c4d21d92e2')
for name in FIELDS:model.load_head(artifacts/name)
prior=json.loads((root/'reports/compiled-demo.json').read_text())
cases=json.loads((root/'reports/checkpoint-fresh-evaluation.json').read_text())
checks=[]
for case in cases:
    actual=model.decide(case['context'],FIELDS)
    previous=next(r for r in prior['rows'] if r['id']==case['id'])
    for k,d in actual.items():
        delta=max(abs(a-b) for a,b in zip(d.probabilities,previous['fields'][k]['probabilities']))
        assert d.value==previous['fields'][k]['value'] and delta<1e-6
        checks.append({'id':case['id'],'field':k,'max_probability_delta':delta})
(root/'reports/head-reload-check.json').write_text(json.dumps(checks,indent=2)+'\n')
print(len(checks),'outputs match after loading into a new model instance')
