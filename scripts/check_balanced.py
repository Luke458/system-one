"""Verify packaged balanced API against the frozen evaluation measurements."""
import json
from pathlib import Path
from system_one import DecisionModel
from system_one.qc import FIELDS

root=Path(__file__).resolve().parents[1]
cases=json.loads((root/'reports/reliability-dataset.json').read_text())
expected=json.loads((root/'reports/reliability-experiment.json').read_text())
model=DecisionModel.from_pretrained(revision='12a3808a956f869c767195e9266b59c4d21d92e2')
checks=[]
for case in cases:
    if case['split']!='eval':continue
    result=model.decide_balanced(case['context'],FIELDS)
    previous=next(r for r in expected['rows'] if r['id']==case['id'] and r['method']=='balanced')
    for k,v in result.items():
        old=previous['fields'][k]
        delta=max(abs(a-b) for a,b in zip(v.probabilities,old['probabilities']))
        assert v.value==old['value'] and delta<1e-7
        checks.append({'id':case['id'],'field':k,'probability_delta':delta,'order_stable':v.order_stable})
(root/'reports/balanced-api-check.json').write_text(json.dumps(checks,indent=2)+'\n')
print(f'{len(checks)} field outputs match the frozen experiment.')
