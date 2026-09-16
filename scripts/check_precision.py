"""Check whether the reliability failure is mainly reduced-precision arithmetic."""
import json
from pathlib import Path
import torch
from system_one import DecisionModel
from system_one.qc import FIELDS
root=Path(__file__).resolve().parents[1]
cases=[r for r in json.loads((root/'reports/reliability-dataset.json').read_text()) if r['split']=='dev']
prior=json.loads((root/'reports/reliability-experiment.json').read_text())
model=DecisionModel.from_pretrained(revision='12a3808a956f869c767195e9266b59c4d21d92e2',dtype=torch.float32)
model.model.set_attn_implementation('eager')
rows=[]
for case in cases:
    result=model.decide(case['context'],FIELDS)
    old=next(r for r in prior['rows'] if r['id']==case['id'] and r['method']=='baseline')
    for k,d in result.items():
        rows.append({'id':case['id'],'field':k,'value':d.value,'correct':d.value==case['expected'][k],
                     'bf16_value':old['fields'][k]['value'],
                     'changed':d.value!=old['fields'][k]['value'],
                     'probability_delta':max(abs(a-b) for a,b in zip(d.probabilities,old['fields'][k]['probabilities']))})
report={'dtype':'float32','attention':'eager','accuracy':sum(r['correct'] for r in rows)/len(rows),
        'changed_fields':sum(r['changed'] for r in rows),'rows':rows}
(root/'reports/precision-check.json').write_text(json.dumps(report,indent=2)+'\n')
print({k:v for k,v in report.items() if k!='rows'})
