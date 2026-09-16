"""Small synthetic end-to-end demo, not an independently validated classifier."""
import json,time
from pathlib import Path
from system_one import DecisionModel
from system_one.qc import FIELDS
from system_one.quality_audit import quality

root=Path(__file__).resolve().parents[1]
train=[r for r in json.loads((root/'reports/reliability-dataset.json').read_text()) if r['split']=='dev']
evaluation=json.loads((root/'reports/checkpoint-fresh-evaluation.json').read_text())
assert not {r['id'] for r in train}&{r['id'] for r in evaluation}
model=DecisionModel.from_pretrained(revision='12a3808a956f869c767195e9266b59c4d21d92e2')
artifacts=root/'artifacts'/time.strftime('heads-%Y%m%d-%H%M%S')
fits={}
for name,field in FIELDS.items():
    start=time.perf_counter()
    head=model.compile(name,field,[{'id':r['id'],'context':r['context'],'value':r['expected'][name]} for r in train],components=8)
    fits[name]={'seconds':time.perf_counter()-start,'training_n':len(train)}
    head.save(artifacts/name)
    model.remove_head(name);model.load_head(artifacts/name)
    print('Trained/saved/reloaded',name,flush=True)
rows=[]
for case in evaluation:
    start=time.perf_counter();result=model.decide(case['context'],FIELDS)
    rows.append({'id':case['id'],'latency_ms':(time.perf_counter()-start)*1000,
                 'fields':{k:{'value':d.value,'label':FIELDS[k].values.index(case['expected'][k]),
                     'correct':d.value==case['expected'][k],'probabilities':list(d.probabilities),
                     'strategy':d.strategy} for k,d in result.items()}})
report={'model':model.model_id,'revision':model.revision,'layer':-1,'components':8,
    'fits':fits,'training_ids':[r['id'] for r in train], 'evaluation_ids':[r['id'] for r in evaluation],
    'accuracy':sum(v['correct'] for r in rows for v in r['fields'].values())/(len(rows)*len(FIELDS)),
    'fields':{k:quality([r['fields'][k]['probabilities'] for r in rows],[r['fields'][k]['label'] for r in rows]) for k in FIELDS},
    'notes':['Synthetic demo; fixed hyperparameters, no tuning or calibration.',
             'PCA/whitening and linear heads fitted on development training examples only.',
             'Evaluation set was used in earlier model comparisons; not a fresh confirmatory evaluation.',
             'Backbone remains frozen. Three same-layer heads share a context feature forward during decide.'],
    'rows':rows}
(root/'reports/compiled-demo.json').write_text(json.dumps(report,indent=2)+'\n')
print('Field accuracy',report['accuracy'],flush=True)
