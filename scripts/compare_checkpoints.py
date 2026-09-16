"""Development-only configuration selection, then fixed fresh synthetic evaluation."""
import gc
import hashlib
import json
from pathlib import Path
import statistics
import time
import torch
import transformers
from system_one import DecisionModel
from system_one.qc import FIELDS
from system_one.quality_audit import quality

ROOT=Path(__file__).resolve().parents[1]
FINAL=('openbmb/MiniCPM5-2B','12a3808a956f869c767195e9266b59c4d21d92e2')
SFT=('openbmb/MiniCPM5-2B-SFT','3e0690c57f6ac772180498467c82e56170cc00ac')
# Fix these texts and labels before loading/evaluating the comparison checkpoints.
fresh={
'missing_stores':[
'The revenue chart fell 8%. Item-level checks pass for every received file. However, an export task failed for 9 active shops, whose complete local receipts never reached reporting. No shops closed.',
'An analyst blames declining demand. The logs instead show that the eastern district store connector timed out: those open shops are absent from reporting. Reporting contains all items from the other shops.',
'Sales remained close to forecast because trading elsewhere was strong, but 6 expected open outlets are missing from the loaded table after their deliveries failed. All items are intact in deliveries that arrived.',
'The branch master and trading calendar confirm 330 locations should report. Only 315 appear. Reconciliation finds the other 15 feeds stuck in a failed queue, not intentionally excluded.'],
'missing_products':[
'All locations appear and their delivery counts match. Source receipts include medicine items, but a warehouse join rejects every medicine SKU after an identifier change. No real assortment removal occurred.',
'Total sales increased despite the fault. Every branch uploaded, but a filter accidentally discarded the same two product groups from all uploads. The groups still sold and are present in source receipts.',
'A dashboard claims store coverage problems, but every store is verified present. The actual discrepancy is that unmapped apparel items were dropped by an ETL join; raw deliveries contain them.',
'Every location file passed transfer checks. A deployment changed an item-code conversion and eliminated toy transactions from the reporting table although those items continue to sell.'],
'real_market_movement':[
'Twenty stores permanently closed as planned. All remaining active stores and expected products are represented. Receipts reconcile exactly with reporting; the sales decrease is explained by the real closures.',
'A product line was deliberately discontinued before the period. All products actually sold and all active shops are fully reported. Warehouse totals match receipts, and lower sales reflect the genuine assortment change.',
'A transfer alarm occurred yesterday but was repaired and all files were backfilled before this report. Complete store and item reconciliation passes. Receipts themselves show demand fell after a price increase.',
'Observed sales increased by 40% during a festival. Independent receipts confirm that increase exactly, all expected locations and items are present, and no data discrepancy remains.'],
'insufficient_evidence':[
'The chart is unusual and a manager is certain the feed is broken. No ingestion logs, source receipts, or coverage checks have been provided to support that opinion.',
'All stores are listed, but nobody checked whether their item-level records or totals are complete. Revenue is lower, and no reconciliation or demand evidence is available.',
'Product counts appear normal, but store coverage, receipts and processing logs are unavailable. Sales changed and the reason cannot yet be established.',
'An old incident report mentions a connector failure last month. For this week there is only a changed sales total, with no current logs or completeness checks.']}
new=[]
for cause,texts in fresh.items():
    for i,text in enumerate(texts):
        issue=cause in ('missing_stores','missing_products')
        new.append({'id':f'fresh-{cause}-{i}','context':text,
                    'expected':{'data_issue':issue,'cause':cause,'escalate':issue}})
(ROOT/'reports/checkpoint-fresh-evaluation.json').write_text(json.dumps(new,indent=2)+'\n')
original=json.loads((ROOT/'reports/reliability-dataset.json').read_text())
dev=[r for r in original if r['split']=='dev']
prior=json.loads((ROOT/'reports/reliability-experiment.json').read_text())
assert prior['revision']==FINAL[1]


def evaluate(model,cases,mode):
    rows=[]
    for case in cases:
        start=time.perf_counter()
        outputs=model.decide(case['context'],FIELDS) if mode=='single' else model.decide_balanced(case['context'],FIELDS)
        elapsed=(time.perf_counter()-start)*1000
        fields={}
        for k,d in outputs.items():
            fields[k]={'value':d.value,'probabilities':list(d.probabilities),
                       'label':FIELDS[k].values.index(case['expected'][k]),
                       'correct':d.value==case['expected'][k],
                       'order_stable':getattr(d,'order_stable',None)}
        rows.append({'id':case['id'],'latency_ms':elapsed,'fields':fields})
    return rows


def summarize(rows):
    predictions=[v for r in rows for v in r['fields'].values()]
    return {'accuracy':sum(v['correct'] for v in predictions)/len(predictions),
            'exact_match':sum(all(v['correct'] for v in r['fields'].values()) for r in rows)/len(rows),
            'median_ms':statistics.median(r['latency_ms'] for r in rows),
            'fields':{k:quality([r['fields'][k]['probabilities'] for r in rows],[r['fields'][k]['label'] for r in rows]) for k in FIELDS}}


def load(which):
    repo,revision=FINAL if which=='final' else SFT
    return DecisionModel.from_pretrained(repo,revision=revision)

report={'checkpoints':{'final':FINAL,'sft':SFT},'development':{},'evaluation':{},'rows':{},
        'fresh_evaluation_sha256':hashlib.sha256(json.dumps(new,sort_keys=True).encode()).hexdigest(),
        'environment':{'torch':torch.__version__,'transformers':transformers.__version__,'dtype':'bfloat16'},
        'notes':['Final-checkpoint development metrics reused from the pinned prior experiment.',
                 'Selection: development field accuracy, then exact match; insertion order breaks exact ties.',
                 'Fresh evaluation fixed before comparison inference. Synthetic, same author/task; not real-world validation.',
                 'No calibration fit; no deployment or confidence threshold selected.']}
for mode,old in [('single','baseline'),('balanced','balanced')]:
    report['development'][f'final/{mode}']=prior['development'][old]
model=load('sft')
for mode in ('single','balanced'):
    rows=evaluate(model,dev,mode);key=f'sft/{mode}'
    report['development'][key]=summarize(rows);report['rows']['dev/'+key]=rows
    print('DEV',key,report['development'][key]['accuracy'],flush=True)
selected=max(report['development'],key=lambda k:(report['development'][k]['accuracy'],report['development'][k]['exact_match']))
report['selected']=selected
print('SELECTED',selected,flush=True)
del model;gc.collect();torch.cuda.empty_cache()
for key in dict.fromkeys(('final/balanced',selected)):
    which,mode=key.split('/');model=load(which)
    rows=evaluate(model,new,mode)
    report['evaluation'][key]=summarize(rows);report['rows']['eval/'+key]=rows
    print('EVAL',key,report['evaluation'][key]['accuracy'],flush=True)
    del model;gc.collect();torch.cuda.empty_cache()
(ROOT/'reports/checkpoint-comparison.json').write_text(json.dumps(report,indent=2)+'\n')
