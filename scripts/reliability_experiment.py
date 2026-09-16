import json,time,itertools
from pathlib import Path
import torch
from system_one import DecisionModel,Boolean
from system_one.qc import FIELDS,CASES
from system_one.quality_audit import quality

ROOT=Path(__file__).resolve().parents[1]
# Written before inference; synthetic examples, not real-world held-out validation.
records=[]
contexts={
'dev':{
'missing_stores':[
'Only 75 of 100 scheduled shops supplied data. Transfer logs confirm the other 25 uploads failed. All product categories are intact for reporting shops.',
'Revenue totals are low because 12 branch files were rejected by the loader. The branches were open and their source receipts exist. Product mappings are complete.',
'The expected 60 outlets all traded today. Warehouse data includes only 40 outlets after a network outage blocked 20 feeds.'],
'missing_products':[
'Every store file arrived, but a broken category filter discarded every beverage item. Source files contain the beverages and all other categories.',
'All outlet IDs reconcile. The new SKU join dropped ten valid product codes from every store; their source transactions are present.',
'Shop coverage is complete. An ingestion bug removed the bakery category while retaining all other product categories.'],
'real_market_movement':[
'All stores and SKUs are present, and warehouse totals match source receipts. Foot traffic and purchases declined during a storm; independent counters agree.',
'Every store and product reconciles with source records. Prices rose and customers bought fewer units. Auditors found no missing data or processing error.',
'The promotion finished. Complete source and warehouse totals both show lower purchases; every outlet and product is present.'],
'insufficient_evidence':[
'Only the sales total is available. It is lower today; coverage, receipt data, and ingestion logs have not been checked.',
'A manager reports an unexpected revenue change. There are no figures for stores or products and no evidence identifying the reason.',
'Sales appear low, but no source reconciliation or pipeline diagnostic has been performed. The cause is unknown.']},
'eval':{
'missing_stores':[
'A scheduler skipped 18 of the 90 active locations. Their receipts never entered the reporting table. The remaining locations have their full item ranges.',
'The retail warehouse excludes the northern region after its connector failed. Those outlets were trading normally and retained complete receipts locally.',
'A delivery manifest lists 210 shops; the warehouse contains 203. Logs identify seven store uploads rejected due to a header error. Items from the other 203 are complete.'],
'missing_products':[
'Location counts match the manifest exactly. A faulty item whitelist excluded all dairy transactions, although those records exist in the source deliveries.',
'All branches reported, but source-to-target reconciliation finds a set of household SKUs absent after a product-code translation failure.',
'Each store feed was accepted. A pipeline rule mistakenly deleted the frozen-food lines, making totals incomplete across all locations.'],
'real_market_movement':[
'An independent receipt audit matches the reporting system exactly, with complete outlet and item coverage. Fewer customers shopped during roadworks.',
'All location and item checks pass and every receipt reconciles. A competitor opened nearby; both source receipts and warehouse totals show fewer purchases.',
'Reported volumes increased after a successful campaign. Full store and product coverage is verified and totals match the original transactions exactly.'],
'insufficient_evidence':[
'A chart shows a sharp change in weekly revenue. No completeness checks, source totals, or system logs accompany the chart.',
'The only observation is that revenue differs from forecast. There is no evidence about missing locations, absent items, or genuine changes in demand.',
'An analyst suspects a reporting problem but has supplied no records, coverage counts, or reconciliation results. There is not enough evidence to determine a cause.']}}
for split,groups in contexts.items():
 for cause,texts in groups.items():
  for i,text in enumerate(texts):
   issue=cause in ('missing_stores','missing_products')
   records.append({'id':f'{split}-{cause}-{i}','split':split,'context':text,
                   'expected':{'data_issue':issue,'cause':cause,'escalate':issue}})
(ROOT/'reports/reliability-dataset.json').write_text(json.dumps(records,indent=2)+'\n')
model=DecisionModel.from_pretrained(revision='12a3808a956f869c767195e9266b59c4d21d92e2')
original=model.field_prompt
style='original'
def prompt(context,field,option_order=None):
 if style=='original':return original(context,field,option_order)
 n=len(field.values);order=tuple(range(n)) if option_order is None else tuple(option_order)
 # Explicit definitions improve interpretation without examples or dataset-specific answers.
 meanings={False:'No. The evidence does not establish that the answer is yes.',
           True:'Yes. The evidence establishes that the answer is yes.',
           'missing_stores':'Records from expected stores are absent because of a data transfer or processing failure.',
           'missing_products':'Records for expected products or categories are absent because of a data processing failure.',
           'real_market_movement':'Complete reconciled records show a genuine change in transactions or customer demand.',
           'insufficient_evidence':'The available evidence does not establish a specific cause.'}
 labels=tuple(chr(65+i) for i in range(n))
 options='\n'.join(f'{labels[pos]}: {meanings.get(field.values[i],str(field.values[i]))}' for pos,i in enumerate(order))
 text=('Assess the evidence and choose the best answer to the question. Do not assume missing facts. '
       'Return only the option letter.\nEvidence: '+context+'\nQuestion: '+field.question+'\n'+options)
 return model.chat(text),tuple(labels[order.index(i)] for i in range(n))
model.field_prompt=prompt

def evaluate(cases,method):
 global style
 style='original' if method in ('baseline','balanced') else 'defined'
 rows=[]
 for case in cases:
  expanded={};orders={};names={}
  for key,f in FIELDS.items():
   variants=[tuple(range(len(f.values)))] if method=='baseline' else list(itertools.permutations(range(len(f.values))))
   names[key]=[]
   for i,order in enumerate(variants):
    alias=f'{key}_{i}';expanded[alias]=f;orders[alias]=order;names[key].append(alias)
  start=time.perf_counter();result=model.decide(case['context'],expanded,option_orders=orders);elapsed=(time.perf_counter()-start)*1000
  values={}
  for key,aliases in names.items():
   probs=torch.tensor([result[a].probabilities for a in aliases],dtype=torch.float64).mean(0)
   selected=int(probs.argmax());vals=FIELDS[key].values
   unanimous=all(result[a].value==vals[selected] for a in aliases)
   values[key]={'probabilities':probs.tolist(),'value':vals[selected], 'unanimous':unanimous,
                'label':vals.index(case['expected'][key]),'correct':vals[selected]==case['expected'][key]}
  rows.append({'id':case['id'],'method':method,'latency_ms':elapsed,'fields':values})
 return rows

def summarize(rows):
 allfields=[v for r in rows for v in r['fields'].values()]
 accepted=[v for v in allfields if v['unanimous']]
 return {'accuracy':sum(v['correct'] for v in allfields)/len(allfields),
         'exact_match':sum(all(v['correct'] for v in r['fields'].values()) for r in rows)/len(rows),
         'agreement_coverage':len(accepted)/len(allfields),
         'agreement_accuracy':sum(v['correct'] for v in accepted)/len(accepted) if accepted else None,
         'fields':{k:quality([r['fields'][k]['probabilities'] for r in rows],[r['fields'][k]['label'] for r in rows]) for k in FIELDS}}
report={'model':model.model_id,'revision':model.revision,'development':{},'evaluation':{},'rows':[],
        'notes':['Synthetic development/evaluation experiment; not external real-world validation.']}
for method in ('baseline','balanced','defined_balanced'):
 rows=evaluate([r for r in records if r['split']=='dev'],method)
 report['development'][method]=summarize(rows);report['rows']+=rows
 print('DEV',method,report['development'][method]['accuracy'],flush=True)
# Predeclared selection rule: development accuracy, then exact match; baseline wins exact ties.
selected=max(report['development'],key=lambda m:(report['development'][m]['accuracy'],report['development'][m]['exact_match']))
report['selected']=selected
for method in dict.fromkeys(('baseline',selected)):
 rows=evaluate([r for r in records if r['split']=='eval'],method)
 report['evaluation'][method]=summarize(rows);report['rows']+=rows
 print('EVAL',method,report['evaluation'][method]['accuracy'],flush=True)
(ROOT/'reports/reliability-experiment.json').write_text(json.dumps(report,indent=2)+'\n')
