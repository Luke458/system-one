"""Descriptive option-order audit. Never fits or selects a prompt on evaluation data."""
import argparse
import hashlib
import json
from pathlib import Path
import platform
import time

import torch
import transformers
from .engine import Boolean, Choice, DecisionModel, schema_valid, validate_fields
from .calibration import brier_score, expected_calibration_error, reliability_data
from .qc import CASES, FIELDS
from .orders import option_orders




def load_dataset(path=None):
    if path is None:
        return CASES, FIELDS, 'synthetic_qc_smoke'
    data=json.loads(Path(path).read_text())
    fields={}
    for name,spec in data['fields'].items():
        if spec['type']=='boolean': fields[name]=Boolean(spec['question'])
        elif spec['type']=='choice': fields[name]=Choice(spec['question'],spec['values'])
        else: raise ValueError('Unknown field type')
    validate_fields(fields)
    cases=data['cases']
    if not cases: raise ValueError('Dataset must contain cases')
    seen=set()
    for case in cases:
        if type(case['id']) is not str or not case['id'] or case['id'] in seen:
            raise ValueError('Case IDs must be unique nonempty strings')
        seen.add(case['id'])
        if type(case['context']) is not str or not schema_valid(case['expected'],fields):
            raise ValueError('Invalid context or expected typed labels')
    return cases,fields,data.get('name','external_dataset')


def quality(probabilities, labels):
    p=torch.as_tensor(probabilities,dtype=torch.float64)
    y=torch.as_tensor(labels,dtype=torch.long)
    # Metric validators reject malformed input before further reductions.
    brier=brier_score(p,y)
    confidence,prediction=p.max(1)
    correct=prediction.eq(y)
    selective=[]
    for threshold in (.5,.9,.95,.99,.995):
        accepted=confidence >= threshold
        count=int(accepted.sum())
        selective.append({'threshold':threshold,'accepted':count,'coverage':count/len(y),
                          'error_rate':float((~correct[accepted]).double().mean()) if count else None})
    return {'n':len(y),'accuracy':float(correct.double().mean()),'brier':brier,
            'ece':expected_calibration_error(p,y),'reliability':reliability_data(p,y),
            'risk_coverage':selective}


def summarize(rows,fields):
    summaries={}
    for name,field in fields.items():
        selected=[r for r in rows if r['field']==name]
        identity=list(range(len(field.values)))
        original=[r for r in selected if r['option_order']==identity]
        ids=sorted({r['case'] for r in selected})
        unstable=0;max_span=0.;disagreement=0;comparisons=0
        for case_id in ids:
            variants=[r for r in selected if r['case']==case_id]
            baseline=next(r for r in variants if r['option_order']==identity)
            unstable+=len({json.dumps(r['value']) for r in variants})>1
            for r in variants:
                if r['option_order']!=identity:
                    disagreement+=r['value']!=baseline['value'];comparisons+=1
            for i in range(len(field.values)):
                probs=[r['probabilities'][i] for r in variants]
                max_span=max(max_span,max(probs)-min(probs))
        summaries[name]={
            'unique_cases':len(ids),'permutations_per_case':len(selected)//len(ids),
            'unstable_cases':unstable,'disagreement_with_identity':disagreement/comparisons if comparisons else 0,
            'max_class_probability_span':max_span,
            'identity':quality([r['probabilities'] for r in original],[r['label'] for r in original]),
            # Permutations are repeated measurements, NOT independent examples.
            'all_orders_descriptive':quality([r['probabilities'] for r in selected],[r['label'] for r in selected]),
        }
    return summaries


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--model',default='openbmb/MiniCPM5-2B')
    parser.add_argument('--revision',default='12a3808a956f869c767195e9266b59c4d21d92e2')
    parser.add_argument('--device',default='cuda')
    parser.add_argument('--execution',choices=['batched','shared_prefix'],default='batched')
    parser.add_argument('--dataset',default=None)
    parser.add_argument('--output',default='reports/quality-audit.json')
    args=parser.parse_args()
    cases,fields,name=load_dataset(args.dataset)
    model=DecisionModel.from_pretrained(args.model,revision=args.revision,device=args.device)
    rows=[]
    for case in cases:
        for field_name,field in fields.items():
            for order in option_orders(len(field.values)):
                decision=model.decide(case['context'],{field_name:field},execution=args.execution,
                                      option_orders={field_name:order})[field_name]
                rows.append({'case':case['id'],'field':field_name,'option_order':list(order),
                             'value':decision.value,'values':list(field.values),
                             'probabilities':list(decision.raw_probabilities),
                             'logits':list(decision.logits),'candidate_mass':decision.candidate_mass,
                             'token_ids':list(decision.token_ids),
                             'label':field.values.index(case['expected'][field_name])})
            print(f"Audited {case['id']} / {field_name}",flush=True)
    spec={k:{'type':'boolean' if isinstance(f,Boolean) else 'choice','question':f.question,
             'values':list(f.values)} for k,f in fields.items()}
    dataset={'name':name,'fields':spec,'cases':cases}
    report={'timestamp_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),
            'model':model.model_id,'revision':model.revision,'settings':vars(args),
            'dataset':dataset,'dataset_sha256':hashlib.sha256(json.dumps(dataset,sort_keys=True).encode()).hexdigest(),
            'environment':{'torch':torch.__version__,'transformers':transformers.__version__,
                'python':platform.python_version(),'dtype':str(next(model.model.parameters()).dtype),
                'device':str(model.device),'gpu':torch.cuda.get_device_name(model.device) if model.device.type=='cuda' else None},
            'notes':['Each field is scored alone, holding batch shape fixed across option permutations.',
                     'All permutations for <=4 classes; cyclic orders only for larger fields.',
                     'Repeated orders are not independent examples; no significance claims.',
                     'No temperatures fitted, no prompt/option order selected, no automated escalation.',
                     'Threshold curves are descriptive; they do not establish safe operating thresholds.'],
            'summary':summarize(rows,fields),'rows':rows}
    target=Path(args.output);target.parent.mkdir(parents=True,exist_ok=True)
    target.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:{'unstable_cases':v['unstable_cases'],'identity_accuracy':v['identity']['accuracy']}
                      for k,v in report['summary'].items()},indent=2))

if __name__=='__main__':main()
