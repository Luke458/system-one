"""Measure shared-prefix break-even and numeric drift, not task quality."""
import argparse
import json
import math
from pathlib import Path
import platform
import statistics
import time

import torch
import transformers
from .engine import DecisionModel
from .cache import common_prefix_length
from .benchmark import timed
from .qc import CASES, FIELDS


def compare(reference, actual):
    return {
        'field_count': len(reference),
        'decision_disagreements': sum(reference[k].value != actual[k].value for k in reference),
        'max_logit_delta': max(abs(a-b) for k in reference for a,b in zip(reference[k].logits,actual[k].logits)),
        'max_probability_delta': max(abs(a-b) for k in reference for a,b in zip(reference[k].probabilities,actual[k].probabilities)),
        'max_candidate_mass_delta': max(abs(reference[k].candidate_mass-actual[k].candidate_mass) for k in reference),
    }


def distribution(values):
    values=sorted(values)
    return {'median_ms':statistics.median(values), 'p95_ms':values[math.ceil(.95*len(values))-1]}


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--model',default='openbmb/MiniCPM5-2B')
    p.add_argument('--revision',default='12a3808a956f869c767195e9266b59c4d21d92e2')
    p.add_argument('--device',default='cuda')
    p.add_argument('--context-tokens',type=int,nargs='+',default=[128,1024,4096])
    p.add_argument('--field-counts',type=int,nargs='+',default=[1,4,16])
    p.add_argument('--batch-size',type=int,default=4)
    p.add_argument('--repeats',type=int,default=3)
    p.add_argument('--output',default='reports/cache-benchmark.json')
    args=p.parse_args()
    if min(args.context_tokens+args.field_counts+[args.repeats,args.batch_size])<1:
        p.error('Counts must be positive')
    model=DecisionModel.from_pretrained(args.model,revision=args.revision,device=args.device,
        field_batch_size=args.batch_size,max_input_tokens=max(args.context_tokens)+2048)
    # Original QC cases test real questions and variable suffix lengths, unchanged.
    qc=[]
    for case in CASES:
        full=model.decide(case['context'],FIELDS)
        cached=model.decide(case['context'],FIELDS,execution='shared_prefix')
        qc.append({'case':case['id'],**compare(full,cached),
                   'batched':{k:v.to_dict() for k,v in full.items()},
                   'shared_prefix':{k:v.to_dict() for k,v in cached.items()}})
    rows=[]
    tokenizer=model.tokenizer
    filler=' Archive entry: store inventory received; counts reconciled; no further event recorded.\n'
    for budget in args.context_tokens:
        ids=tokenizer.encode(filler*(budget//8+1),add_special_tokens=False)[:budget]
        context=tokenizer.decode(ids,skip_special_tokens=True)
        real_context_tokens=len(tokenizer.encode(context,add_special_tokens=False))
        for count in args.field_counts:
            fields={f'field_{i}':list(FIELDS.values())[i%len(FIELDS)] for i in range(count)}
            prompts=[model.field_prompt(context,f)[0] for f in fields.values()]
            tokens=[tokenizer.encode(t,add_special_tokens=False) for t in prompts]
            prefix=common_prefix_length(tokens)
            timings={method:[] for method in ('batched','shared_prefix')}
            measurements=[]
            # One warmup per method at each shape; copy/tokenization costs remain timed.
            for method in timings:
                model.decide(context,fields,execution=method)
            for repeat in range(args.repeats):
                outputs={}
                order=list(timings) if repeat%2==0 else list(reversed(timings))
                for method in order:
                    out,ms,peak=timed(model.device,lambda: model.decide(context,fields,execution=method))
                    outputs[method]=out
                    timings[method].append(ms)
                    measurements.append({'repeat':repeat,'method':method,'latency_ms':ms,'peak_allocated_bytes':peak})
                parity=compare(outputs['batched'],outputs['shared_prefix'])
                measurements.append({'repeat':repeat,'parity':parity})
            # Approximate padded token positions computed; cached suffix padding is included.
            padded_full=sum(max(map(len,tokens[i:i+args.batch_size]))*len(tokens[i:i+args.batch_size])
                            for i in range(0,len(tokens),args.batch_size))
            padded_cached=prefix+sum((max(map(len,tokens[i:i+args.batch_size]))-prefix)*len(tokens[i:i+args.batch_size])
                                    for i in range(0,len(tokens),args.batch_size))
            stats={method:distribution(v) for method,v in timings.items()}
            row={'context_tokens':real_context_tokens,'requested_context_tokens':budget,'fields':count,
                 'prefix_tokens':prefix,'prompt_tokens':[len(t) for t in tokens],
                 'transformer_token_positions':{'batched':padded_full,'shared_prefix':padded_cached},
                 'timing':stats,'speedup':stats['batched']['median_ms']/stats['shared_prefix']['median_ms'],
                 'measurements':measurements}
            rows.append(row)
            print(f"context={real_context_tokens} fields={count} speedup={row['speedup']:.2f}x parity={parity}",flush=True)
    report={'timestamp_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),
            'model':model.model_id,'revision':model.revision,'settings':vars(args),
            'environment':{'python':platform.python_version(),'torch':torch.__version__,
                'transformers':transformers.__version__,'hip':torch.version.hip,
                'device':str(model.device),'gpu':torch.cuda.get_device_name(model.device) if model.device.type=='cuda' else None,
                'dtype':str(next(model.model.parameters()).dtype)},
            'notes':['End-to-end warm calls including validation, tokenization, cache copies and transfers.',
                     'Cache is rebuilt per call; no cache hits across requests.',
                     'Synthetic repeated fields/filler measure scaling, not semantic accuracy.',
                     'BF16 drift and decision disagreements are reported, not suppressed.'],
            'qc_parity':qc,'rows':rows}
    path=Path(args.output);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(report,indent=2)+'\n')

if __name__=='__main__':main()
