"""Frozen-LM PCA-whitened linear heads; no backbone training or pickle loading."""
from dataclasses import dataclass
import hashlib
import json
import uuid
from pathlib import Path
import torch
from safetensors.torch import save_file, load_file
from .engine import Decision, validate_fields


def schema(field):
    return {'question':field.question,'values':list(field.values),
            'types':[type(v).__name__ for v in field.values]}


def identity(model):
    data={'model':getattr(model,'model_id',None),'revision':getattr(model,'revision',None),
          'config':model.model.config.to_dict(), 'template':model.tokenizer.chat_template if hasattr(model.tokenizer,'chat_template') else None,
          'tokenizer':type(model.tokenizer).__name__,
          'dtype':str(next(model.model.parameters()).dtype),'feature_format':'context_mean_v1'}
    if data['model'] is None or data['revision'] is None:
        # Anonymous in-memory models cannot safely exchange heads by architecture alone.
        if not hasattr(model, '_head_identity_nonce'):
            model._head_identity_nonce = uuid.uuid4().hex
        data['instance'] = model._head_identity_nonce
    # Config records runtime paths and attention implementation in some versions.
    for key in ('_name_or_path','_commit_hash','transformers_version'):
        data['config'].pop(key,None)
    return hashlib.sha256(json.dumps(data,sort_keys=True,default=str).encode()).hexdigest()


@torch.inference_mode()
def features(model,contexts,layer=-1):
    if type(layer) is not int:raise ValueError('layer must be an integer')
    if not contexts or any(type(x) is not str for x in contexts):raise ValueError('Expected nonempty string contexts')
    rows=[]
    for start in range(0,len(contexts),model.field_batch_size):
        prompts=[model.chat('Represent the following context for classification.\nContext: '+json.dumps(c))
                 for c in contexts[start:start+model.field_batch_size]]
        inputs=model.encode(prompts)
        positions=(inputs.attention_mask.cumsum(-1)-1).clamp_min(0)
        # Use the backbone: avoid materializing vocabulary logits.
        out=model.model.base_model(**inputs,position_ids=positions,use_cache=False,output_hidden_states=True,return_dict=True)
        if not -len(out.hidden_states)<=layer<len(out.hidden_states):raise ValueError('Hidden-state layer out of range')
        hidden=out.hidden_states[layer].float()
        mask=inputs.attention_mask.unsqueeze(-1)
        pooled=(hidden*mask).sum(1)/mask.sum(1)
        rows.append(pooled.cpu())
    result=torch.cat(rows)
    if not torch.isfinite(result).all():raise RuntimeError('Nonfinite hidden features')
    return result


@dataclass
class CompiledHead:
    mean: torch.Tensor
    projection: torch.Tensor
    weight: torch.Tensor
    bias: torch.Tensor
    metadata: dict

    @classmethod
    def fit(cls,x,y,*,metadata,components=32,l2=0.01):
        x=torch.as_tensor(x,dtype=torch.float64,device='cpu').clone()
        y=torch.as_tensor(y,device='cpu')
        nclasses=len(metadata['schema']['values'])
        if x.ndim!=2 or len(x)<2 or not torch.isfinite(x).all():raise ValueError('Invalid training features')
        if y.ndim!=1 or len(y)!=len(x) or y.dtype==torch.bool or y.is_floating_point():raise ValueError('Invalid labels')
        if set(y.tolist())!=set(range(nclasses)):raise ValueError('Training must contain every class')
        if type(components)is not int or components<1 or not 0<l2<1e6:raise ValueError('Invalid fit parameters')
        mean=x.mean(0);centered=x-mean
        _,s,vh=torch.linalg.svd(centered,full_matrices=False)
        rank=min(components,len(x)-1,x.shape[1])
        scale=(s[:rank]/(len(x)-1)**.5).clamp_min(1e-4)
        projection=vh[:rank].T/scale
        z=centered@projection
        weight=torch.zeros((rank,nclasses),dtype=x.dtype,requires_grad=True)
        bias=torch.zeros(nclasses,dtype=x.dtype,requires_grad=True)
        optimizer=torch.optim.LBFGS([weight,bias],max_iter=100,line_search_fn='strong_wolfe')
        def closure():
            optimizer.zero_grad()
            loss=torch.nn.functional.cross_entropy(z@weight+bias,y.long())+l2*weight.square().sum()/2
            loss.backward();return loss
        optimizer.step(closure)
        return cls(mean.float(),projection.float(),weight.detach().float(),bias.detach().float(),dict(metadata,training_n=len(x),components=rank,l2=l2))

    def logits(self,x):
        return (torch.as_tensor(x,dtype=torch.float32,device='cpu')-self.mean)@self.projection@self.weight+self.bias

    def decision(self,x,field,temperature=1.,applied=False):
        if schema(field)!=self.metadata['schema']:raise ValueError('Compiled field semantics changed')
        logits=self.logits(x).reshape(-1)
        if not torch.isfinite(logits).all():raise RuntimeError('Nonfinite head logits')
        raw=logits.softmax(0);p=(logits/temperature).softmax(0)
        return Decision(field.values[int(p.argmax())],tuple(field.values),tuple(p.tolist()),tuple(raw.tolist()),
                        tuple(logits.tolist()),None,(),temperature,applied,'compiled')

    def save(self,directory):
        target=Path(directory)
        target.mkdir(parents=True,exist_ok=False)
        save_file({k:getattr(self,k).contiguous() for k in ('mean','projection','weight','bias')},str(target/'head.safetensors'))
        (target/'metadata.json').write_text(json.dumps(self.metadata,indent=2)+'\n')

    @classmethod
    def load(cls,directory):
        directory=Path(directory);t=load_file(str(directory/'head.safetensors'))
        meta=json.loads((directory/'metadata.json').read_text())
        if set(t)!={'mean','projection','weight','bias'} or any(not torch.isfinite(v).all() for v in t.values()):raise ValueError('Invalid head tensors')
        m,p,w,b=(t[k] for k in ('mean','projection','weight','bias'))
        if m.ndim!=1 or p.ndim!=2 or w.ndim!=2 or b.ndim!=1 or p.shape[0]!=len(m) or p.shape[1]!=w.shape[0] or w.shape[1]!=len(b) or len(b)!=len(meta['schema']['values']):raise ValueError('Invalid head dimensions')
        return cls(m,p,w,b,meta)


def compile_field(model,name,field,examples,*,layer=-1,components=32,l2=.01):
    validate_fields({name:field})
    if not examples:raise ValueError('Training examples required')
    ids=[r['id'] for r in examples]
    if any(type(i)is not str or not i for i in ids) or len(ids)!=len(set(ids)):raise ValueError('Training IDs must be unique strings')
    labels=[]
    for r in examples:
        matches=[i for i,v in enumerate(field.values) if type(v)is type(r['value']) and v==r['value']]
        if not matches:raise ValueError('Invalid typed training label')
        labels.append(matches[0])
    # Only explicitly supplied training examples influence PCA and probe fitting.
    x=features(model,[r['context'] for r in examples],layer)
    head=CompiledHead.fit(x,labels,components=components,l2=l2,
        metadata={'format_version':1,'field':name,'schema':schema(field),'identity':identity(model),
                  'layer':layer,'training_ids':ids,'feature_format':'context_mean_v1'})
    model._compiled_heads[name]=head
    return head
