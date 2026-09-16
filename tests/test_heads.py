import pytest
import torch
from system_one import Boolean,Choice
from system_one.heads import CompiledHead,features,identity,schema
from test_cache import tiny_model


def test_probe_fit_persistence(tmp_path):
    x=torch.tensor([[-2.,0.],[-1.,.1],[1.,-.1],[2.,0.]])
    f=Boolean('?');meta={'schema':schema(f)}
    head=CompiledHead.fit(x,[0,0,1,1],metadata=meta,components=1)
    assert head.logits(x).argmax(1).tolist()==[0,0,1,1]
    head.save(tmp_path/'head');loaded=CompiledHead.load(tmp_path/'head')
    assert torch.equal(loaded.logits(x),head.logits(x))
    with pytest.raises(FileExistsError):head.save(tmp_path/'head')
    with pytest.raises(ValueError):CompiledHead.fit(x,[0,0,0,0],metadata=meta)


def test_compile_mixed_and_reload(tmp_path):
    model=tiny_model();f=Boolean('Is it positive?')
    rows=[{'id':str(i),'context':text,'value':value} for i,(text,value) in enumerate([
        ('sunny happy',True),('bright cheerful',True),('dark sad',False),('gloomy miserable',False)])]
    head=model.compile('sentiment',f,rows,components=2)
    assert head.metadata['training_n']==4
    assert all(p.grad is None for p in model.model.parameters())
    output=model.decide('sunny happy',{'sentiment':f,'other':Boolean('Other?')})
    assert output['sentiment'].strategy=='compiled' and output['sentiment'].candidate_mass is None
    assert output['other'].strategy=='dynamic'
    assert output['sentiment'].token_ids==()
    head.save(tmp_path/'head');model.remove_head('sentiment');model.load_head(tmp_path/'head')
    again=model.decide('sunny happy',{'sentiment':f})['sentiment']
    assert again.probabilities==output['sentiment'].probabilities
    with pytest.raises(ValueError):model.decide('',{'sentiment':Boolean('Different question?')})
    with pytest.raises(ValueError):model.decide_balanced('',{'sentiment':f})
    with pytest.raises(ValueError):model.decide('',{'sentiment':f},option_orders={'sentiment':(1,0)})
    model.model_id='other'
    with pytest.raises(ValueError):model.load_head(tmp_path/'head')


def test_training_validation_and_features():
    model=tiny_model();field=Boolean('?')
    with pytest.raises(ValueError):model.compile('x',field,[])
    rows=[{'id':'a','context':'x','value':True},{'id':'a','context':'y','value':False}]
    with pytest.raises(ValueError):model.compile('x',field,rows)
    rows[1]['id']='b';rows[0]['value']=1
    with pytest.raises(ValueError):model.compile('x',field,rows)
    with pytest.raises(ValueError):features(model,['x'],99)
    batched=features(model,['short','a longer sentence'])
    single=features(model,['short'])
    assert torch.allclose(batched[0],single[0],atol=1e-6)

def test_named_model_artifact_portability(tmp_path):
    first=tiny_model();second=tiny_model()
    for model in (first,second):
        model.model_id='tiny-fixture';model.revision='seed21-v1'
    f=Boolean('?')
    rows=[{'id':'a','context':'bright','value':True},{'id':'b','context':'dark','value':False}]
    first.compile('x',f,rows,components=1).save(tmp_path/'head')
    second.load_head(tmp_path/'head')
    assert first.decide('bright',{'x':f})['x'].probabilities==second.decide('bright',{'x':f})['x'].probabilities
