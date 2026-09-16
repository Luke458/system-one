import json
import pytest
from system_one import Boolean, Choice, DecisionModel
from system_one.quality_audit import option_orders,quality,load_dataset,summarize
from test_engine import FakeLM,Tokenizer
from test_cache import tiny_model


def test_option_mapping():
    model=DecisionModel(FakeLM(),Tokenizer())
    fields={'flag':Boolean('?')}
    normal=model.decide('',fields)['flag']
    reverse=model.decide('',fields,option_orders={'flag':(1,0)})['flag']
    assert normal.value is True and reverse.value is False
    assert reverse.values==(False,True)
    assert reverse.token_ids==(66,65)
    assert reverse.probabilities==tuple(reversed(normal.probabilities))
    prompt,labels=model.field_prompt('',fields['flag'],(1,0))
    assert 'A: true\nB: false' in prompt
    assert labels==('B','A')


@pytest.mark.parametrize('order',[(0,0),(0,),(-1,0),(True,False),(1,2)])
def test_bad_order(order):
    model=DecisionModel(FakeLM(),Tokenizer())
    with pytest.raises(ValueError):model.decide('',{'x':Boolean('?')},option_orders={'x':order})
    with pytest.raises(ValueError):model.decide('',{'x':Boolean('?')},option_orders={'other':(0,1)})


def test_permuted_cached_parity():
    model=tiny_model();fields={'x':Choice('?',('a','b','c')),'y':Boolean('Longer?')}
    orders={'x':(2,0,1),'y':(1,0)}
    full=model.decide('context',fields,option_orders=orders)
    cached=model.decide('context',fields,option_orders=orders,execution='shared_prefix')
    for k in fields:
        assert cached[k].values==full[k].values
        assert cached[k].probabilities==pytest.approx(full[k].probabilities,abs=2e-6)


def test_metrics_and_orders():
    assert len(option_orders(4))==24
    assert len(option_orders(5))==5
    result=quality([[.9,.1],[.3,.7]],[0,0])
    assert result['accuracy']==.5
    assert result['risk_coverage'][1]['coverage']==.5
    assert result['risk_coverage'][1]['error_rate']==0
    assert result['risk_coverage'][-1]['error_rate'] is None
    rows=[{'field':'x','case':'a','option_order':[0,1],'value':False,'probabilities':[.9,.1],'label':0},
          {'field':'x','case':'a','option_order':[1,0],'value':True,'probabilities':[.2,.8],'label':0}]
    summary=summarize(rows,{'x':Boolean('?')})['x']
    assert summary['unstable_cases']==1
    assert summary['disagreement_with_identity']==1
    assert summary['max_class_probability_span']==pytest.approx(.7)


def test_dataset_validation(tmp_path):
    path=tmp_path/'data.json'
    data={'fields':{'x':{'type':'boolean','question':'?'}},'cases':[{'id':'a','context':'text','expected':{'x':True}}]}
    path.write_text(json.dumps(data));assert len(load_dataset(path)[0])==1
    data['cases'][0]['expected']['x']=1
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError):load_dataset(path)
    data['cases'][0]['expected']['x']=True
    data['cases']*=2;path.write_text(json.dumps(data))
    with pytest.raises(ValueError):load_dataset(path)
