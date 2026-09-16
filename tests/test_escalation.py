from dataclasses import replace
import pytest
from system_one import Boolean,DecisionModel,EscalationPolicy,decide_with_fallback
from test_engine import FakeLM,Tokenizer


def test_explicit_fallback_and_skip():
    model=DecisionModel(FakeLM(),Tokenizer());fields={'x':Boolean('?')}
    calls=[]
    def fallback(context,subset):
        calls.append((context,subset));return model.decide(context,subset)
    out=decide_with_fallback(model,'context',fields,policy=EscalationPolicy(.99),fallback=fallback)
    assert len(calls)==1 and out.sources['x']=='fallback'
    assert out.reasons['x']==('low_probability',)
    decide_with_fallback(model,'context',fields,policy=EscalationPolicy(0),fallback=fallback)
    assert len(calls)==1


def test_invalid_fallback_fails_closed():
    model=DecisionModel(FakeLM(),Tokenizer());fields={'x':Boolean('?')}
    with pytest.raises(ValueError):decide_with_fallback(model,'',fields,policy=EscalationPolicy(1),fallback=lambda *_:{})
    d=model.decide('',fields)['x']
    with pytest.raises(ValueError):decide_with_fallback(model,'',fields,policy=EscalationPolicy(1),fallback=lambda *_:{'x':replace(d,probabilities=(.9,.9))})
    def failure(*_):raise RuntimeError('Provider failed')
    with pytest.raises(RuntimeError):decide_with_fallback(model,'',fields,policy=EscalationPolicy(1),fallback=failure)
    for x in [True,-1,2,float('nan')]:
        with pytest.raises(ValueError):EscalationPolicy(x)


def test_balanced_disagreement_escalates():
    model=DecisionModel(FakeLM(),Tokenizer());fields={'x':Boolean('?')}
    out=decide_with_fallback(model,'',fields,policy=EscalationPolicy(0),balanced=True,
                            fallback=lambda context,fields:model.decide(context,fields))
    assert 'option_order_disagreement' in out.reasons['x']
