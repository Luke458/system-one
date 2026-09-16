import pytest
from system_one import Boolean,Choice,DecisionModel
from test_engine import FakeLM,Tokenizer
from test_cache import tiny_model


def test_removes_pure_letter_bias_and_exposes_disagreement():
    model=DecisionModel(FakeLM(),Tokenizer())
    result=model.decide_balanced('',{'flag':Boolean('?')})['flag']
    assert result.probabilities==pytest.approx((.5,.5),abs=1e-7)
    assert result.agreement_fraction==.5
    assert not result.order_stable
    assert len(result.variant_probabilities)==2
    assert result.probability_kind=='order_averaged_raw'
    assert result.to_dict()['order_stable'] is False


def test_exhaustive_class_reordering_and_cached_path():
    model=tiny_model()
    first=model.decide_balanced('hello',{'x':Choice('Which?',('one','two','three'))})['x']
    reverse=model.decide_balanced('hello',{'x':Choice('Which?',('three','two','one'))})['x']
    assert first.probabilities==pytest.approx(tuple(reversed(reverse.probabilities)),abs=2e-6)
    cached=model.decide_balanced('hello',{'x':Choice('Which?',('one','two','three'))},execution='shared_prefix')['x']
    assert cached.probabilities==pytest.approx(first.probabilities,abs=2e-6)
    assert len(first.option_orders)==6


def test_fields_and_large_choices():
    model=DecisionModel(FakeLM(),Tokenizer())
    fields={'0:0':Boolean('?'),'more':Choice('?',tuple('abcdef'))}
    result=model.decide_balanced('',fields)
    assert set(result)==set(fields)
    assert len(result['more'].option_orders)==6
    assert result['more'].probabilities==pytest.approx([1/6]*6,abs=1e-7)
    with pytest.raises(ValueError):model.decide_balanced('',{})


def test_internal_alias_does_not_dispatch_unrelated_compiled_head():
    model=DecisionModel(FakeLM(),Tokenizer())
    model._compiled_heads['0:0']=object()
    result=model.decide_balanced('',{'flag':Boolean('?')})['flag']
    assert result.probabilities==pytest.approx((.5,.5),abs=1e-7)
