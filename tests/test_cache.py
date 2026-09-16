import pytest
import torch
from transformers import LlamaConfig, LlamaForCausalLM, GPT2Config, GPT2LMHeadModel
from system_one import Boolean, Choice, DecisionModel
from system_one.cache import common_prefix_length
from test_engine import Tokenizer


def tiny_model():
    torch.manual_seed(21)
    return DecisionModel(LlamaForCausalLM(LlamaConfig(vocab_size=256, hidden_size=32,
        intermediate_size=48, num_hidden_layers=2, num_attention_heads=4,
        num_key_value_heads=2, max_position_embeddings=2048)), Tokenizer())


@pytest.mark.parametrize('count,batch_size', [(1,1),(2,2),(5,2),(5,8)])
@pytest.mark.parametrize('context', ['', 'context '*40])
def test_cached_parity(count,batch_size,context):
    model = tiny_model()
    model.field_batch_size = batch_size
    fields = {f'f{i}': Choice('Question ' + 'long '*i + '?', ('yes','no','maybe')) for i in range(count)}
    full = model.decide(context, fields)
    cached = model.decide(context, fields, execution='shared_prefix')
    reverse = model.decide(context, dict(reversed(list(fields.items()))), execution='shared_prefix')
    for key in fields:
        assert cached[key].logits == pytest.approx(full[key].logits, abs=2e-6)
        assert cached[key].probabilities == pytest.approx(full[key].probabilities, abs=2e-6)
        assert cached[key].candidate_mass == pytest.approx(full[key].candidate_mass, abs=2e-6)
        assert reverse[key].logits == pytest.approx(cached[key].logits, abs=2e-6)
    # Fresh request after unrelated cached inference must not retain/mutate state.
    model.decide('unrelated', {'other': Boolean('Other question?')}, execution='shared_prefix')
    again = model.decide(context, fields, execution='shared_prefix')
    for key in fields:
        assert again[key].logits == pytest.approx(cached[key].logits, abs=2e-6)


def test_prefix():
    assert common_prefix_length([[1,2,3],[1,2,4]]) == 2
    assert common_prefix_length([[1,2,3],[1,2,3]]) == 2
    assert common_prefix_length([[1,2,3]]) == 2
    assert common_prefix_length([[1,2],[3,4]]) == 0
    with pytest.raises(ValueError): common_prefix_length([[]])


def test_identical_fields_and_temperature():
    model=tiny_model()
    fields={'a':Boolean('?'),'b':Boolean('?')}
    result=model.decide('test',fields,execution='shared_prefix',temperatures={'a':2})
    assert result['a'].logits == result['b'].logits
    assert result['a'].probability_kind == 'temperature_scaled'
    assert result['b'].probability_kind == 'raw'
    assert result['b'].raw_probability == result['b'].probability
    assert result['a'].raw_probability == result['b'].raw_probability
    assert result['a'].probability <= result['a'].raw_probability


def test_invalid_backend_and_overflow():
    model=tiny_model()
    with pytest.raises(ValueError): model.decide('',{'x':Boolean('?')},execution='bad')
    model.model.config.sliding_window=32
    with pytest.raises(ValueError,match='full-attention'): model.decide('',{'x':Boolean('?')},execution='shared_prefix')
    model.model.config.sliding_window=None
    model.max_input_tokens=4
    with pytest.raises(ValueError,match='max_input_tokens'): model.decide('',{'x':Boolean('?')},execution='shared_prefix')


def test_second_architecture_without_logits_to_keep():
    torch.manual_seed(21)
    lm=GPT2LMHeadModel(GPT2Config(vocab_size=256,n_embd=32,n_layer=2,
        n_head=4,n_positions=1024))
    original=lm.forward
    def forward(input_ids,attention_mask,position_ids,use_cache,past_key_values=None):
        return original(input_ids=input_ids,attention_mask=attention_mask,position_ids=position_ids,
                        use_cache=use_cache,past_key_values=past_key_values)
    lm.forward=forward
    model=DecisionModel(lm,Tokenizer())
    fields={'a':Boolean('?'),'b':Choice('Longer question?',('x','y','z'))}
    assert not model._supports_logits_to_keep
    full=model.decide('context',fields)
    cached=model.decide('context',fields,execution='shared_prefix')
    for k in fields:
        assert cached[k].logits == pytest.approx(full[k].logits,abs=2e-6)
