from types import SimpleNamespace
import json
import pytest
import torch
from system_one import Boolean, Choice, DecisionModel
from system_one.engine import candidate_ids, schema_valid

class Tokenizer:
    pad_token_id = 0
    all_special_ids = [0]
    def encode(self, text, **kwargs):
        return [ord(c) for c in text]
    def decode(self, ids, **kwargs):
        return ''.join(chr(i) for i in ids)
    def apply_chat_template(self, messages, **kwargs):
        return messages[0]['content'] + '\nAnswer:\n'
    def __call__(self, texts, **kwargs):
        rows = [self.encode(t) for t in texts]
        n = max(map(len, rows))
        ids = torch.tensor([[0] * (n-len(r)) + r for r in rows])
        class Batch(dict):
            __getattr__ = dict.__getitem__
            def to(self, device):
                return Batch({k: v.to(device) for k, v in self.items()})
        return Batch(input_ids=ids, attention_mask=(ids != 0).long())

class FakeLM(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(1))
        self.calls = 0
    def forward(self, input_ids, attention_mask, position_ids, use_cache, logits_to_keep):
        self.calls += 1
        assert not use_cache and logits_to_keep == 1
        assert torch.equal(position_ids, (attention_mask.cumsum(-1)-1).clamp_min(0))
        logits = torch.zeros((len(input_ids), 1, 256))
        logits[:, :, 66] = 2
        return SimpleNamespace(logits=logits)
    def generate(self, **kwargs):
        raise AssertionError('decide must never generate')

@pytest.fixture
def engine():
    return DecisionModel(FakeLM(), Tokenizer())

def test_typed_decisions_and_mass(engine):
    fields = {'flag': Boolean('Issue?'), 'cause': Choice('Cause?', ('a long value', 'b'))}
    out = engine.decide('context', fields)
    assert out['flag'].value is True
    assert out['cause'].value == 'b'
    assert engine.model.calls == 1
    assert sum(out['flag'].probabilities) == pytest.approx(1)
    assert out['flag'].candidate_mass < .1
    assert schema_valid({k: v.value for k,v in out.items()}, fields)
    assert json.loads(json.dumps(out['flag'].to_dict()))['distribution'][0]['value'] is False

def test_temperature(engine):
    a = engine.decide('', {'x': Boolean('?')})['x']
    b = engine.decide('', {'x': Boolean('?')}, temperatures={'x': 2})['x']
    assert a.confidence > b.confidence
    assert a.raw_probabilities == b.raw_probabilities
    assert a.candidate_mass == b.candidate_mass

@pytest.mark.parametrize('temp', [0, -1, float('nan'), float('inf'), True])
def test_bad_temperature(engine, temp):
    with pytest.raises(ValueError):
        engine.decide('', {'x': Boolean('?')}, temperatures={'x': temp})

def test_validation(engine):
    for options in [('x',), ('a', 'a'), ('', 'x'), tuple(str(i) for i in range(27))]:
        with pytest.raises(ValueError): Choice('?', options)
    for fields in [{}, {'': Boolean('?')}, {'x': Boolean('')}]:
        with pytest.raises(ValueError): engine.decide('', fields)
    with pytest.raises(TypeError): engine.decide('', {'x': 'bad'})
    with pytest.raises(ValueError): engine.decide('', {'x': Boolean('?')}, temperatures={'z': 1})
    engine.max_input_tokens = 2
    with pytest.raises(ValueError): engine.decide('long', {'x': Boolean('?')})

def test_chunking(engine):
    engine.field_batch_size = 1
    engine.decide('', {'a': Boolean('?'), 'b': Boolean('longer?')})
    assert engine.model.calls == 2

def test_exact_schema():
    fields = {'a': Boolean('?')}
    assert schema_valid({'a': True}, fields)
    for value in [{'a': 1}, {'a': 'true'}, {'a': True, 'b': False}, [], None]:
        assert not schema_valid(value, fields)

def test_candidates():
    tok = Tokenizer()
    assert candidate_ids(tok, 'prompt\n', ['A', 'B']) == (65, 66)
    for labels in [['AA', 'B'], ['A', 'A'], ['\x00', 'B']]:
        with pytest.raises(ValueError): candidate_ids(tok, 'prompt\n', labels)
    class MergeTokenizer(Tokenizer):
        def encode(self, text, **kwargs):
            return [1, 65] if text.endswith('A') else super().encode(text)
    with pytest.raises(ValueError): candidate_ids(MergeTokenizer(), 'prompt\n', ['A'])

def test_nonfinite(engine):
    def bad(**kwargs):
        return SimpleNamespace(logits=torch.full((1,1,256), float('nan')))
    engine.model.forward = bad
    with pytest.raises(RuntimeError): engine.decide('', {'x': Boolean('?')})

def test_real_tiny_llama_batch_equivalence():
    from transformers import LlamaConfig, LlamaForCausalLM
    torch.manual_seed(1)
    lm = LlamaForCausalLM(LlamaConfig(vocab_size=256, hidden_size=16,
        intermediate_size=32, num_hidden_layers=1, num_attention_heads=2,
        num_key_value_heads=1, max_position_embeddings=2048))
    model = DecisionModel(lm, Tokenizer())
    fields = {'a': Boolean('Short?'), 'b': Choice('A much longer question?', ('one','two','three'))}
    together = model.decide('hello', fields)
    for k, f in fields.items():
        alone = model.decide('hello', {k: f})[k]
        assert together[k].logits == pytest.approx(alone.logits, abs=1e-6)
