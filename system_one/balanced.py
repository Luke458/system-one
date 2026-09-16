"""Average canonical class probabilities over balanced answer presentations."""
from dataclasses import dataclass
import torch
from .engine import validate_fields
from .orders import option_orders


@dataclass(frozen=True)
class BalancedDecision:
    value: bool | str
    values: tuple[bool | str, ...]
    probabilities: tuple[float, ...]
    agreement_fraction: float
    class_probability_span: tuple[float, ...]
    candidate_mass_min: float
    option_orders: tuple[tuple[int, ...], ...]
    variant_probabilities: tuple[tuple[float, ...], ...]

    @property
    def probability(self):
        return max(self.probabilities)

    @property
    def probability_kind(self):
        return 'order_averaged_raw'

    @property
    def order_stable(self):
        return self.agreement_fraction == 1.0

    def to_dict(self):
        return {'value':self.value,'probability':self.probability,'probability_kind':self.probability_kind,
                'distribution':[{'value':v,'probability':p} for v,p in zip(self.values,self.probabilities)],
                'order_stable':self.order_stable,'agreement_fraction':self.agreement_fraction,
                'class_probability_span':list(self.class_probability_span),
                'candidate_mass_min':self.candidate_mass_min,
                'option_orders':[list(x) for x in self.option_orders],
                'variant_probabilities':[list(x) for x in self.variant_probabilities]}


def decide_balanced(model,context,fields,*,execution='batched'):
    """No fitted calibration. Disagreement is exposed, never used to trigger actions.

    Uses all permutations for <=4 classes and cyclic rotations for larger sets.
    The latter balances positions but is not invariant to every possible ordering.
    """
    validate_fields(fields)
    if set(fields) & set(model._compiled_heads):
        raise ValueError("Balanced scoring is for dynamic fields; use decide for compiled heads")
    expanded={};orders={};groups={}
    for name,field in fields.items():
        groups[name]=[]
        for index,order in enumerate(option_orders(len(field.values))):
            alias=f'{len(groups)-1}:{index}'
            while alias in model._compiled_heads or alias in expanded:
                alias = '_' + alias
            expanded[alias]=field;orders[alias]=order;groups[name].append(alias)
    variants=model.decide(context,expanded,option_orders=orders,execution=execution)
    result={}
    for name,aliases in groups.items():
        probabilities=torch.tensor([variants[a].raw_probabilities for a in aliases],dtype=torch.float64)
        means=probabilities.mean(0)
        selected=int(means.argmax())
        winners=probabilities.argmax(1)
        result[name]=BalancedDecision(
            fields[name].values[selected],tuple(fields[name].values),tuple(means.tolist()),
            float((winners==selected).double().mean()),
            tuple((probabilities.max(0).values-probabilities.min(0).values).tolist()),
            min(variants[a].candidate_mass for a in aliases),tuple(orders[a] for a in aliases),
            tuple(tuple(row.tolist()) for row in probabilities))
    return result
