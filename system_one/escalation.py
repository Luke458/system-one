"""Explicit, synchronous fallback callbacks. No provider/network calls are built in."""
from dataclasses import dataclass
import math
from .engine import Decision, schema_valid


@dataclass(frozen=True)
class EscalationPolicy:
    min_probability: float
    min_candidate_mass: float = 0.

    def __post_init__(self):
        for v in (self.min_probability,self.min_candidate_mass):
            if isinstance(v,bool) or not isinstance(v,(int,float)) or not math.isfinite(v) or not 0<=v<=1:
                raise ValueError('Policy thresholds must be finite probabilities')

    def reasons(self,decision):
        result=[]
        if decision.probability<self.min_probability:result.append('low_probability')
        mass=getattr(decision,'candidate_mass',getattr(decision,'candidate_mass_min',None))
        if self.min_candidate_mass>0 and (mass is None or mass<self.min_candidate_mass):result.append('candidate_mass_unavailable_or_low')
        if hasattr(decision,'order_stable') and not decision.order_stable:result.append('option_order_disagreement')
        return tuple(result)


@dataclass(frozen=True)
class EscalationResult:
    decisions: dict
    initial: dict
    reasons: dict
    sources: dict


def validate_decision(decision,field):
    if not isinstance(decision,Decision):raise TypeError('Fallback must return Decision objects')
    if not schema_valid({'x':decision.value},{'x':field}):raise ValueError('Fallback returned an invalid typed value')
    if len(decision.values)!=len(field.values) or any(type(a)is not type(b) or a!=b for a,b in zip(decision.values,field.values)):
        raise ValueError('Fallback class order does not match')
    p=decision.probabilities
    if len(p)!=len(field.values) or any(not math.isfinite(v) or not 0<=v<=1 for v in p) or abs(sum(p)-1)>1e-5:
        raise ValueError('Fallback returned invalid probabilities')
    if decision.value!=field.values[max(range(len(p)),key=lambda i:p[i])]:raise ValueError('Fallback value disagrees with probabilities')


def decide_with_fallback(model,context,fields,*,policy,fallback,balanced=False,execution='batched'):
    if not isinstance(policy,EscalationPolicy) or not callable(fallback):raise TypeError('An explicit policy and callback are required')
    initial=(model.decide_balanced if balanced else model.decide)(context,fields,execution=execution)
    reasons={k:r for k,d in initial.items() if (r:=policy.reasons(d))}
    decisions=dict(initial);sources={k:'local' for k in fields}
    if reasons:
        subset={k:fields[k] for k in reasons}
        replacement=fallback(context,subset)
        if not isinstance(replacement,dict) or set(replacement)!=set(subset):raise ValueError('Fallback must return exactly the requested fields')
        for k,d in replacement.items():validate_decision(d,fields[k])
        decisions.update(replacement);sources.update({k:'fallback' for k in replacement})
    return EscalationResult(decisions,initial,reasons,sources)
