# Related Work and Updated Project Direction

## Key Finding

There does not currently appear to be a mature open-source project that combines all of the capabilities we are targeting:

1. Zero-training typed decisions over arbitrary questions.
2. Direct probability distributions rather than generated JSON/text.
3. Shared context prefill / KV-cache reuse across many decisions.
4. Empirical probability calibration.
5. Optional learned hidden-state heads for repeated decisions.
6. Automatic escalation to a larger reasoning model when confidence is insufficient.
7. A simple model-agnostic API that can sit on top of ordinary Hugging Face causal LMs.

The individual techniques already exist in adjacent projects and research. The potentially interesting contribution is combining them into a coherent **semantic decision runtime**.

---

# Related Work

## 1. LatentGate

Paper:

https://aclanthology.org/2026.acl-industry.153/

LatentGate is probably the closest academic relative to the proposed learned-head component.

Its basic architecture is:

```text
Frozen small language model
        ↓
intermediate hidden state
        ↓
mean pooling
        ↓
PCA whitening
        ↓
small linear classifier
        ↓
P(class)
```

The paper reports approximately:

- 98.8% in-domain routing accuracy across 100 agents.
- ~80% OOD accuracy in the reported 100-agent setting.
- ~28 ms inference latency on an NVIDIA T4.
- Very small learned classifier state.
- Sub-10-ms warm-start retraining of the lightweight probe as feasible.

An important finding is that raw causal-LM hidden states are highly anisotropic and are poor embeddings directly. PCA whitening substantially improves class separation.

### Difference from our project

LatentGate solves a fixed classification problem.

Conceptually:

```python
router.predict(context)
# -> one of the classes the router was trained on
```

Our zero-shot path should instead support arbitrary runtime decisions:

```python
engine.decide(
    context,
    fields={
        "anomaly": Boolean(),
        "cause": Choice(...),
        "severity": Choice(...),
        "escalate": Boolean(),
    },
)
```

without training a new classifier for every question.

LatentGate is therefore better viewed as the basis for a future **compiled decision head** rather than the entire runtime.

---

# 2. Hidden-State Control Is Becoming an Established Pattern

Several recent projects strengthen the hypothesis that intermediate LM representations can drive lightweight control systems.

## OverflowGuard

OverflowGuard uses a small MLP over mid-layer hidden states to decide whether a compressed representation can safely be used or whether inference should fall back to the full context.

Repository:

https://github.com/s-nlp/overflowguard

This demonstrates that hidden states can provide useful control signals without requiring autoregressive generation.

## Dr.LLM

Dr.LLM freezes the underlying language model and introduces very small routers around transformer blocks.

Conceptually:

```text
Transformer block
       ↓
tiny router
       ↓
skip / execute / repeat
       ↓
next block
```

The reported router parameters are a very small fraction of the underlying model.

Repository:

https://github.com/parameterlab/dr-llm

## Multi-Head Latent Control

Recent work has also explored attaching multiple lightweight heads to latent representations for separate control decisions such as:

```text
capability
tool use
clarify
abstain
answer
```

This is closely related to a possible future architecture for our runtime:

```text
                    MiniCPM5-2B
                         │
                   hidden states
                         │
        ┌────────────────┼────────────────┐
        ▼                ▼                ▼
   anomaly head      routing head     escalation head
        │                │                │
       T/F          tool A/B/C/D        T/F
```

The important broader observation is that **a causal LM's latent representations can be useful independently of its text-generation capability**.

---

# 3. Constrained Decoding Is Related but Not Equivalent

Existing libraries such as:

- Outlines
- XGrammar
- llguidance
- custom Hugging Face `LogitsProcessor` implementations

can constrain an autoregressive model to legal output structures.

For example:

```text
severity:
LOW | MEDIUM | HIGH
```

can be implemented by masking illegal vocabulary tokens.

However, these systems generally still perform:

```text
LLM
 ↓
token
 ↓
token
 ↓
token
 ↓
valid JSON
```

Our design asks a different question:

> If the downstream consumer wants a typed value, why generate the textual serialization at all?

The runtime should ideally return the candidate probability distribution directly.

---

# 4. LMCache and Existing KV-Cache Infrastructure

LMCache is relevant to the future shared-prefill implementation.

Paper:

https://arxiv.org/abs/2510.09665

LMCache treats KV-cache state as reusable inference infrastructure and supports retaining/sharing cached states across requests and inference engines.

This overlaps with the Phase 2 architecture:

```text
                     LONG CONTEXT
                          │
                          ▼
                  MiniCPM5-2B prefill
                          │
                          ▼
                       KV CACHE
                          │
        ┌─────────────────┼─────────────────┐
        ▼                 ▼                 ▼
    question A        question B        question C
      YES/NO          A/B/C/D          LOW/MED/HIGH
```

Therefore we should avoid unnecessarily reinventing production-grade KV-cache infrastructure.

Longer term, investigate compatibility/integration with:

- LMCache
- vLLM
- SGLang

The initial implementation can remain plain PyTorch/Transformers for transparency and ROCm compatibility.

---

# Updated Product Direction

Rather than positioning the project as a "Jev clone", a more useful abstraction is:

> **An open semantic decision runtime for ordinary language models.**

MiniCPM5-2B is simply the initial backend.

The public-facing API should eventually be model-independent.

Example:

```python
from systemone import DecisionModel, Boolean, Choice

model = DecisionModel("openbmb/MiniCPM5-2B")

result = model.decide(
    context,
    fields={
        "anomaly": Boolean(),
        "cause": Choice(
            "source",
            "coding",
            "warehouse",
            "market",
        ),
    },
)

print(result.anomaly.value)
# True

print(result.anomaly.probability)
# 0.981

print(result.cause.value)
# "source"

print(result.cause.probability)
# 0.873
```

The important abstraction is:

```text
unstructured semantic context
             ↓
      DecisionModel
             ↓
typed probability distributions
```

rather than:

```text
context
   ↓
LLM generates JSON
   ↓
parser
   ↓
typed values
```

---

# Proposed Four-Level Architecture

## Level 1 — Dynamic Zero-Shot Decisions

No task-specific training.

```text
context
   ↓
MiniCPM5-2B
   ↓
shared prefill
   ↓
candidate-logit evaluation
   ↓
Boolean / Choice probability distributions
```

This is the initial implementation.

It should support decisions that have never previously been compiled or trained.

---

## Level 2 — Calibrated Decisions

Raw softmax probabilities should **not** be assumed to represent empirical confidence.

Store:

```text
prediction
raw probability
eventual ground truth
```

and evaluate:

- accuracy
- Brier score
- Expected Calibration Error
- reliability curves

Support calibration methods such as:

- temperature scaling
- class-specific temperature scaling
- isotonic regression

The API should distinguish raw and calibrated confidence where appropriate.

Example:

```python
result.cause.raw_probability
result.cause.probability
```

The second value should eventually represent the calibrated estimate.

---

## Level 3 — Compiled Decision Heads

Frequently repeated decisions should optionally be converted into LatentGate-style learned heads.

Potential API:

```python
model.compile(
    field="cause",
    training_examples=examples,
)
```

Conceptually:

```text
DYNAMIC MODE

Choice(...)
    ↓
constrained candidate logits


        model.compile()
              ↓


COMPILED MODE

MiniCPM hidden state
        ↓
pooling
        ↓
PCA whitening
        ↓
tiny classifier
        ↓
P(class)
```

This gives the runtime two different execution strategies.

### Dynamic

Advantages:

- no training
- arbitrary questions
- arbitrary choices
- immediate use

Disadvantages:

- requires question evaluation through the transformer

### Compiled

Advantages:

- extremely low latency
- potentially very high task-specific accuracy
- tiny retraining cost
- easy continual learning from confirmed outcomes

Disadvantages:

- requires labeled examples
- fixed decision semantics

The runtime could eventually select the execution path automatically.

---

# Level 4 — Escalation

Confidence should become an execution-control primitive.

Example architecture:

```text
                     semantic decision runtime
                               │
             ┌─────────────────┼──────────────────┐
             │                 │                  │
        high confidence   medium confidence   low confidence
             │                 │                  │
             ▼                 ▼                  ▼
      automatic action    cheap SLM agent    frontier model
                                                   │
                                                   ▼
                                             deep reasoning
```

Example:

```python
if decision.probability >= 0.995:
    automatic_action()

elif decision.probability >= 0.90:
    lightweight_agent()

else:
    reasoning_agent()
```

Thresholds should only become meaningful once calibration has been empirically validated.

---

# Potential Distinguishing Feature: `compile()`

The most interesting longer-term feature may be transparently converting frequently used semantic questions into trained latent heads.

Example:

```python
cause = Choice(
    "source",
    "coding",
    "warehouse",
    "market",
)
```

Initially:

```text
Choice
  ↓
dynamic constrained-logit inference
```

After sufficient labeled examples:

```python
model.compile("cause", examples)
```

the runtime can create:

```text
MiniCPM hidden representation
             ↓
      PCA whitening
             ↓
       learned probe
             ↓
          P(cause)
```

Application code does not need to change.

Conceptually:

```text
                       Decision API
                            │
              ┌─────────────┴─────────────┐
              │                           │
       uncompiled field              compiled field
              │                           │
              ▼                           ▼
     constrained logits             latent probe
              │                           │
              └─────────────┬─────────────┘
                            ▼
                 calibrated Decision
```

This combination appears substantially less explored than either constrained decoding or semantic routing individually.

---

# Positioning

Avoid claiming that this reproduces:

- TypeSafe Jev
- TypeSafe's architecture
- TypeSafe RLCD

Those details remain proprietary/insufficiently published.

A better description is:

> **A model-agnostic experimental runtime that turns ordinary causal language models into fast, typed, probabilistic semantic decision engines.**

The project combines ideas from:

- constrained inference
- candidate-logit classification
- shared KV-cache inference
- probability calibration
- latent-space classifiers
- semantic routing
- uncertainty-based model escalation

The initial backend is MiniCPM5-2B, but the architecture should avoid MiniCPM-specific assumptions wherever practical.

---

# Research Questions to Preserve

Benchmarks should eventually answer:

1. How much faster is direct candidate-logit evaluation than structured autoregressive generation?
2. How does latency scale as the number of simultaneous decision fields increases?
3. How much does shared context prefill/KV reuse improve multi-field inference?
4. How well calibrated are raw MiniCPM5-2B candidate probabilities?
5. How much can simple post-hoc calibration improve ECE/Brier score?
6. Which MiniCPM5 layer provides the best representations for LatentGate-style probes?
7. How do dynamic constrained-logit decisions compare with compiled latent heads?
8. At what number of repeated examples does compiling a decision become worthwhile?
9. Can compiled heads retain useful OOD performance?
10. How do MiniCPM5-2B Base, SFT and post-trained checkpoints differ in decision accuracy and calibration?
11. How does the runtime compare with Jev on identical decision datasets once Jev API access is available?
12. Can calibrated uncertainty reliably determine when escalation to a larger reasoning model is necessary?

The goal should not merely be lower latency.

The more important question is whether a small frozen causal LM can become a reusable **semantic computation substrate**, with text generation being only one possible interface rather than its mandatory output mechanism.