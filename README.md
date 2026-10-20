# Hallucination Detector

A dependency-free Python tool that checks an AI-generated answer against a
reference context, claim by claim. Each claim is classified as one of:

| Label | Meaning |
| --- | --- |
| `SUPPORTED` | The context contains enough verbatim evidence for the claim, with no contradicting signal. |
| `CONTRADICTED` | The context actively conflicts with the claim (negation flip, antonym, incompatible figure, or restrictive quantifier). |
| `NOT_IN_CONTEXT` | The context is insufficient to verify the claim. This is the default whenever evidence is missing. |

Every classification comes with a confidence score, a list of **verbatim
evidence spans taken from the context** (with exact character offsets), and
human-readable reasons that quote the actual tokens involved. The detector
never invents evidence. If it cannot find support, it says so with
`NOT_IN_CONTEXT`.

## Install and run

Python 3.9+ (the runtime is standard library only):

```bash
pip install -r requirements.txt   # installs pytest, needed only for the tests
python -m pytest                  # run the test suite
```

## Usage

### CLI

```bash
python -m hallucination_detector \
  -c "The company sold 2 million units last year. Revenue grew by 31 percent." \
  -a "The company sold 3 million units last year. Revenue grew by 31 percent."
```

```
[1] CONTRADICTED (confidence 0.88)
    Claim: "The company sold 3 million units last year."
    Evidence: "The company sold 2 million units last year." (context chars 0-43)
    Why: the claim states the figure '3 million' while the evidence states '2 million'
    Why: the claim's key terms overlap the context evidence by 100%

[2] SUPPORTED (confidence 0.95)
    Claim: "Revenue grew by 31 percent."
    Evidence: "Revenue grew by 31 percent." (context chars 44-71)
    Why: the claim's key terms 'Revenue', 'percent', 'grew' are attested in the context evidence
    Why: no contradicting signal (negation, opposing term, or figure mismatch) was found in the evidence

Verdict: POTENTIAL_HALLUCINATION - 1 claim(s) are contradicted or not supported by the context.
```

Other flags: `--context-file` / `--answer-file` to read from files, `--json`
for a machine-readable report, `--strict` to exit with status 1 when any claim
is not `SUPPORTED`, and `--min-support-overlap` / `--min-contradiction-overlap`
to tune the thresholds.

### Python API

```python
from hallucination_detector import detect

context = "The board approved the merger on Monday."
answer = "The board did not approve the merger."

report = detect(context, answer)
claim = report.claims[0]

print(claim.label)        # Label.CONTRADICTED
print(claim.confidence)   # 0.94
print(claim.evidence[0])  # EvidenceSpan(quote='The board approved the merger on Monday.', start=0, end=37)
print(claim.reasons[0])   # the claim negates the statement (cue 'not') while the evidence states it without negation: "..."

print(report.to_json())   # full JSON report
print(report.format_text())  # human-readable version
print(report.is_faithful)    # False
```

`detect(context, answer, config=DetectorConfig(...))` lets you tune
thresholds, for example `DetectorConfig(min_support_overlap=0.75)` for a
stricter notion of what counts as support.

A runnable example lives in `examples/run_demo.py`:

```bash
python examples/run_demo.py
```

## How it works

1. **Claim extraction** (`claims.py`). The answer is split into sentences, then
   into atomic claims at semicolons and coordinating conjunctions. A
   conjunction only splits when both sides contain a predicate, so "bread and
   butter" stays whole while "released a phone and raised guidance" becomes two
   claims. Every claim keeps offsets into the original answer.

2. **Evidence retrieval** (`evidence.py`). Each context sentence is indexed by
   its lemmatized content words. For a claim, sentences are ranked by
   IDF-weighted term coverage and the best few are selected. Because retrieval
   returns sentence spans with exact character offsets, reported evidence is
   by construction a verbatim slice of the context.

3. **Classification** (`classify.py`), a conservative cascade:
   - **Contradiction signals** fire only when the claim overlaps the evidence
     strongly enough (`min_contradiction_overlap`, default 0.45): a negation
     flip ("approved" vs "did not approve"), an antonym clash
     ("increased" vs "decreased"), a figure conflict between same-unit numbers
     ("20 billion" vs "10 billion"), or a universal generalization against
     restrictive evidence ("all" vs "few"). Any such signal yields
     `CONTRADICTED`, and the reason quotes the exact tokens on both sides.
   - A claim whose **entity is absent from the context**, or which asserts a
     **figure the context never states**, can never be `SUPPORTED`. It becomes
     `NOT_IN_CONTEXT`.
   - Otherwise, coverage at or above `min_support_overlap` (default 0.70) with
     no contradicting signal yields `SUPPORTED`.
   - Everything else is `NOT_IN_CONTEXT`. An insufficient context never
     produces support, and nothing is ever fabricated to fill a gap.

4. **Confidence** is a deterministic function of term coverage, evidence
   precision, and contradiction strength, clamped into sensible bands per
   label. It expresses how strongly the lexical evidence supports the assigned
   label, not a calibrated probability.

5. **Verification** (`detector.py`). Before a report is returned, every
   evidence span is re-checked against the context
   (`context[start:end] == quote`). A mismatch raises instead of reporting
   fabricated evidence.

## Guarantees

- **No invented evidence.** Evidence quotes are exact substrings of the
  context, pinned to character offsets. The test suite asserts this invariant
  over every fixture.
- **Insufficient context means `NOT_IN_CONTEXT`.** Missing support is never
  reported as confirmation, and never as a contradiction.
- **Deterministic.** The same inputs always produce the same report. No
  network calls, no randomness, no model weights.
- **Standard library only** at runtime, so it runs anywhere Python runs.

## Tests

```bash
python -m pytest
```

The suite covers sentence splitting and offsets, claim extraction, all three
labels (including negation, figure, antonym, and quantifier contradictions),
evidence integrity, confidence ranges, JSON serialization, report summaries,
threshold tuning, and the CLI (including process-level invocation). See
`tests/` for the fixture table that drives the parametrized end-to-end tests.

## Limitations

This is a lexical and heuristic detector, not a natural-language-inference
model. Be aware of the following:

- **Paraphrase blindness.** "The firm acquired a rival" will not match context
  that says "The firm bought a competitor" unless the lemmas coincide, so a
  genuinely supported claim may come back `NOT_IN_CONTEXT`. This is the
  deliberate failure direction: when unsure, it withholds support.
- **Negation scope is sentence-level.** A claim and an evidence sentence that
  disagree only about where a negation applies can be flagged as
  `CONTRADICTED` even when both are true in context.
- **Antonymy is a fixed lexicon** of unambiguous pairs. Domain-specific
  oppositions are not covered.
- **Figures compare by unit shape.** "20 billion" is checked against "10
  billion" and years against years, but the tool does not do arithmetic (it
  will not derive "3 million" from "1.5 million twice").
- **Common capitalized nouns** at the start of a claim are treated as entity
  candidates. A claim opening with a word the context never uses is reported
  as `NOT_IN_CONTEXT` with that entity named as the reason.

For high-stakes use, treat the output as a triage layer: route
`CONTRADICTED` and `NOT_IN_CONTEXT` claims to human review or a full NLI
model, and use the evidence spans it hands you to make that review fast.

## Project layout

```
hallucination_detector/
  text.py       sentence splitting, tokenization, lemmatization, numbers
  signals.py    negation cues, antonyms, quantifiers, entity detection
  claims.py     atomic claim extraction with offsets
  evidence.py   context indexing and IDF-weighted evidence retrieval
  classify.py   per-claim classification cascade with confidence
  report.py     Label, EvidenceSpan, ClaimResult, DetectionReport
  detector.py   Detector + detect(), evidence verification
  config.py     DetectorConfig thresholds
  __main__.py   command-line interface
tests/          pytest suite
examples/       runnable demo
```
