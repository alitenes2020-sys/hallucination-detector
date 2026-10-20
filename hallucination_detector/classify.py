"""Per-claim classification.

Decision cascade (conservative by design):

1. If the claim's key terms overlap a context sentence strongly enough    (``min_contradiction_overlap``) AND a contradiction signal fires
    (negation flip, antonym clash, incompatible figure, universal-vs-restrictive
    quantifier), the claim is CONTRADICTED. Every signal is backed by verbatim
   tokens from the claim and the evidence sentence.
2. Otherwise, if a key entity of the claim is absent, or the claim asserts a
   figure the context never states, the claim is NOT_IN_CONTEXT.
3. Otherwise, if IDF-weighted term coverage reaches ``min_support_overlap``
   with no contradicting signal, the claim is SUPPORTED.    4. Everything else is NOT_IN_CONTEXT. An insufficient context never yields
   SUPPORTED, and no evidence is ever invented: reported spans are exact
   slices of the context.
"""

from __future__ import annotations

from typing import Dict, List, Set, Tuple

from .claims import Claim
from .config import DetectorConfig
from .evidence import ContextIndex, RetrievedEvidence, SentenceEvidence
from .report import ClaimResult, EvidenceSpan, Label
from .signals import antonyms_in, entity_tokens, negation_cues, universal_cues
from .text import (
    content_lemmas,
    extract_numbers,
    iter_words,
    number_shape,
    number_surfaces,
    numbers_compatible,
)

_FLAG_STRENGTH: Dict[str, float] = {
    "negation": 0.90,
    "antonym": 0.85,
    "figure": 0.80,
    "quantifier": 0.75,
}


def _clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, value))


def _to_evidence_spans(index: ContextIndex, retrieved: RetrievedEvidence) -> List[EvidenceSpan]:
    return [
        EvidenceSpan(
            quote=sentence.text(index.context),
            start=sentence.span.start,
            end=sentence.span.end,
        )
        for sentence in retrieved.sentences
    ]


def _contradiction_flags(
    claim: Claim,
    index: ContextIndex,
    retrieved: RetrievedEvidence,
    claim_terms: Set[str],
    claim_surface: Dict[str, List[str]],
    claim_numbers: List[str],
    min_overlap: float,
) -> List[Tuple[str, str]]:
    """Return ``(kind, reason)`` for each distinct contradiction signal."""
    if not retrieved.sentences:
        return []
    flags: List[Tuple[str, str]] = []
    seen_kinds: Set[str] = set()

    claim_words = [word for _, _, word in iter_words(claim.text)]

    # 1) negation flip: exactly one side negated, same topic
    claim_neg = negation_cues(claim_words)
    if claim_neg or any(sentence.negation for sentence in retrieved.sentences):
        for sentence in retrieved.sentences:
            if bool(claim_neg) != bool(sentence.negation):
                cue = claim_neg[0] if claim_neg else sentence.negation[0]
                if claim_neg:
                    reason = (
                        f"the claim negates the statement (cue '{cue}') while the "
                        f"evidence states it without negation: "
                        f"\"{sentence.text(index.context)}\""
                    )
                else:
                    reason = (
                        f"the evidence negates the statement (cue '{cue}'): "
                        f"\"{sentence.text(index.context)}\" while the claim asserts it"
                    )
                flags.append(("negation", reason))
                break

    # 2) antonym clash
    for sentence in retrieved.sentences:
        clashes = antonyms_in(claim_terms, sentence.lemmas)
        if clashes:
            claim_lemma, evidence_lemma = clashes[0]
            surface_claim = claim_surface.get(claim_lemma, [claim_lemma])[0]
            surface_evidence = sentence.surface_by_lemma.get(
                evidence_lemma, [evidence_lemma]
            )[0]
            reason = (
                f"the claim uses '{surface_claim}' but the evidence uses the "
                f"opposing term '{surface_evidence}': "
                f"\"{sentence.text(index.context)}\""
            )
            flags.append(("antonym", reason))
            break

    # 3) figure mismatch: the claim states a figure that conflicts with a
    #    figure of the same unit (e.g. '20 billion' vs '10 billion')
    if claim_numbers:
        evidence_numbers: List[str] = []
        evidence_surfaces: List[str] = []
        for sentence in retrieved.sentences:
            evidence_numbers.extend(sentence.numbers)
            evidence_surfaces.extend(number_surfaces(sentence.text(index.context)))
        unmatched = [
            claim_number
            for claim_number in claim_numbers
            if not any(
                numbers_compatible(claim_number, evidence_number)
                for evidence_number in evidence_numbers
            )
        ]
        if unmatched:
            shapes = {number_shape(value) for value in evidence_numbers}
            conflicting = [
                value for value in unmatched if number_shape(value) in shapes
            ]
            if conflicting:
                index_of = {
                    value: pos
                    for pos, value in enumerate(extract_numbers(claim.text))
                }
                claim_surfaces = number_surfaces(claim.text)
                first = conflicting[0]
                stated_claim = (
                    claim_surfaces[index_of[first]]
                    if first in index_of and index_of[first] < len(claim_surfaces)
                    else first
                )
                shape = number_shape(first)
                evidence_index = next(
                    (
                        pos
                        for pos, value in enumerate(evidence_numbers)
                        if number_shape(value) == shape
                    ),
                    0,
                )
                stated_evidence = (
                    evidence_surfaces[evidence_index]
                    if evidence_index < len(evidence_surfaces)
                    else evidence_numbers[evidence_index]
                )
                reason = (
                    f"the claim states the figure '{stated_claim}' while the "
                    f"evidence states '{stated_evidence}'"
                )
                flags.append(("figure", reason))

    # 4) universal generalization vs. restrictive evidence
    claim_universal = universal_cues(claim_words)
    if claim_universal:
        for sentence in retrieved.sentences:
            if sentence.restrictive:
                reason = (
                    f"the claim generalizes with '{claim_universal[0]}' but the "
                    f"evidence restricts it ('{sentence.restrictive[0]}'): "
                    f"\"{sentence.text(index.context)}\""
                )
                flags.append(("quantifier", reason))
                break

    deduped: List[Tuple[str, str]] = []
    for kind, reason in flags:
        if kind not in seen_kinds:
            seen_kinds.add(kind)
            deduped.append((kind, reason))
    return deduped


def classify_claim(claim: Claim, index: ContextIndex, config: DetectorConfig) -> ClaimResult:
    """Classify a single claim against the indexed context."""
    words = list(iter_words(claim.text))
    pairs = content_lemmas(claim.text)
    claim_terms: Set[str] = {base for _, base in pairs}
    claim_surface: Dict[str, List[str]] = {}
    for word, base in pairs:
        claim_surface.setdefault(base, []).append(word)
    claim_numbers = extract_numbers(claim.text)
    # A claim's first word is often its subject ("Globex reported ..."), so
    # sentence-initial capitalization must not disqualify it as an entity.
    claim_entities = entity_tokens(words, sentence_start=-1)

    # A claim with no checkable content ("It is true.") can never be verified.
    if not claim_terms:
        return ClaimResult(
            claim=claim.text,
            start=claim.start,
            end=claim.end,
            label=Label.NOT_IN_CONTEXT,
            confidence=0.9,
            evidence=[],
            reasons=["the claim contains no verifiable content words"],
        )

    retrieved = index.retrieve(
        claim_terms,
        max_sentences=config.max_evidence_sentences,
        min_new_coverage=config.min_new_sentence_coverage,
        min_evidence_coverage=config.min_evidence_coverage,
    )
    coverage = retrieved.coverage
    evidence_spans = _to_evidence_spans(index, retrieved)

    # --- 1. contradiction signals -----------------------------------------
    flags = _contradiction_flags(
        claim,
        index,
        retrieved,
        claim_terms,
        claim_surface,
        claim_numbers,
        config.min_contradiction_overlap,
    )
    if flags:
        strength = max(_FLAG_STRENGTH[kind] for kind, _ in flags)
        strength = _clamp(strength + 0.03 * (len(flags) - 1), 0.0, 0.97)
        confidence = _clamp(0.40 * coverage + 0.60 * strength, 0.50, 0.95)
        reasons = [reason for _, reason in flags]
        reasons.append(
            f"the claim's key terms overlap the context evidence by {coverage:.0%}"
        )
        return ClaimResult(
            claim=claim.text,
            start=claim.start,
            end=claim.end,
            label=Label.CONTRADICTED,
            confidence=round(confidence, 3),
            evidence=evidence_spans,
            reasons=reasons,
        )

    # --- 2. missing entities / figures block SUPPORTED ---------------------
    evidence_text_lower = " ".join(
        sentence.text(index.context).lower() for sentence in retrieved.sentences
    )
    context_lower = index.context.lower()
    missing_entities = [
        entity for entity in claim_entities if entity.lower() not in evidence_text_lower
    ]
    unmatched_numbers = [
        claim_number
        for claim_number in claim_numbers
        if not any(
            numbers_compatible(claim_number, evidence_number)
            for sentence in retrieved.sentences
            for evidence_number in sentence.numbers
        )
    ]

    numbers_ok = not claim_numbers or not unmatched_numbers
    entities_ok = not missing_entities

    # --- 3. supported? -----------------------------------------------------
    if coverage >= config.min_support_overlap and numbers_ok and entities_ok:
        evidence_terms: Set[str] = set()
        for sentence in retrieved.sentences:
            evidence_terms |= sentence.lemmas
        total_evidence_weight = sum(index.idf.get(t, index.unseen_idf) for t in evidence_terms)
        overlap_weight = sum(
            index.idf.get(t, index.unseen_idf) for t in retrieved.overlap_lemmas
        )
        precision = overlap_weight / total_evidence_weight if total_evidence_weight else 0.0
        confidence = _clamp(0.55 * coverage + 0.25 * precision + 0.20, 0.50, 0.95)

        key_terms = sorted(retrieved.overlap_lemmas, key=lambda t: -index.idf.get(t, 0))[:4]
        surfaces = ", ".join(
            f"'{claim_surface[t][0]}'" for t in key_terms if claim_surface.get(t)
        )
        reasons = [
            f"the claim's key terms {surfaces} are attested in the context evidence"
            if surfaces
            else "the claim's key terms are attested in the context evidence",
            "no contradicting signal (negation, opposing term, or figure "
            "mismatch) was found in the evidence",
        ]
        return ClaimResult(
            claim=claim.text,
            start=claim.start,
            end=claim.end,
            label=Label.SUPPORTED,
            confidence=round(confidence, 3),
            evidence=evidence_spans,
            reasons=reasons,
        )

    # --- 4. otherwise: insufficient context --------------------------------
    reasons: List[str] = []
    confidence = 0.35 + 0.45 * (1.0 - coverage)
    missing_anywhere = [
        entity for entity in claim_entities if entity.lower() not in context_lower
    ]
    if missing_anywhere:
        reasons.append(
            f"the entity '{missing_anywhere[0]}' is not mentioned anywhere in the context"
        )
        confidence += 0.15
    elif missing_entities:
        reasons.append(
            f"no supporting evidence for this claim contains the entity "
            f"'{missing_entities[0]}'"
        )
        confidence += 0.05
    if claim_numbers and unmatched_numbers:
        surfaces = number_surfaces(claim.text)
        stated = (
            surfaces[claim_numbers.index(unmatched_numbers[0])]
            if unmatched_numbers[0] in claim_numbers and surfaces
            else unmatched_numbers[0]
        )
        reasons.append(f"the context does not state the figure '{stated}'")
        confidence += 0.05
    if coverage >= 0.95:
        reasons.append(
            "the claim's key terms are attested in the context, but a detail "
            "in the claim is not verified by the context"
        )
    elif coverage > 0:
        reasons.append(
            f"only {coverage:.0%} of the claim's key terms match any part of the context"
        )
    else:
        reasons.append("no part of the claim matches the context")

    return ClaimResult(
        claim=claim.text,
        start=claim.start,
        end=claim.end,
        label=Label.NOT_IN_CONTEXT,
        confidence=round(_clamp(confidence, 0.35, 0.92), 3),
        evidence=evidence_spans,
        reasons=reasons,
    )
