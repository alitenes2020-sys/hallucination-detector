"""Tunable thresholds for the detector."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DetectorConfig:
    """Thresholds governing classification.

    Attributes:
        min_support_overlap: minimum IDF-weighted coverage of a claim's key
            terms in the retrieved evidence required to call it SUPPORTED.
        min_contradiction_overlap: minimum coverage required before any
            contradiction signal (negation, antonym, figure mismatch,
            quantifier clash) is considered meaningful.
        max_evidence_sentences: maximum context sentences reported per claim.
        min_new_sentence_coverage: how much new term coverage a second (or
            third) evidence sentence must add to be included.
        min_evidence_coverage: a sentence below this coverage is never
            attached as evidence for a claim (it is barely related).
    """

    min_support_overlap: float = 0.70
    min_contradiction_overlap: float = 0.45
    max_evidence_sentences: int = 3
    min_new_sentence_coverage: float = 0.15
    min_evidence_coverage: float = 0.15


DEFAULT_CONFIG = DetectorConfig()
