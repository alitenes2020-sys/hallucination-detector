"""Context indexing and evidence retrieval.

The detector only ever reports evidence that is a verbatim slice of the
context, identified by exact character offsets. Retrieval is lexical
(IDF-weighted term coverage), so it can never fabricate a supporting span.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Set

from .signals import negation_cues, restrictive_cues, universal_cues
from .text import Span, content_lemmas, extract_numbers, iter_words, split_sentences

# Multiplier applied to claim terms that are absent from the context entirely.
UNSEEN_PENALTY = 1.5


@dataclass
class SentenceEvidence:
    """Precomputed features for one context sentence."""

    span: Span
    lemmas: Set[str]
    surface_by_lemma: Dict[str, List[str]]
    words: List[str]
    numbers: List[str]
    negation: List[str]
    universal: List[str]
    restrictive: List[str]

    def text(self, context: str) -> str:
        return self.span.slice(context)


@dataclass
class RetrievedEvidence:
    """Best-matching context sentences for a claim."""

    sentences: List[SentenceEvidence]
    coverage: float  # fraction of claim term weight covered by the union
    overlap_lemmas: Set[str]


class ContextIndex:
    """Lexical index over the reference context."""

    def __init__(self, context: str) -> None:
        self.context = context
        self.sentences: List[SentenceEvidence] = []

        document_frequency: Dict[str, int] = {}
        for span in split_sentences(context):
            sentence_text = span.slice(context)
            pairs = content_lemmas(sentence_text)
            lemmas = {base for _, base in pairs}
            surface: Dict[str, List[str]] = {}
            for word, base in pairs:
                surface.setdefault(base, []).append(word)
            words = [word for _, _, word in iter_words(sentence_text)]
            self.sentences.append(
                SentenceEvidence(
                    span=span,
                    lemmas=lemmas,
                    surface_by_lemma=surface,
                    words=words,
                    numbers=extract_numbers(sentence_text),
                    negation=negation_cues(words),
                    universal=universal_cues(words),
                    restrictive=restrictive_cues(words),
                )
            )
            for base in lemmas:
                document_frequency[base] = document_frequency.get(base, 0) + 1

        count = len(self.sentences)
        self.idf: Dict[str, float] = {}
        for base, df in document_frequency.items():
            self.idf[base] = math.log(1.0 + count / (1.0 + df))
        # A term unseen in the context still has a well-defined weight. It is
        # deliberately weighted higher than a seen term: a claim that mentions
        # something the context never does is weaker than one that does.
        self.unseen_idf = (math.log(1.0 + count) if count else 0.0) * UNSEEN_PENALTY

    # ------------------------------------------------------------------
    def _weight(self, lemma_key: str) -> float:
        return self.idf.get(lemma_key, self.unseen_idf)

    def _sentence_coverage(self, claim_terms: Set[str], sentence: SentenceEvidence) -> float:
        total = sum(self._weight(t) for t in claim_terms)
        if total <= 0:
            return 0.0
        matched = sum(self._weight(t) for t in claim_terms if t in sentence.lemmas)
        return matched / total

    def retrieve(
        self,
        claim_terms: Set[str],
        max_sentences: int = 3,
        min_new_coverage: float = 0.15,
        min_evidence_coverage: float = 0.15,
    ) -> RetrievedEvidence:
        """Select the context sentences that best cover the claim's terms.

        If even the best-matching sentence covers less than
        ``min_evidence_coverage`` of the claim, no evidence is attached: a
        barely-related sentence must not masquerade as supporting evidence.
        """
        if not claim_terms or not self.sentences:
            return RetrievedEvidence(sentences=[], coverage=0.0, overlap_lemmas=set())

        scored = [
            (self._sentence_coverage(claim_terms, sentence), sentence)
            for sentence in self.sentences
        ]
        scored = [(score, sent) for score, sent in scored if score > 0]
        scored.sort(key=lambda item: (-item[0], item[1].span.start))

        if not scored:
            return RetrievedEvidence(sentences=[], coverage=0.0, overlap_lemmas=set())
        if scored[0][0] < min_evidence_coverage:
            return RetrievedEvidence(
                sentences=[], coverage=scored[0][0], overlap_lemmas=set()
            )

        chosen: List[SentenceEvidence] = []
        covered: Set[str] = set()

        def union_coverage(terms: Set[str], lemmas: Set[str]) -> float:
            total = sum(self._weight(t) for t in terms)
            if total <= 0:
                return 0.0
            return sum(self._weight(t) for t in terms if t in lemmas) / total

        for score, sentence in scored:
            if not chosen:
                chosen.append(sentence)
                covered |= sentence.lemmas
                continue
            new_terms = {t for t in claim_terms if t not in covered}
            if not new_terms:
                break
            added = union_coverage(new_terms, sentence.lemmas)
            if added >= min_new_coverage and len(chosen) < max_sentences:
                chosen.append(sentence)
                covered |= sentence.lemmas

        return RetrievedEvidence(
            sentences=chosen,
            coverage=union_coverage(claim_terms, covered),
            overlap_lemmas={t for t in claim_terms if t in covered},
        )
