"""Linguistic signals used for contradiction detection.

All scans here operate on the *actual* tokens of the claim and of the evidence
sentence; the reported cues are always verbatim surface forms found in the
text, never generated paraphrases.
"""

from __future__ import annotations

import re
from typing import Dict, FrozenSet, List, Set, Tuple

from .text import STOPWORDS, lemma

# --------------------------------------------------------------------------
# Negation
# --------------------------------------------------------------------------

NEGATION_CUES = frozenset(
    {
        "not", "never", "no", "none", "neither", "nor", "nobody", "nothing",
        "nowhere", "cannot", "without", "lacks", "lack", "denies", "denied",
        "deny", "refused", "refuse", "fails", "fail", "failed", "unlikely",
        "seldom", "rarely", "hardly", "disprove", "disproved", "contradicts",
        "contradicted", "impossible",
    }
)

_CONTRACTION_RE = re.compile(r"\w+n't$")


def negation_cues(words: List[str]) -> List[str]:
    """Return the negation cue surface forms present in ``words`` (lowercased)."""
    found: List[str] = []
    for word in words:
        lower = word.lower()
        if _CONTRACTION_RE.match(lower) or lower in NEGATION_CUES:
            found.append(lower)
    return found


# --------------------------------------------------------------------------
# Antonyms (unambiguous pairs only; lemmatized at build time)
# --------------------------------------------------------------------------

_ANTONYM_PAIRS: List[Tuple[str, str]] = [
    ("increase", "decrease"), ("increase", "decline"),
    ("increase", "reduce"), ("rise", "fall"), ("rise", "decline"),
    ("grow", "shrink"), ("expand", "contract"), ("expand", "shrink"),
    ("improve", "worsen"), ("gain", "lose"), ("gain", "drop"),
    ("raise", "lower"), ("alive", "dead"), ("open", "closed"),
    ("success", "failure"), ("victory", "defeat"), ("approve", "reject"),
    ("legal", "illegal"), ("safe", "dangerous"), ("rich", "poor"),
    ("begin", "end"), ("start", "stop"), ("buy", "sell"),
    ("import", "export"), ("profit", "loss"), ("unite", "divide"),
    ("accept", "refuse"), ("win", "lose"), ("hot", "cold"),
    ("fast", "slow"), ("big", "small"), ("large", "tiny"),
    ("modern", "ancient"), ("higher", "lower"), ("maximum", "minimum"),
    ("positive", "negative"), ("pass", "fail"), ("borrow", "lend"),
    ("borrower", "lender"), ("urban", "rural"),
]

_ANTONYM_PAIRS = [(a, b) for a, b in _ANTONYM_PAIRS if a != b]


def build_antonym_index(pairs: List[Tuple[str, str]]) -> Dict[str, FrozenSet[str]]:
    index: Dict[str, Set[str]] = {}
    for first, second in pairs:
        a, b = lemma(first), lemma(second)
        if a == b:
            continue
        index.setdefault(a, set()).add(b)
        index.setdefault(b, set()).add(a)
    return {key: frozenset(value) for key, value in index.items()}


ANTONYMS: Dict[str, FrozenSet[str]] = build_antonym_index(_ANTONYM_PAIRS)


def antonyms_in(
    claim_lemmas: Set[str], sentence_lemmas: Set[str]
) -> List[Tuple[str, str]]:
    """Return ``(claim_lemma, evidence_lemma)`` antonym clashes."""
    clashes: List[Tuple[str, str]] = []
    for token in sorted(claim_lemmas):
        for other in sorted(sentence_lemmas):
            if other in ANTONYMS.get(token, ()):
                clashes.append((token, other))
    return clashes


# --------------------------------------------------------------------------
# Quantifiers (universal claims vs. restrictive evidence)
# --------------------------------------------------------------------------

UNIVERSAL_CUES = frozenset(
    {"all", "every", "each", "always", "everyone", "everything", "everywhere"}
)

RESTRICTIVE_CUES = frozenset({"few", "hardly", "rarely", "seldom", "minority"})


def universal_cues(words: List[str]) -> List[str]:
    return [w.lower() for w in words if w.lower() in UNIVERSAL_CUES]


def restrictive_cues(words: List[str]) -> List[str]:
    return [w.lower() for w in words if w.lower() in RESTRICTIVE_CUES]


# --------------------------------------------------------------------------
# Named-entity-ish tokens
# --------------------------------------------------------------------------

# Capitalized words that are sentence furniture rather than entity mentions.
_CAPITALIZED_STOPWORDS = STOPWORDS | {
    "monday", "tuesday", "wednesday", "thursday", "friday", "saturday",
    "sunday", "january", "february", "march", "april", "may", "june", "july",
    "august", "september", "october", "november", "december", "new", "old",
    "one", "two", "three", "first", "second", "third", "last", "next",
    "however", "therefore", "meanwhile", "additionally", "finally",
    "overall", "according", "monday's", "tuesday's", "wednesday's",
    "thursday's", "friday's", "saturday's", "sunday's",
}


def entity_tokens(words: List[Tuple[int, int, str]], sentence_start: int) -> List[str]:
    """Return capitalized non-sentence-initial tokens (entity candidates)."""
    entities: List[str] = []
    for start, _, word in words:
        if not word.isalpha() or len(word) < 2:
            continue
        if not word[0].isupper():
            continue
        if start == sentence_start:
            continue  # sentence-initial capitalization is not evidence of a name
        if word.lower() in _CAPITALIZED_STOPWORDS:
            continue
        entities.append(word)
    return entities
