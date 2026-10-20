"""Split an AI-generated answer into atomic claims with exact offsets."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Set

from .text import lemma

_AUXILIARIES = frozenset(
    {
        "is", "are", "was", "were", "be", "been", "being", "has", "have", "had",
        "do", "does", "did", "will", "would", "can", "could", "may", "might",
        "must", "shall", "should",
    }
)

# A pragmatic lexicon of common main verbs in their frequent surface forms.
# Used only to decide whether both sides of a split point contain a predicate.
_COMMON_VERBS = frozenset(
    {
        "say", "tell", "state", "claim", "report", "announce", "note", "write",
        "describe", "argue", "suggest", "recommend", "confirm", "deny",
        "refuse", "add", "explain", "mention", "declare", "assert", "conclude",
        "estimate", "measure", "record", "find", "show", "prove", "test",
        "study", "analyze", "compare", "include", "contain", "feature",
        "require", "allow", "enable", "prevent", "cause", "help", "lead",
        "make", "create", "design", "develop", "build", "produce", "generate",
        "release", "launch", "open", "close", "sign", "acquire", "sell", "buy",
        "hire", "elect", "appoint", "award", "name", "become", "begin", "start",
        "end", "finish", "stop", "continue", "happen", "occur", "live", "die",
        "rise", "fall", "grow", "shrink", "increase", "decrease", "improve",
        "worsen", "expand", "cut", "add", "lose", "win", "fail", "succeed",
        "work", "use", "run", "move", "reach", "hit", "hold", "keep", "set",
        "put", "give", "take", "bring", "send", "pay", "cost", "hire",
        "approve", "reject", "introduce", "pass", "vote", "meet", "talk",
        "say", "call", "order", "ask", "answer", "expect", "predict", "plan",
        "intend", "try", "want", "need", "like", "prefer", "choose", " decide",
    }
)

_COMMON_VERBS = frozenset(v.strip() for v in _COMMON_VERBS)

# Coordinating split candidates. Semicolons always split; the conjunction and
# comma forms only split when *both* resulting fragments contain a predicate.
_SPLIT_RE = re.compile(
    r"(?:;)"
    r"|(?:,\s+(?:and|but|while|whereas)\b)"
    r"|(?:(?<=\s)(?:and|but|or|while|whereas|so)(?=\s))"
)

_MIN_CLAIM_CHARS = 4


@dataclass(frozen=True)
class Claim:
    """One checkable assertion extracted from the answer."""

    text: str
    start: int
    end: int


def _has_predicate(fragment: str) -> bool:
    """True if the fragment plausibly contains a verb."""
    words = re.findall(r"[A-Za-z0-9']+", fragment.lower())
    if not words:
        return False
    for index, word in enumerate(words):
        if word in _AUXILIARIES or word in _COMMON_VERBS:
            return True
        if index > 0 and words[index - 1] == "to" and len(word) > 2:
            return True
        if len(word) > 4 and (word.endswith("ed") or word.endswith("ing")):
            return True
        if lemma(word) in _COMMON_VERBS:
            return True
    return False


def extract_claims(answer: str) -> List[Claim]:
    """Extract atomic claims from ``answer``.

    Strategy: split the answer into sentences, then split sentences further at
    semicolons and at coordinating conjunctions, but only when both sides of
    a conjunction contain a predicate (so "bread and butter" stays intact and
    "released a phone and raised guidance" becomes two claims). Every claim
    keeps character offsets into the original ``answer`` string.
    """
    claims: List[Claim] = []

    # Local import avoids a circular dependency at module load time.
    from .text import split_sentences

    for sentence in split_sentences(answer):
        segments = [(sentence.start, sentence.end)]
        pieces: List[tuple] = []
        cursor = sentence.start
        for match in _SPLIT_RE.finditer(answer[sentence.start : sentence.end]):
            absolute = sentence.start + match.start()
            if match.group().strip() == ";":
                pieces.append((cursor, absolute))
                cursor = sentence.start + match.end()
                continue
            left = answer[cursor:absolute]
            right_end = sentence.start + match.end()
            right_probe_end = min(sentence.end, right_end + 60)
            right = answer[right_end:right_probe_end]
            if _has_predicate(left) and _has_predicate(right):
                pieces.append((cursor, absolute))
                cursor = right_end
        pieces.append((cursor, sentence.end))
        segments = pieces

        for start, end in segments:
            raw = answer[start:end]
            stripped = raw.strip()
            if len(stripped) < _MIN_CLAIM_CHARS:
                continue
            lead = len(raw) - len(raw.lstrip())
            claim_start = start + lead
            claim_end = claim_start + len(stripped.rstrip())
            text = answer[claim_start:claim_end]
            text = text.strip(" ,;")
            if len(text) < _MIN_CLAIM_CHARS:
                continue
            claims.append(Claim(text=text, start=claim_start, end=claim_end))
    return claims
