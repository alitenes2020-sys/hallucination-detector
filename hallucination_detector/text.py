"""Text utilities for the hallucination detector.

Everything here preserves exact character offsets into the original text so
that any evidence the detector reports can be traced back to a verbatim span.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, Iterator, List, Tuple

# --------------------------------------------------------------------------
# Tokenization
# --------------------------------------------------------------------------

# Words (with internal apostrophes/hyphens) or single non-space symbols.
_WORD_RE = re.compile(r"[A-Za-z0-9]+(?:['\-][A-Za-z0-9]+)*|[^\w\s]")

# Sentence terminators (only at a word boundary), incl. trailing quotes/brackets.
_TERMINATOR_RE = re.compile(r"[.!?]+[\"'”’)\]]*(?=\s|$)")

_ABBREVIATIONS = frozenset(
    {
        "mr.", "mrs.", "ms.", "dr.", "prof.", "sr.", "jr.", "st.", "mt.",
        "vs.", "etc.", "e.g.", "i.e.", "fig.", "no.", "ca.", "al.", "cf.",
        "u.s.", "u.k.", "a.m.", "p.m.", "inc.", "ltd.", "co.", "corp.",
        "dept.", "est.", "approx.", "ed.", "vol.", "pp.", "jan.", "feb.",
        "mar.", "apr.", "jun.", "jul.", "aug.", "sep.", "sept.", "oct.",
        "nov.", "dec.",
    }
)


@dataclass(frozen=True)
class Span:
    """A verbatim slice of a text, identified by character offsets."""

    start: int
    end: int

    def slice(self, text: str) -> str:
        return text[self.start : self.end]


def iter_words(text: str, offset: int = 0) -> Iterator[Tuple[int, int, str]]:
    """Yield ``(start, end, token)`` for every token in ``text``.

    ``offset`` shifts the reported start/end positions (used when tokenizing a
    slice of a larger document).
    """
    for match in _WORD_RE.finditer(text):
        yield offset + match.start(), offset + match.end(), match.group()


def split_sentences(text: str) -> List[Span]:
    """Split ``text`` into sentences, keeping exact character offsets.

    Handles common abbreviations (``Dr.``, ``e.g.``, ``U.S.``), decimal numbers
    and single-letter initials so they are not treated as sentence boundaries.
    """
    spans: List[Span] = []
    segment_start = 0

    for match in _TERMINATOR_RE.finditer(text):
        # Walk back to the token immediately preceding the punctuation,
        # keeping the punctuation itself so "Dr." / "U.S." are recognized.
        idx = match.start() - 1
        while idx >= segment_start and not text[idx].isspace():
            idx -= 1
        token = text[idx + 1 : match.end()].rstrip('\",)\'’”')

        if token.lower() in _ABBREVIATIONS:
            continue
        bare = token.rstrip(".")
        if len(bare) == 1 and bare.isalpha() and bare.isupper():
            continue  # single-letter initial, e.g. "J. R. R. Tolkien"

        segment_end = match.end()
        raw = text[segment_start:segment_end]
        stripped = raw.strip()
        if stripped:
            lead = len(raw) - len(raw.lstrip())
            spans.append(
                Span(segment_start + lead, segment_start + lead + len(stripped))
            )
        segment_start = segment_end

    tail = text[segment_start:]
    stripped = tail.strip()
    if stripped:
        lead = len(tail) - len(tail.lstrip())
        spans.append(Span(segment_start + lead, segment_start + lead + len(stripped)))
    return spans


# --------------------------------------------------------------------------
# Lemmatization (small, deterministic, heuristic)
# --------------------------------------------------------------------------

IRREGULAR_LEMMAS: Dict[str, str] = {
    "is": "be", "are": "be", "was": "be", "were": "be", "been": "be", "being": "be",
    "has": "have", "had": "have", "does": "do", "did": "do", "done": "do", "doing": "do",
    "men": "man", "women": "woman", "children": "child", "mice": "mouse", "feet": "foot",
    "teeth": "tooth", "geese": "goose", "people": "people",
    "went": "go", "gone": "go", "came": "come", "said": "say", "says": "say",
    "made": "make", "took": "take", "taken": "take", "taught": "teach",
    "thought": "think", "caught": "catch", "bought": "buy", "sought": "seek",
    "fought": "fight", "brought": "bring", "began": "begin", "begun": "begin",
    "grew": "grow", "grown": "grow", "knew": "know", "known": "know",
    "saw": "see", "seen": "see", "wrote": "write", "written": "write",
    "ran": "run", "led": "lead", "met": "meet", "built": "build", "sold": "sell",
    "won": "win", "lost": "lose", "rose": "rise", "risen": "rise",
    "fell": "fall", "fallen": "fall", "shrank": "shrink", "shrunk": "shrink",
    "gave": "give", "given": "give", "told": "tell", "held": "hold", "kept": "keep",
    "felt": "feel", "left": "leave", "read": "read", "struck": "strike",
}


def lemma(token: str) -> str:
    """Reduce a word to a crude base form used for matching.

    Deterministic suffix rules plus a small irregular map. This is not a real
    morphological analyzer; it only needs to map surface forms that should
    match each other onto the same string (``sold``/``sells`` -> ``sell``).
    """
    word = token.lower()
    if word in IRREGULAR_LEMMAS:
        return IRREGULAR_LEMMAS[word]
    if len(word) <= 3:
        return word
    if word.endswith("ies") and len(word) > 4:
        return word[:-3] + "y"
    if word.endswith(("sses", "shes", "ches", "xes", "zes")):
        return word[:-2]
    if word.endswith("s") and not word.endswith(("ss", "us", "is")):
        word = word[:-1]
    if word.endswith("ing") and len(word) > 5:
        stem = word[:-3]
        if len(stem) >= 3 and stem[-1] == stem[-2] and stem[-1] not in "aeiou":
            stem = stem[:-1]
        return stem
    if word.endswith("ed") and len(word) > 4:
        word = word[:-2]
        if len(word) >= 3 and word[-1] == word[-2] and word[-1] not in "aeiou":
            word = word[:-1]
    if word.endswith("e") and len(word) > 4:
        word = word[:-1]
    return word


# --------------------------------------------------------------------------
# Stopwords (kept out of "content" tokens; signal scans below still see them)
# --------------------------------------------------------------------------

STOPWORDS = frozenset(
    {
        # determiners / articles / pronouns
        "the", "a", "an", "this", "that", "these", "those", "my", "your", "his",
        "her", "its", "our", "their", "some", "any", "each", "every", "all",
        "both", "another", "such", "own", "same",
        # conjunctions / prepositions
        "and", "but", "or", "nor", "so", "yet", "while", "whereas", "although",
        "though", "because", "since", "until", "unless", "if", "then", "than",
        "of", "in", "on", "at", "by", "for", "with", "from", "to", "into",
        "onto", "over", "under", "about", "above", "below", "between", "among",
        "across", "through", "during", "before", "after", "within", "without",
        "against", "toward", "towards", "upon", "per", "via", "as", "is", "are",
        "was", "were", "be", "been", "being", "am",
        # auxiliaries / modal / pronouns-lite
        "has", "have", "had", "do", "does", "did", "will", "would", "can",
        "could", "may", "might", "must", "shall", "should", "ca",
        # generic verbs too weak to count as content on their own
        "it", "he", "she", "we", "you", "they", "them", "him", "us", "me",
        "i", "one", "two", "three", "there", "here",
        # adverbs / negation-as-content (negation is detected separately)
        "not", "no", "never", "very", "really", "actually", "also", "only",
        "just", "even", "well", "much", "many", "more", "most", "least",
        "few", "several", "often", "always", "sometimes", "usually", "rarely",
        "seldom", "hardly", "again", "still", "already", "almost", "quite",
        # misc light words
        "said", "according", "however", "whether", "like", "get", "got", "via",
    }
)


def content_lemmas(text: str) -> List[Tuple[str, str]]:
    """Return ``(surface, lemma)`` for content words (stopwords removed).

    Purely numeric tokens are excluded: figures are compared separately by
    the number-extraction logic, which understands scaling ("2 million").
    """
    out: List[Tuple[str, str]] = []
    for _, _, word in iter_words(text):
        if not any(ch.isalnum() for ch in word):
            continue
        if word.isdigit():
            continue
        base = lemma(word)
        if base in STOPWORDS or word.lower() in STOPWORDS:
            continue
        out.append((word, base))
    return out


# --------------------------------------------------------------------------
# Numbers
# --------------------------------------------------------------------------

_NUMBER_RE = re.compile(
    r"(?<![A-Za-z0-9.])\d[\d,]*(?:\.\d+)?\s?(?:%|per\s?cent|percent|million|billion|trillion|thousand|hundred)?",
)

_SPELLED_NUMBERS = {
    "zero": "0", "one": "1", "two": "2", "three": "3", "four": "4", "five": "5",
    "six": "6", "seven": "7", "eight": "8", "nine": "9", "ten": "10",
    "eleven": "11", "twelve": "12", "thirteen": "13", "fourteen": "14",
    "fifteen": "15", "sixteen": "16", "seventeen": "17", "eighteen": "18",
    "nineteen": "19", "twenty": "20", "thirty": "30", "forty": "40",
    "fifty": "50", "sixty": "60", "seventy": "70", "eighty": "80", "ninety": "90",
    "hundred": "100", "thousand": "1000", "million": "1000000",
    "billion": "1000000000",
}

_MULTIPLIERS = {
    "thousand": 1_000,
    "million": 1_000_000,
    "billion": 1_000_000_000,
    "trillion": 1_000_000_000_000,
    "hundred": 100,
}

# Spelled-out numbers that denote a value on their own. Multiplier words
# ("million" ...) are excluded: after "2" in "2 million" they carry no
# independent value and would otherwise neutralize figure comparisons.
_SPELLED_SCALARS = {
    key: value for key, value in _SPELLED_NUMBERS.items() if key not in _MULTIPLIERS
}


def _normalize_number(raw: str) -> str:
    """Canonicalize a numeric surface form for comparison."""
    value = raw.lower().strip()
    value = value.replace(",", "")
    value = value.replace("%", " percent")
    value = value.replace("per cent", "percent")
    value = re.sub(r"\s+", " ", value).strip()

    # "7 million" -> also record the expanded digit form
    match = re.fullmatch(r"(\d+(?:\.\d+)?)\s?([a-z]+)", value)
    if match and match.group(2) in _MULTIPLIERS:
        expanded = float(match.group(1)) * _MULTIPLIERS[match.group(2)]
        digits = str(int(expanded)) if expanded == int(expanded) else str(expanded)
        return f"{value}|{digits}"
    if value.endswith(".0") and value[:-2].isdigit():
        return value[:-2]
    return value


def number_surfaces(text: str) -> List[str]:
    """Return the verbatim numeric surface forms found in ``text``."""
    out: List[str] = []
    for match in _NUMBER_RE.finditer(text):
        token = match.group().strip()
        if any(ch.isdigit() for ch in token):
            out.append(token)
    for _, _, word in iter_words(text):
        if word.lower().rstrip(".,;:") in _SPELLED_SCALARS:
            out.append(word)
    return out


def extract_numbers(text: str) -> List[str]:
    """Extract normalized numeric mentions (digits and spelled-out) from text.

    Each returned value is a canonical comparison key such as ``"2019"``,
    ``"3 million|3000000"`` or ``"50 percent"``. The surface forms themselves
    are never modified; this is only used for matching.
    """
    values: List[str] = []
    for match in _NUMBER_RE.finditer(text):
        token = match.group().strip()
        if not any(ch.isdigit() for ch in token):
            continue
        values.append(_normalize_number(token))
    for _, _, word in iter_words(text):
        base = word.lower().rstrip(".,;:")
        if base in _SPELLED_SCALARS:
            values.append(_SPELLED_SCALARS[base])
    return values


def numbers_compatible(left: str, right: str) -> bool:
    """True if two normalized number keys denote the same value."""
    if left == right:
        return True
    left_forms = set(left.split("|"))
    right_forms = set(right.split("|"))
    return bool(left_forms & right_forms)


def number_shape(value: str) -> str:
    """Unit class of a normalized number: 'billion', 'percent', 'digit'...

    Two figures of the same shape ('20 billion' vs '10 billion', '1998' vs
    '1995') are plausibly claims about the same quantity, so a mismatch is a
    real contradiction rather than missing context.
    """
    base = value.split("|")[0]
    parts = base.split(" ")
    if len(parts) == 2 and parts[1] in _MULTIPLIERS or base.endswith("percent"):
        return parts[-1]
    if base.isdigit():
        return "digit"
    return base
