"""Tests for claim extraction."""

from hallucination_detector.claims import extract_claims


class TestClaimExtraction:
    def test_split_on_conjunction_with_two_predicates(self):
        answer = "The company released a new phone and raised its guidance."
        claims = extract_claims(answer)
        assert [claim.text for claim in claims] == [
            "The company released a new phone",
            "raised its guidance.",
        ]
        # offsets must index back into the original answer
        for claim in claims:
            assert answer[claim.start : claim.end] == claim.text

    def test_noun_phrase_conjunction_not_split(self):
        answer = "Bread and butter is served daily."
        claims = extract_claims(answer)
        assert [claim.text for claim in claims] == ["Bread and butter is served daily."]

    def test_semicolon_always_splits(self):
        answer = "The plan was approved; details remain unclear."
        claims = extract_claims(answer)
        assert len(claims) == 2

    def test_multiple_sentences_become_multiple_claims(self):
        answer = "The sky is blue. Birds sing in the morning. Water is wet."
        claims = extract_claims(answer)
        assert len(claims) == 3

    def test_comma_but_split(self):
        answer = "Revenue grew, but margins shrank."
        claims = extract_claims(answer)
        assert len(claims) == 2
        assert claims[0].text == "Revenue grew"
        assert claims[1].text == "margins shrank."

    def test_offsets_preserved_across_sentences(self):
        answer = "First claim here. Second claim here."
        claims = extract_claims(answer)
        assert answer[claims[0].start : claims[0].end] == "First claim here."
        assert answer[claims[1].start : claims[1].end] == "Second claim here."

    def test_empty_answer(self):
        assert extract_claims("") == []
