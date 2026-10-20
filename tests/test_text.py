"""Tests for the low-level text utilities."""

from hallucination_detector.text import (
    content_lemmas,
    extract_numbers,
    iter_words,
    lemma,
    number_surfaces,
    numbers_compatible,
    split_sentences,
)


def _slices(text):
    return [span.slice(text) for span in split_sentences(text)]


class TestSentenceSplitting:
    def test_basic_split_with_offsets(self):
        text = "The sky is blue. It is bright today!"
        spans = split_sentences(text)
        assert [span.slice(text) for span in spans] == [
            "The sky is blue.",
            "It is bright today!",
        ]
        for span in spans:
            assert text[span.start : span.end] == span.slice(text)

    def test_abbreviations_do_not_split(self):
        text = "Dr. Smith arrived yesterday. She was late."
        sentences = _slices(text)
        assert sentences == ["Dr. Smith arrived yesterday.", "She was late."]

    def test_decimal_numbers_do_not_split(self):
        text = "The margin was 3.5 points. It improved."
        assert _slices(text) == ["The margin was 3.5 points.", "It improved."]

    def test_country_abbreviation(self):
        text = "The U.S. delegation arrived. Talks began."
        assert _slices(text) == ["The U.S. delegation arrived.", "Talks began."]

    def test_offsets_survive_leading_whitespace(self):
        text = "  Hello there. Bye now."
        spans = split_sentences(text)
        assert spans[0].slice(text) == "Hello there."
        assert text[spans[0].start : spans[0].end] == "Hello there."

    def test_single_letter_initial(self):
        text = "J. R. R. Tolkien wrote books. They are long."
        sentences = _slices(text)
        assert sentences[0].startswith("J. R. R. Tolkien")
        assert len(sentences) == 2


class TestTokenization:
    def test_word_offsets_index_back_into_text(self):
        text = "Acme Corp's revenue rose."
        for start, end, token in iter_words(text):
            assert text[start:end] == token


class TestLemmatization:
    def test_irregulars(self):
        assert lemma("sold") == "sell"
        assert lemma("rose") == "rise"
        assert lemma("fell") == "fall"
        assert lemma("was") == "be"

    def test_suffix_forms_converge(self):
        assert lemma("increased") == lemma("increase") == lemma("increasing")
        assert lemma("companies") == lemma("company")
        assert lemma("closed") == lemma("close") == lemma("closing")
        assert lemma("reported") == lemma("reports") == lemma("report")


class TestNumbers:
    def test_digit_and_comma_forms(self):
        assert "3000" in extract_numbers("There were 3,000 attendees.")
        assert "2019" in extract_numbers("It happened in 2019.")

    def test_scaled_numbers_expand(self):
        values = extract_numbers("The company sold 2 million units.")
        assert any(v.startswith("2 million") for v in values)
        assert any("2000000" in v for v in values)

    def test_compatibility(self):
        assert numbers_compatible("2 million|2000000", "2000000")
        assert not numbers_compatible("2 million|2000000", "3 million|3000000")
        # percent signs are normalized before comparison
        assert "50 percent" in extract_numbers("Sales grew 50% in one year.")
        assert numbers_compatible("50 percent", "50 percent")

    def test_spelled_numbers(self):
        assert "2" in extract_numbers("The study had two groups.")
        # multiplier words alone must not become independent figures
        assert "1000000" not in extract_numbers("The company sold 2 million units.")

    def test_number_surfaces_are_verbatim(self):
        assert number_surfaces("Sales reached 3 million and two deals.") == [
            "3 million",
            "two",
        ]


class TestContentTokens:
    def test_stopwords_removed(self):
        pairs = content_lemmas("The cat is on the mat.")
        surfaces = [word.lower() for word, _ in pairs]
        assert "the" not in surfaces
        assert "is" not in surfaces
        assert "cat" in surfaces
        assert "mat" in surfaces
