"""End-to-end tests for the detector.

The fixture table below is also used by the evidence-integrity test: every
evidence span the detector reports must be a verbatim, offset-exact slice of
the context, so the detector can never invent evidence.
"""

import json

import pytest

from hallucination_detector import Detector, Label, detect

CONTEXT_NEWS = (
    "The company reported revenue of 2 billion dollars in 2022. "
    "The CEO said the firm would expand into Asia. "
    "The board approved the plan."
)

# (id, context, answer, expected labels per claim)
CASES = [
    (
        "supported",
        "The Eiffel Tower is located in Paris.",
        "The Eiffel Tower is located in Paris.",
        [Label.SUPPORTED],
    ),
    (
        "negation_claim",
        "The board approved the merger on Monday.",
        "The board did not approve the merger.",
        [Label.CONTRADICTED],
    ),
    (
        "negation_context",
        "Mercury is not visible to the naked eye.",
        "Mercury is visible to the naked eye.",
        [Label.CONTRADICTED],
    ),
    (
        "figure_mismatch",
        "The company sold 2 million units last year.",
        "The company sold 3 million units last year.",
        [Label.CONTRADICTED],
    ),
    (
        "figure_mismatch_with_matching_year",
        "Novo Nordisk reported operating profit of 10 billion dollars in 2023.",
        "Novo Nordisk reported operating profit of 20 billion dollars in 2023.",
        [Label.CONTRADICTED],
    ),
    (
        "spelled_figure_mismatch",
        "The study had two groups.",
        "The study had three groups.",
        [Label.CONTRADICTED],
    ),
    (
        "antonym",
        "Revenue increased in the fourth quarter.",
        "Revenue decreased in the fourth quarter.",
        [Label.CONTRADICTED],
    ),
    (
        "quantifier",
        "Few employees work remotely.",
        "All employees work remotely.",
        [Label.CONTRADICTED],
    ),
    (
        "not_in_context",
        "The board approved the merger on Monday.",
        "The merger is valued at 4 billion dollars.",
        [Label.NOT_IN_CONTEXT],
    ),
    (
        "entity_missing",
        "Acme Corp reported earnings on Friday.",
        "Globex Corp reported earnings on Friday.",
        [Label.NOT_IN_CONTEXT],
    ),
    (
        "empty_context",
        "",
        "The sky is blue.",
        [Label.NOT_IN_CONTEXT],
    ),
    (
        "mixed_answer",
        CONTEXT_NEWS,
        (
            "The company reported revenue of 2 billion dollars in 2022. "
            "The CEO said the firm would expand into Europe. "
            "The board did not approve the plan."
        ),
        [Label.SUPPORTED, Label.NOT_IN_CONTEXT, Label.CONTRADICTED],
    ),
]

CASE_IDS = [case[0] for case in CASES]


@pytest.mark.parametrize("context,answer,expected", [c[1:] for c in CASES], ids=CASE_IDS)
def test_labels(context, answer, expected):
    report = detect(context, answer)
    assert [claim.label for claim in report.claims] == expected


@pytest.mark.parametrize("context,answer,expected", [c[1:] for c in CASES], ids=CASE_IDS)
def test_evidence_is_never_invented(context, answer, expected):
    """Invariant: every reported evidence span is an exact slice of the context."""
    report = detect(context, answer)
    for claim in report.claims:
        for span in claim.evidence:
            assert span.quote, "evidence quotes must not be empty"
            assert context[span.start : span.end] == span.quote
            assert span.quote in context


@pytest.mark.parametrize("context,answer,expected", [c[1:] for c in CASES], ids=CASE_IDS)
def test_confidences_are_probabilities(context, answer, expected):
    report = detect(context, answer)
    for claim in report.claims:
        assert 0.0 <= claim.confidence <= 1.0
        assert claim.reasons, "every classification must explain itself"


def test_supported_claim_has_verbatim_evidence_quote():
    context = "The Eiffel Tower is located in Paris."
    report = detect(context, "The Eiffel Tower is located in Paris.")
    claim = report.claims[0]
    assert claim.label is Label.SUPPORTED
    assert claim.evidence[0].quote == context
    assert claim.evidence[0].start == 0
    assert claim.evidence[0].end == len(context)


def test_contradicted_reasons_cite_real_tokens():
    context = "The company sold 2 million units last year."
    answer = "The company sold 3 million units last year."
    claim = detect(context, answer).claims[0]
    joined = " ".join(claim.reasons)
    assert "3 million" in joined  # the claim's figure, verbatim
    assert "2 million" in joined  # the evidence figure, verbatim
    assert claim.label is Label.CONTRADICTED
    assert claim.evidence and claim.evidence[0].quote in context


def test_empty_context_yields_not_in_context_everywhere():
    report = detect("", "The sky is blue. Birds sing loudly.")
    assert len(report.claims) == 2
    for claim in report.claims:
        assert claim.label is Label.NOT_IN_CONTEXT
        assert claim.evidence == []
        assert claim.confidence >= 0.7


def test_detection_is_deterministic():
    context = CONTEXT_NEWS
    answer = (
        "The company reported revenue of 2 billion dollars in 2022. "
        "The board did not approve the plan."
    )
    first = detect(context, answer).to_dict()
    second = detect(context, answer).to_dict()
    assert first == second


def test_summary_counts_and_verdict():
    context = CONTEXT_NEWS
    answer = (
        "The company reported revenue of 2 billion dollars in 2022. "
        "The board did not approve the plan."
    )
    report = detect(context, answer)
    summary = report.summary
    assert summary["claim_count"] == 2
    assert summary["counts"]["SUPPORTED"] == 1
    assert summary["counts"]["CONTRADICTED"] == 1
    assert summary["potential_hallucinations"] == 1
    assert summary["verdict"] == "POTENTIAL_HALLUCINATION"
    assert not report.is_faithful


def test_faithful_answer_verdict():
    context = "The Eiffel Tower is located in Paris."
    report = detect(context, "The Eiffel Tower is located in Paris.")
    assert report.summary["verdict"] == "FAITHFUL"
    assert report.is_faithful


def test_report_serializes_to_json():
    report = detect(CONTEXT_NEWS, "The board did not approve the plan.")
    payload = json.loads(report.to_json())
    assert payload["claims"][0]["label"] == "CONTRADICTED"
    assert set(payload["claims"][0]) == {
        "claim",
        "start",
        "end",
        "label",
        "confidence",
        "evidence",
        "reasons",
    }
    assert "summary" in payload


def test_text_format_rendering():
    context = "The board approved the merger on Monday."
    report = detect(context, "The board did not approve the merger.")
    rendered = report.format_text()
    assert "CONTRADICTED" in rendered
    assert "POTENTIAL_HALLUCINATION" in rendered
    assert '"The board approved the merger on Monday."' in rendered


def test_custom_config_thresholds():
    context = "Revenue increased in the fourth quarter with strong demand."
    answer = "Revenue increased in the fourth quarter with strong demand."
    # "strong demand" is nowhere in a longer context, so the default config
    # rejects the claim as NOT_IN_CONTEXT; lowering the support threshold
    # shows the knob is wired end to end.
    from hallucination_detector import DetectorConfig

    strict = detect(
        "Revenue increased in the fourth quarter. Another sentence about taxes.",
        answer,
    )
    assert strict.claims[0].label is Label.NOT_IN_CONTEXT

    lenient = Detector(config=DetectorConfig(min_support_overlap=0.3)).detect(
        "Revenue increased in the fourth quarter. Another sentence about taxes.",
        answer,
    )
    assert lenient.claims[0].label is Label.SUPPORTED


def test_detector_rejects_nothing_and_handles_multiline_context():
    context = "The board approved the merger on Monday.\n\nA second meeting is planned."
    report = detect(context, "The board approved the merger on Monday.")
    assert report.claims[0].label is Label.SUPPORTED
    span = report.claims[0].evidence[0]
    assert context[span.start : span.end] == span.quote
