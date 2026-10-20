"""Public entry point: the :class:`Detector` and the module-level ``detect``."""

from __future__ import annotations

from typing import Optional

from .claims import extract_claims
from .classify import classify_claim
from .config import DEFAULT_CONFIG, DetectorConfig
from .evidence import ContextIndex
from .report import ClaimResult, DetectionReport


class Detector:
    """Compares an AI-generated answer against a reference context.

    Every evidence span the detector reports is verified to be a verbatim,
    offset-exact slice of the context before the report is returned. If the
    context does not contain enough information to decide, the claim is
    classified ``NOT_IN_CONTEXT``. The detector never guesses.
    """

    def __init__(self, config: Optional[DetectorConfig] = None) -> None:
        self.config = config or DEFAULT_CONFIG

    def detect(self, context: str, answer: str) -> DetectionReport:
        """Classify each claim in ``answer`` against ``context``."""
        index = ContextIndex(context)
        results = []
        for claim in extract_claims(answer):
            result = classify_claim(claim, index, self.config)
            self._verify_evidence(context, result)
            results.append(result)
        return DetectionReport(context=context, answer=answer, claims=results)

    @staticmethod
    def _verify_evidence(context: str, result: ClaimResult) -> None:
        """Hard guarantee: reported evidence is verbatim context text."""
        for span in result.evidence:
            if context[span.start : span.end] != span.quote:
                raise RuntimeError(
                    "internal error: evidence span does not match the context"
                )
            if span.quote not in context:
                raise RuntimeError(
                    "internal error: evidence quote is not present in the context"
                )


def detect(context: str, answer: str, config: Optional[DetectorConfig] = None) -> DetectionReport:
    """Convenience wrapper around :class:`Detector`."""
    return Detector(config=config).detect(context, answer)
