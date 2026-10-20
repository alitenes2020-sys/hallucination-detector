"""Result types for the hallucination detector."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List


class Label(str, Enum):
    """Verdict for a single claim."""

    SUPPORTED = "SUPPORTED"
    CONTRADICTED = "CONTRADICTED"
    NOT_IN_CONTEXT = "NOT_IN_CONTEXT"


@dataclass(frozen=True)
class EvidenceSpan:
    """A verbatim quote from the context, pinned to exact character offsets."""

    quote: str
    start: int
    end: int

    def to_dict(self) -> Dict[str, object]:
        return {"quote": self.quote, "start": self.start, "end": self.end}


@dataclass(frozen=True)
class ClaimResult:
    """Classification of one claim."""

    claim: str
    start: int
    end: int
    label: Label
    confidence: float
    evidence: List[EvidenceSpan] = field(default_factory=list)
    reasons: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, object]:
        return {
            "claim": self.claim,
            "start": self.start,
            "end": self.end,
            "label": self.label.value,
            "confidence": self.confidence,
            "evidence": [span.to_dict() for span in self.evidence],
            "reasons": list(self.reasons),
        }


@dataclass(frozen=True)
class DetectionReport:
    """Full report for one (context, answer) pair."""

    context: str
    answer: str
    claims: List[ClaimResult] = field(default_factory=list)

    @property
    def summary(self) -> Dict[str, object]:
        counts = {label.value: 0 for label in Label}
        for claim in self.claims:
            counts[claim.label.value] += 1
        non_supported = counts[Label.CONTRADICTED.value] + counts[Label.NOT_IN_CONTEXT.value]
        if not self.claims:
            verdict = "NO_CLAIMS_DETECTED"
        elif non_supported == 0:
            verdict = "FAITHFUL"
        else:
            verdict = "POTENTIAL_HALLUCINATION"
        mean_confidence = (
            round(sum(c.confidence for c in self.claims) / len(self.claims), 3)
            if self.claims
            else 0.0
        )
        return {
            "verdict": verdict,
            "claim_count": len(self.claims),
            "counts": counts,
            "potential_hallucinations": non_supported,
            "mean_confidence": mean_confidence,
        }

    @property
    def is_faithful(self) -> bool:
        return self.summary["verdict"] == "FAITHFUL"

    def to_dict(self) -> Dict[str, object]:
        return {
            "summary": self.summary,
            "claims": [claim.to_dict() for claim in self.claims],
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)

    def format_text(self) -> str:
        """Human-readable rendering of the report."""
        lines: List[str] = []
        summary = self.summary
        lines.append(
            f"Context: {len(self.context)} chars | Answer: {len(self.answer)} chars"
        )
        lines.append(f"Claims: {summary['claim_count']}")
        lines.append("")
        if not self.claims:
            lines.append("No checkable claims were found in the answer.")
        for index, claim in enumerate(self.claims, start=1):
            lines.append(
                f"[{index}] {claim.label.value} (confidence {claim.confidence:.2f})"
            )
            lines.append(f'    Claim: "{claim.claim}"')
            for span in claim.evidence:
                lines.append(
                    f'    Evidence: "{span.quote}" (context chars {span.start}-{span.end})'
                )
            if not claim.evidence:
                lines.append("    Evidence: <none found in context>")
            for reason in claim.reasons:
                lines.append(f"    Why: {reason}")
            lines.append("")
        verdict = summary["verdict"]
        if verdict == "FAITHFUL":
            lines.append("Verdict: FAITHFUL - every claim is supported by the context.")
        elif verdict == "NO_CLAIMS_DETECTED":
            lines.append("Verdict: NO_CLAIMS_DETECTED.")
        else:
            lines.append(
                "Verdict: POTENTIAL_HALLUCINATION - "
                f"{summary['potential_hallucinations']} claim(s) are contradicted "
                "or not supported by the context."
            )
        return "\n".join(lines)
