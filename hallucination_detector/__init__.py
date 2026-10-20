"""Hallucination detector: verify AI answers against a reference context.

Classifies each claim in an answer as SUPPORTED, CONTRADICTED, or
NOT_IN_CONTEXT, with a confidence score and verbatim evidence spans from the
context. Pure standard library at runtime.
"""

from __future__ import annotations

from .claims import Claim, extract_claims
from .config import DEFAULT_CONFIG, DetectorConfig
from .detector import Detector, detect
from .report import ClaimResult, DetectionReport, EvidenceSpan, Label

__all__ = [
    "Claim",
    "ClaimResult",
    "DetectionReport",
    "Detector",
    "DetectorConfig",
    "DEFAULT_CONFIG",
    "EvidenceSpan",
    "Label",
    "detect",
    "extract_claims",
]

__version__ = "1.0.0"
