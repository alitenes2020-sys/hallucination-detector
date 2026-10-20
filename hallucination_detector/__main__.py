"""Command-line interface.

Examples:
    python -m hallucination_detector --context context.txt --answer answer.txt
    python -m hallucination_detector -c "The sky is blue." -a "The sky is green." --json
"""

from __future__ import annotations

import argparse
import sys
from typing import List, Optional

from .config import DetectorConfig
from .detector import Detector


def _read_file(path: str) -> str:
    with open(path, "r", encoding="utf-8") as handle:
        return handle.read()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="hallucination_detector",
        description=(
            "Classify each claim in an AI-generated answer as SUPPORTED, "
            "CONTRADICTED, or NOT_IN_CONTEXT relative to a reference context."
        ),
    )
    source = parser.add_argument_group("context source")
    source.add_argument("-c", "--context", help="reference context text")
    source.add_argument("--context-file", help="path to a file with the reference context")

    answer = parser.add_argument_group("answer source")
    answer.add_argument("-a", "--answer", help="AI-generated answer text")
    answer.add_argument("--answer-file", help="path to a file with the AI-generated answer")

    parser.add_argument("--json", action="store_true", help="emit the report as JSON")
    parser.add_argument(
        "--min-support-overlap",
        type=float,
        default=DetectorConfig.min_support_overlap,
        help="IDF-weighted term coverage required to call a claim SUPPORTED",
    )
    parser.add_argument(
        "--min-contradiction-overlap",
        type=float,
        default=DetectorConfig.min_contradiction_overlap,
        help="minimum term coverage before a contradiction signal is trusted",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="exit with status 1 if any claim is not SUPPORTED",
    )
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if not args.context and not args.context_file:
        parser.error("provide --context or --context-file")
    if not args.answer and not args.answer_file:
        parser.error("provide --answer or --answer-file")

    context = args.context if args.context is not None else _read_file(args.context_file)
    answer = args.answer if args.answer is not None else _read_file(args.answer_file)

    config = DetectorConfig(
        min_support_overlap=args.min_support_overlap,
        min_contradiction_overlap=args.min_contradiction_overlap,
    )
    report = Detector(config=config).detect(context, answer)

    if args.json:
        print(report.to_json())
    else:
        print(report.format_text())

    if args.strict and not report.is_faithful:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
