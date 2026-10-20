"""Runnable demo for the hallucination detector.

Usage:
    python examples/run_demo.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from hallucination_detector import detect  # noqa: E402

CONTEXT = (
    "Novo Nordisk reported operating profit of 10 billion dollars in 2023. "
    "The company sells Ozempic, a diabetes drug. "
    "Revenue grew by 31 percent in 2023. "
    "The firm is based in Denmark."
)

ANSWER = (
    "Novo Nordisk reported operating profit of 20 billion dollars in 2023. "
    "The company sells Ozempic, a diabetes drug. "
    "Revenue grew by 31 percent in 2023. "
    "The firm is based in Switzerland. "
    "Ozempic is approved for weight loss in children."
)


def main() -> None:
    report = detect(CONTEXT, ANSWER)
    print(report.format_text())
    print()
    print("--- JSON ---")
    print(report.to_json())


if __name__ == "__main__":
    main()
