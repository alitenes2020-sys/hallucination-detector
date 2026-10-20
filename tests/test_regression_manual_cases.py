import subprocess
import sys


def classify(context, answer):
    result = subprocess.run(
        [
            sys.executable, "-m", "hallucination_detector",
            "-c", context, "-a", answer,
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout


def test_supported_claim():
    output = classify(
        "The company increased its revenue.",
        "The company increased its revenue.",
    )
    assert "SUPPORTED" in output
    assert "FAITHFUL" in output


def test_contradicted_number():
    output = classify(
        "Revenue was 15 million.",
        "Revenue was 20 million.",
    )
    assert "CONTRADICTED" in output


def test_claim_not_in_context():
    output = classify(
        "Revenue was 15 million.",
        "The company has 200 employees.",
    )
    assert "NOT_IN_CONTEXT" in output
