"""Tests for the CLI (``python -m hallucination_detector``)."""

import json
import subprocess
import sys

from hallucination_detector.__main__ import main

CONTEXT = "The company sold 2 million units last year."
ANSWER = "The company sold 3 million units last year."


def test_cli_json_output(capsys):
    exit_code = main(["-c", CONTEXT, "-a", ANSWER, "--json"])
    payload = json.loads(capsys.readouterr().out)
    assert exit_code == 0
    assert payload["claims"][0]["label"] == "CONTRADICTED"


def test_cli_text_output(capsys):
    exit_code = main(["-c", CONTEXT, "-a", ANSWER])
    out = capsys.readouterr().out
    assert exit_code == 0
    assert "CONTRADICTED" in out
    assert "POTENTIAL_HALLUCINATION" in out


def test_cli_strict_exit_code(capsys):
    exit_code = main(["-c", CONTEXT, "-a", ANSWER, "--strict"])
    assert exit_code == 1


def test_cli_strict_passes_on_faithful_answer(capsys):
    faithful = "The company sold 2 million units last year."
    exit_code = main(["-c", CONTEXT, "-a", faithful, "--strict"])
    assert exit_code == 0


def test_cli_reads_files(tmp_path, capsys):
    context_file = tmp_path / "context.txt"
    answer_file = tmp_path / "answer.txt"
    context_file.write_text(CONTEXT, encoding="utf-8")
    answer_file.write_text(ANSWER, encoding="utf-8")
    exit_code = main(
        ["--context-file", str(context_file), "--answer-file", str(answer_file), "--json"]
    )
    payload = json.loads(capsys.readouterr().out)
    assert exit_code == 0
    assert payload["claims"][0]["label"] == "CONTRADICTED"


def test_cli_requires_inputs():
    try:
        main([])
    except SystemExit as exc:
        assert exc.code == 2
    else:  # pragma: no cover
        raise AssertionError("expected SystemExit")


def test_module_invocation_as_subprocess():
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "hallucination_detector",
            "-c",
            CONTEXT,
            "-a",
            ANSWER,
            "--json",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    payload = json.loads(completed.stdout)
    assert payload["claims"][0]["label"] == "CONTRADICTED"
