import json

import pytest

from openevallab.benchmarks import load_jsonl_benchmark


def test_load_jsonl_benchmark(tmp_path):
    path = tmp_path / "bench.jsonl"
    path.write_text(
        json.dumps(
            {
                "id": "1",
                "task_type": "qa",
                "prompt": "Capital of France?",
                "gold_answer": "Paris",
                "metadata": {"domain": "geo"},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    examples = load_jsonl_benchmark(path)
    assert examples[0].id == "1"
    assert examples[0].metadata == {"domain": "geo"}


def test_load_jsonl_requires_schema(tmp_path):
    path = tmp_path / "bad.jsonl"
    path.write_text('{"id":"1"}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="Missing required"):
        load_jsonl_benchmark(path)


def test_load_jsonl_rejects_non_string_prompt(tmp_path):
    path = tmp_path / "bad_type.jsonl"
    path.write_text('{"id":"1","task_type":"qa","prompt":42,"gold_answer":"A","metadata":{}}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="prompt"):
        load_jsonl_benchmark(path)
