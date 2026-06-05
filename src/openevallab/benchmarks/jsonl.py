"""JSONL benchmark schema and loader."""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Any, Iterable

REQUIRED_FIELDS = {"id", "task_type", "prompt", "gold_answer", "metadata"}


@dataclass(frozen=True)
class BenchmarkExample:
    """A single benchmark item loaded from JSONL."""

    id: str
    task_type: str
    prompt: str
    gold_answer: str
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_record(cls, record: dict[str, Any], *, line_number: int | None = None) -> "BenchmarkExample":
        missing = REQUIRED_FIELDS.difference(record)
        location = f" on line {line_number}" if line_number is not None else ""
        if missing:
            raise ValueError(f"Missing required field(s){location}: {', '.join(sorted(missing))}")
        if not isinstance(record["metadata"], dict):
            raise ValueError(f"Field 'metadata' must be an object{location}")
        return cls(
            id=str(record["id"]),
            task_type=str(record["task_type"]),
            prompt=str(record["prompt"]),
            gold_answer=str(record["gold_answer"]),
            metadata=record["metadata"],
        )


def load_jsonl_benchmark(path: str | Path) -> list[BenchmarkExample]:
    """Load benchmark examples from a JSON Lines file."""

    benchmark_path = Path(path)
    examples: list[BenchmarkExample] = []
    with benchmark_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                record = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_number}: {exc.msg}") from exc
            if not isinstance(record, dict):
                raise ValueError(f"Expected object on line {line_number}")
            examples.append(BenchmarkExample.from_record(record, line_number=line_number))
    return examples


def iter_jsonl_records(examples: Iterable[BenchmarkExample]) -> Iterable[dict[str, Any]]:
    """Convert examples back to serializable records."""

    for example in examples:
        yield {
            "id": example.id,
            "task_type": example.task_type,
            "prompt": example.prompt,
            "gold_answer": example.gold_answer,
            "metadata": example.metadata,
        }
