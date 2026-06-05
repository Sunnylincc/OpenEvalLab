"""JSONL benchmark schema and loader."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
from typing import Any, Iterable

REQUIRED_FIELDS = {"id", "task_type", "prompt", "gold_answer", "metadata"}
STRING_FIELDS = {"id", "task_type", "prompt", "gold_answer"}


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
        location = f" on line {line_number}" if line_number is not None else ""
        missing = REQUIRED_FIELDS.difference(record)
        if missing:
            raise ValueError(f"Missing required field(s){location}: {', '.join(sorted(missing))}")
        for field_name in STRING_FIELDS:
            if not isinstance(record[field_name], str) or not record[field_name].strip():
                raise ValueError(f"Field '{field_name}' must be a non-empty string{location}")
        if not isinstance(record["metadata"], dict):
            raise ValueError(f"Field 'metadata' must be an object{location}")
        return cls(
            id=record["id"],
            task_type=record["task_type"],
            prompt=record["prompt"],
            gold_answer=record["gold_answer"],
            metadata=record["metadata"],
        )

    def to_record(self) -> dict[str, Any]:
        return asdict(self)


def load_jsonl_benchmark(path: str | Path) -> list[BenchmarkExample]:
    """Load benchmark examples from a JSON Lines file with schema validation."""

    benchmark_path = Path(path)
    examples: list[BenchmarkExample] = []
    try:
        handle = benchmark_path.open("r", encoding="utf-8")
    except FileNotFoundError as exc:
        raise ValueError(f"Benchmark file not found: {benchmark_path}") from exc
    with handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                record = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL format on line {line_number}: {exc.msg}") from exc
            if not isinstance(record, dict):
                raise ValueError(f"Expected a JSON object on line {line_number}")
            examples.append(BenchmarkExample.from_record(record, line_number=line_number))
    return examples


def iter_jsonl_records(examples: Iterable[BenchmarkExample]) -> Iterable[dict[str, Any]]:
    """Convert examples back to serializable records."""

    for example in examples:
        yield example.to_record()
