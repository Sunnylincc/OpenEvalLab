"""Benchmark evaluation loop and result serialization."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
from typing import Any, Callable

from openevallab.benchmarks import BenchmarkExample
from openevallab.metrics import METRICS, MetricResult, contains_answer
from openevallab.models import BaseModelClient

MetricFn = Callable[[str, str], MetricResult]


@dataclass(frozen=True)
class EvaluationResult:
    """Serializable result for one evaluated example."""

    id: str
    prompt: str
    gold_answer: str
    model_answer: str
    score: float
    metric: str
    passed: bool
    task_type: str
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def prediction(self) -> str:
        """Backward-compatible alias for older callers."""

        return self.model_answer

    @property
    def example(self) -> BenchmarkExample:
        """Backward-compatible view as a benchmark example."""

        return BenchmarkExample(
            id=self.id,
            task_type=self.task_type,
            prompt=self.prompt,
            gold_answer=self.gold_answer,
            metadata=self.metadata,
        )


def evaluate_benchmark(
    examples: list[BenchmarkExample],
    model_client: BaseModelClient,
    *,
    metric: str | MetricFn = "contains_answer",
) -> list[EvaluationResult]:
    """Evaluate examples with a model client and one primary metric."""

    if not examples:
        raise ValueError("Benchmark is empty. Add at least one JSONL record before evaluating.")
    metric_fn: MetricFn
    metric_name: str
    if isinstance(metric, str):
        if metric not in METRICS:
            available = ", ".join(sorted(METRICS))
            raise ValueError(f"Unsupported metric '{metric}'. Available metrics: {available}.")
        metric_fn = METRICS[metric]
        metric_name = metric
    else:
        metric_fn = metric
        metric_name = getattr(metric, "__name__", "custom_metric")

    results: list[EvaluationResult] = []
    for example in examples:
        answer = model_client.generate(example.prompt)
        metric_result = metric_fn(answer, example.gold_answer)
        results.append(
            EvaluationResult(
                id=example.id,
                prompt=example.prompt,
                gold_answer=example.gold_answer,
                model_answer=answer,
                score=metric_result.score,
                metric=metric_result.name or metric_name,
                passed=metric_result.passed,
                task_type=example.task_type,
                metadata=example.metadata,
            )
        )
    return results


def summarize_scores(results: list[EvaluationResult]) -> dict[str, float | int]:
    """Aggregate mean score and pass rate."""

    if not results:
        return {"num_examples": 0, "mean_score": 0.0, "pass_rate": 0.0}
    return {
        "num_examples": len(results),
        "mean_score": sum(result.score for result in results) / len(results),
        "pass_rate": sum(result.passed for result in results) / len(results),
    }


def results_payload(
    *,
    results: list[EvaluationResult],
    model_name: str,
    benchmark_path: str,
) -> dict[str, Any]:
    """Create the JSON payload written by the CLI."""

    return {
        "schema_version": "1.0",
        "model_name": model_name,
        "benchmark_path": benchmark_path,
        "aggregate_metrics": summarize_scores(results),
        "results": [asdict(result) for result in results],
    }


def save_results(payload: dict[str, Any], path: str | Path) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def load_results(path: str | Path) -> dict[str, Any]:
    result_path = Path(path)
    try:
        payload = json.loads(result_path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ValueError(f"Results file not found: {result_path}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"Malformed result file '{result_path}': {exc.msg}") from exc
    if not isinstance(payload, dict) or not isinstance(payload.get("results"), list):
        raise ValueError("Malformed result file: expected an object with a 'results' list.")
    return payload


def result_from_dict(record: dict[str, Any]) -> EvaluationResult:
    required = {"id", "prompt", "gold_answer", "model_answer", "score", "metric", "passed", "task_type", "metadata"}
    missing = required.difference(record)
    if missing:
        raise ValueError(f"Malformed result record; missing: {', '.join(sorted(missing))}")
    return EvaluationResult(
        id=str(record["id"]),
        prompt=str(record["prompt"]),
        gold_answer=str(record["gold_answer"]),
        model_answer=str(record["model_answer"]),
        score=float(record["score"]),
        metric=str(record["metric"]),
        passed=bool(record["passed"]),
        task_type=str(record["task_type"]),
        metadata=record["metadata"] if isinstance(record["metadata"], dict) else {},
    )


def results_from_payload(payload: dict[str, Any]) -> list[EvaluationResult]:
    return [result_from_dict(record) for record in payload["results"]]
