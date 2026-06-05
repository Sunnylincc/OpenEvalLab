"""Benchmark evaluation loop."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from openevallab.benchmarks import BenchmarkExample
from openevallab.metrics import MetricResult, contains_answer, exact_match
from openevallab.models import BaseModelClient

MetricFn = Callable[[str, str], MetricResult]


@dataclass(frozen=True)
class EvaluationResult:
    example: BenchmarkExample
    prediction: str
    metrics: dict[str, MetricResult]

    @property
    def passed(self) -> bool:
        return any(result.passed for result in self.metrics.values())


def evaluate_benchmark(
    examples: list[BenchmarkExample],
    model_client: BaseModelClient,
    *,
    metric_fns: list[MetricFn] | None = None,
) -> list[EvaluationResult]:
    """Evaluate examples with a model client and metric functions."""

    metrics = metric_fns or [exact_match, contains_answer]
    results: list[EvaluationResult] = []
    for example in examples:
        response = model_client.generate(example.prompt, metadata=example.metadata)
        metric_results = {}
        for metric in metrics:
            metric_result = metric(response.text, example.gold_answer)
            metric_results[metric_result.name] = metric_result
        results.append(
            EvaluationResult(example=example, prediction=response.text, metrics=metric_results)
        )
    return results


def summarize_scores(results: list[EvaluationResult]) -> dict[str, float]:
    """Aggregate average metric scores and overall pass rate."""

    if not results:
        return {"num_examples": 0.0, "pass_rate": 0.0}
    metric_names = sorted({name for result in results for name in result.metrics})
    summary = {
        "num_examples": float(len(results)),
        "pass_rate": sum(r.passed for r in results) / len(results),
    }
    for name in metric_names:
        values = [result.metrics[name].score for result in results if name in result.metrics]
        summary[name] = sum(values) / len(values) if values else 0.0
    return summary
