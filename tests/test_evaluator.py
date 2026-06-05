import json

from openevallab.benchmarks import BenchmarkExample
from openevallab.evaluator import evaluate_benchmark, results_payload, summarize_scores
from openevallab.models import MockModelClient


def test_evaluation_result_schema():
    examples = [BenchmarkExample("example_001", "reasoning", "2 plus 2?", "4", {})]
    results = evaluate_benchmark(examples, MockModelClient({"2 plus 2?": "4"}))
    record = results_payload(results=results, model_name="mock", benchmark_path="bench.jsonl")["results"][0]
    assert set(record) >= {"id", "prompt", "gold_answer", "model_answer", "score", "metric", "passed", "task_type", "metadata"}
    assert record["passed"] is True


def test_summarize_scores():
    examples = [BenchmarkExample("1", "qa", "Q", "A", {})]
    results = evaluate_benchmark(examples, MockModelClient({"Q": "wrong"}))
    summary = summarize_scores(results)
    assert summary == {"num_examples": 1, "mean_score": 0.0, "pass_rate": 0.0}
