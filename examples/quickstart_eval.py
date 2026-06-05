"""Run a tiny OpenEvalLab evaluation with the deterministic mock model."""

from openevallab.benchmarks import load_jsonl_benchmark
from openevallab.evaluator import evaluate_benchmark, results_payload
from openevallab.models import MockModelClient

examples = load_jsonl_benchmark("data/sample_reasoning.jsonl")
model = MockModelClient()
results = evaluate_benchmark(examples, model)
print(results_payload(results=results, model_name=model.model_name, benchmark_path="data/sample_reasoning.jsonl"))
