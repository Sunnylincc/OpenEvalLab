"""Run a tiny OpenEvalLab evaluation with the deterministic mock model."""

from openevallab.benchmarks import load_jsonl_benchmark
from openevallab.evaluator import evaluate_benchmark, summarize_scores
from openevallab.models import MockModelClient

examples = load_jsonl_benchmark("data/sample_reasoning.jsonl")
model = MockModelClient({example.prompt: example.gold_answer for example in examples})
results = evaluate_benchmark(examples, model)
print(summarize_scores(results))
