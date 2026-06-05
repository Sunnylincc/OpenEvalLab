"""Generate a Markdown report for a local benchmark run."""

from pathlib import Path

from openevallab.analysis import analyze_failures
from openevallab.benchmarks import load_jsonl_benchmark
from openevallab.evaluator import evaluate_benchmark
from openevallab.models import MockModelClient
from openevallab.reports import render_markdown_report

benchmark_path = "data/sample_reasoning.jsonl"
examples = load_jsonl_benchmark(benchmark_path)
model = MockModelClient(default_response="I do not know.")
results = evaluate_benchmark(examples, model)
report = render_markdown_report(
    model_name=model.model_name,
    benchmark_name=Path(benchmark_path).stem,
    results=results,
    failure_analysis=analyze_failures(results),
)
print(report)
