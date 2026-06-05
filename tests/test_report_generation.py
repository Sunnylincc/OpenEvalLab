from datetime import datetime, timezone

from openevallab.analysis import analyze_failures
from openevallab.benchmarks import BenchmarkExample
from openevallab.evaluator import evaluate_benchmark
from openevallab.models import MockModelClient
from openevallab.reports import render_markdown_report


def test_report_contains_required_sections():
    example = BenchmarkExample("r1", "reasoning", "Two plus two?", "4", {"domain": "reasoning"})
    result = evaluate_benchmark([example], MockModelClient({"Two plus two?": "5 because I added incorrectly"}))[0]
    report = render_markdown_report(
        model_name="mock",
        benchmark_path="sample.jsonl",
        results=[result],
        failure_analysis=analyze_failures([result]),
        generated_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
    )
    assert "# OpenEvalLab Evaluation Report" in report
    assert "## Failure Mode Distribution" in report
    assert "## Representative Failed Examples" in report
    assert "## Synthetic Data Recommendations" in report
    assert "2026-01-01T00:00:00+00:00" in report
