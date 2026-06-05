from openevallab.analysis import analyze_failures
from openevallab.benchmarks import BenchmarkExample
from openevallab.evaluator import EvaluationResult
from openevallab.metrics import exact_match
from openevallab.reports import render_markdown_report


def test_report_contains_required_sections():
    example = BenchmarkExample("r1", "qa", "Two plus two?", "4", {"domain": "reasoning"})
    result = EvaluationResult(example, "5 because I added incorrectly", {"exact_match": exact_match("5", "4")})
    report = render_markdown_report(
        model_name="mock-model",
        benchmark_name="sample",
        results=[result],
        failure_analysis=analyze_failures([result]),
    )
    assert "## Score Summary" in report
    assert "## Failure Mode Distribution" in report
    assert "## Representative Examples" in report
    assert "## Suggested Data Synthesis Targets" in report
