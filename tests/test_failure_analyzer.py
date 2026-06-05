from openevallab.analysis import FailureMode, analyze_failures, classify_failure
from openevallab.benchmarks import BenchmarkExample
from openevallab.evaluator import evaluate_benchmark
from openevallab.models import MockModelClient


def _result(prediction: str, task_type="qa", metadata=None):
    example = BenchmarkExample("x", task_type, "Question?", "Answer", metadata or {})
    return evaluate_benchmark([example], MockModelClient({"Question?": prediction}))[0]


def test_classifies_incomplete_answer():
    failure = classify_failure(_result("I do not know."))
    assert failure is not None
    assert failure.mode == FailureMode.INCOMPLETE_ANSWER


def test_analyze_failures_counts_distribution_and_suggestions():
    analysis = analyze_failures([_result("wrong", metadata={"domain": "biomed"})])
    assert analysis["total_failures"] == 1
    assert analysis["failure_mode_counts"]["factual_error"] == 1
    assert analysis["failure_mode_percentages"]["factual_error"] == 100.0
    assert analysis["representative_examples"]
    assert analysis["suggested_synthetic_data_targets"]
