from openevallab.analysis import FailureMode, analyze_failures, classify_failure
from openevallab.benchmarks import BenchmarkExample
from openevallab.evaluator import EvaluationResult
from openevallab.metrics import exact_match


def _result(prediction: str, metadata=None):
    example = BenchmarkExample("x", "qa", "Question?", "Answer", metadata or {})
    return EvaluationResult(example, prediction, {"exact_match": exact_match(prediction, "Answer")})


def test_classifies_incomplete_answer():
    failure = classify_failure(_result("I do not know."))
    assert failure is not None
    assert failure.mode == FailureMode.INCOMPLETE_ANSWER


def test_analyze_failures_counts_distribution():
    analysis = analyze_failures([_result("wrong", {"domain": "biomed"})])
    assert analysis["total_failures"] == 1
    assert analysis["distribution"]["factual_error"]["count"] == 1
