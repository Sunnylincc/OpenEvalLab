"""OpenEvalLab: lightweight model evaluation and failure-driven data synthesis."""

from openevallab.benchmarks import BenchmarkExample, load_jsonl_benchmark
from openevallab.evaluator import EvaluationResult, evaluate_benchmark

__all__ = [
    "BenchmarkExample",
    "EvaluationResult",
    "evaluate_benchmark",
    "load_jsonl_benchmark",
]

__version__ = "0.1.0"
