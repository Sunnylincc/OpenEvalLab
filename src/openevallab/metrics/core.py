"""Core metric implementations."""

from __future__ import annotations

from dataclasses import dataclass
import re


@dataclass(frozen=True)
class MetricResult:
    """Score and explanation for a metric."""

    name: str
    score: float
    passed: bool
    explanation: str


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())


def exact_match(prediction: str, gold_answer: str) -> MetricResult:
    passed = _normalize(prediction) == _normalize(gold_answer)
    return MetricResult(
        name="exact_match",
        score=1.0 if passed else 0.0,
        passed=passed,
        explanation="Normalized prediction exactly matched gold answer." if passed else "No exact match.",
    )


def contains_answer(prediction: str, gold_answer: str) -> MetricResult:
    passed = _normalize(gold_answer) in _normalize(prediction)
    return MetricResult(
        name="contains_answer",
        score=1.0 if passed else 0.0,
        passed=passed,
        explanation="Prediction contains the gold answer." if passed else "Gold answer not found.",
    )


def llm_judge_placeholder(prediction: str, gold_answer: str, rubric: str | None = None) -> MetricResult:
    return MetricResult(
        name="llm_judge_placeholder",
        score=0.0,
        passed=False,
        explanation=(
            "LLM judge is not executed in the local toolkit. Configure a judge client and rubric "
            "to enable semantic scoring."
        ),
    )
