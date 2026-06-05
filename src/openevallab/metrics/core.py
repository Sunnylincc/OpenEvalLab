"""Core metric implementations."""

from __future__ import annotations

from dataclasses import dataclass
import re
import string


@dataclass(frozen=True)
class MetricResult:
    """Score and explanation for a metric."""

    name: str
    score: float
    passed: bool
    explanation: str


def normalize_text(text: str) -> str:
    """Normalize case, whitespace, punctuation, and English articles."""

    lowered = text.lower().strip()
    no_punct = "".join(ch for ch in lowered if ch not in string.punctuation)
    no_articles = re.sub(r"\b(a|an|the)\b", " ", no_punct)
    return re.sub(r"\s+", " ", no_articles).strip()


def exact_match(prediction: str, gold_answer: str) -> MetricResult:
    passed = prediction.strip() == gold_answer.strip()
    return MetricResult(
        name="exact_match",
        score=1.0 if passed else 0.0,
        passed=passed,
        explanation="Prediction exactly matched the gold answer." if passed else "No exact match.",
    )


def normalized_exact_match(prediction: str, gold_answer: str) -> MetricResult:
    passed = normalize_text(prediction) == normalize_text(gold_answer)
    return MetricResult(
        name="normalized_exact_match",
        score=1.0 if passed else 0.0,
        passed=passed,
        explanation="Normalized prediction matched the gold answer." if passed else "No normalized match.",
    )


def contains_answer(prediction: str, gold_answer: str) -> MetricResult:
    normalized_gold = normalize_text(gold_answer)
    normalized_prediction = normalize_text(prediction)
    passed = bool(normalized_gold) and normalized_gold in normalized_prediction
    return MetricResult(
        name="contains_answer",
        score=1.0 if passed else 0.0,
        passed=passed,
        explanation="Prediction contains the normalized gold answer." if passed else "Gold answer not found.",
    )


def heuristic_score(prediction: str, gold_answer: str) -> MetricResult:
    """A simple token-overlap score useful for quick experiments."""

    gold_tokens = set(normalize_text(gold_answer).split())
    prediction_tokens = set(normalize_text(prediction).split())
    if not gold_tokens:
        score = 0.0
    else:
        score = len(gold_tokens & prediction_tokens) / len(gold_tokens)
    return MetricResult(
        name="heuristic_score",
        score=score,
        passed=score >= 0.5,
        explanation=f"Token overlap with gold answer: {score:.2f}.",
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


METRICS = {
    "exact_match": exact_match,
    "normalized_exact_match": normalized_exact_match,
    "contains_answer": contains_answer,
    "heuristic_score": heuristic_score,
}
