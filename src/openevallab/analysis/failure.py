"""Rule-based failure mode analysis."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from enum import Enum

from openevallab.evaluator import EvaluationResult


class FailureMode(str, Enum):
    FACTUAL_ERROR = "factual_error"
    REASONING_ERROR = "reasoning_error"
    INSTRUCTION_FOLLOWING_FAILURE = "instruction_following_failure"
    INCOMPLETE_ANSWER = "incomplete_answer"
    HALLUCINATION = "hallucination"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class FailureRecord:
    example_id: str
    task_type: str
    mode: FailureMode
    prompt: str
    gold_answer: str
    prediction: str
    rationale: str


def classify_failure(result: EvaluationResult) -> FailureRecord | None:
    """Classify one failed evaluation result with transparent heuristics."""

    if result.passed:
        return None
    prediction = result.prediction.strip()
    lowered = prediction.lower()
    metadata = result.example.metadata
    if not prediction or lowered in {"i don't know", "i do not know.", "unknown", "n/a"}:
        mode = FailureMode.INCOMPLETE_ANSWER
        rationale = "Prediction was empty or explicitly incomplete."
    elif any(token in lowered for token in ["as an ai", "cannot comply", "i can't", "i cannot"]):
        mode = FailureMode.INSTRUCTION_FOLLOWING_FAILURE
        rationale = "Prediction refused or did not follow the task instruction."
    elif any(token in lowered for token in ["because", "therefore", "so the answer", "step"]):
        mode = FailureMode.REASONING_ERROR
        rationale = "Prediction attempted reasoning but reached the wrong answer."
    elif metadata.get("domain") in {"biomed", "science", "medicine"}:
        mode = FailureMode.FACTUAL_ERROR
        rationale = "Domain QA miss suggests an incorrect factual claim."
    elif len(prediction.split()) > 40 and result.example.gold_answer.lower() not in lowered:
        mode = FailureMode.HALLUCINATION
        rationale = "Long answer omitted the gold answer and may introduce unsupported details."
    else:
        mode = FailureMode.UNKNOWN
        rationale = "No high-confidence rule matched."
    return FailureRecord(
        example_id=result.example.id,
        task_type=result.example.task_type,
        mode=mode,
        prompt=result.example.prompt,
        gold_answer=result.example.gold_answer,
        prediction=result.prediction,
        rationale=rationale,
    )


def analyze_failures(results: list[EvaluationResult]) -> dict:
    """Return failure records and aggregate distribution."""

    failures = [record for result in results if (record := classify_failure(result)) is not None]
    counts = Counter(record.mode.value for record in failures)
    total = len(failures)
    distribution = {
        mode.value: {
            "count": counts.get(mode.value, 0),
            "rate": (counts.get(mode.value, 0) / total) if total else 0.0,
        }
        for mode in FailureMode
    }
    return {"total_failures": total, "distribution": distribution, "failures": failures}
