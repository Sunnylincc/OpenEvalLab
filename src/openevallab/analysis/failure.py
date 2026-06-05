"""Rule-based failure mode analysis."""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass
from enum import Enum
from typing import Any

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
    model_answer: str
    rationale: str
    metadata: dict[str, Any]

    @property
    def prediction(self) -> str:
        return self.model_answer

    def to_dict(self) -> dict[str, Any]:
        record = asdict(self)
        record["mode"] = self.mode.value
        return record


def classify_failure(result: EvaluationResult) -> FailureRecord | None:
    """Classify one failed evaluation result with transparent heuristics."""

    if result.passed:
        return None
    prediction = result.model_answer.strip()
    lowered = prediction.lower()
    metadata = result.metadata
    if not prediction or lowered in {"i don't know", "i do not know.", "i do not know", "unknown", "n/a"}:
        mode = FailureMode.INCOMPLETE_ANSWER
        rationale = "The model returned no substantive answer."
    elif any(token in lowered for token in ["as an ai", "cannot comply", "i can't", "i cannot"]):
        mode = FailureMode.INSTRUCTION_FOLLOWING_FAILURE
        rationale = "The response refused or ignored the requested task."
    elif result.task_type in {"reasoning", "arithmetic_reasoning", "logic", "multi_step_arithmetic"} or any(
        token in lowered for token in ["because", "therefore", "so the answer", "step"]
    ):
        mode = FailureMode.REASONING_ERROR
        rationale = "The task or response indicates a reasoning attempt with an incorrect final answer."
    elif metadata.get("domain") in {"biomed", "science", "medicine"} or result.task_type in {"biomed_qa", "fact_qa"}:
        mode = FailureMode.FACTUAL_ERROR
        rationale = "The missed domain QA item is likely a factual error."
    elif len(prediction.split()) > 35 and result.gold_answer.lower() not in lowered:
        mode = FailureMode.HALLUCINATION
        rationale = "The response is long, misses the reference answer, and may include unsupported details."
    else:
        mode = FailureMode.UNKNOWN
        rationale = "No high-confidence heuristic matched."
    return FailureRecord(
        example_id=result.id,
        task_type=result.task_type,
        mode=mode,
        prompt=result.prompt,
        gold_answer=result.gold_answer,
        model_answer=result.model_answer,
        rationale=rationale,
        metadata=result.metadata,
    )


def suggested_targets(distribution: dict[str, dict[str, float | int]]) -> list[str]:
    """Create human-readable synthetic data targets from failure distribution."""

    suggestions = []
    if distribution.get("reasoning_error", {}).get("count", 0):
        suggestions.append("Add multi-step reasoning examples with short, verifiable final answers.")
    if distribution.get("factual_error", {}).get("count", 0):
        suggestions.append("Add fact-checking QA examples grounded in common, stable facts.")
    if distribution.get("incomplete_answer", {}).get("count", 0):
        suggestions.append("Add answer-completion examples that require concise but complete responses.")
    if distribution.get("hallucination", {}).get("count", 0):
        suggestions.append("Add grounded-answer examples that penalize unsupported details.")
    if distribution.get("instruction_following_failure", {}).get("count", 0):
        suggestions.append("Add format-following examples with explicit response constraints.")
    if not suggestions:
        suggestions.append("No dominant failure mode found; expand the benchmark with more diverse examples.")
    return suggestions


def analyze_failures(results: list[EvaluationResult]) -> dict[str, Any]:
    """Return failure records, counts, percentages, and synthesis suggestions."""

    failures = [record for result in results if (record := classify_failure(result)) is not None]
    counts = Counter(record.mode.value for record in failures)
    total = len(failures)
    distribution = {
        mode.value: {
            "count": counts.get(mode.value, 0),
            "percentage": (counts.get(mode.value, 0) / total * 100.0) if total else 0.0,
        }
        for mode in FailureMode
    }
    return {
        "total_failures": total,
        "failure_mode_counts": {mode.value: counts.get(mode.value, 0) for mode in FailureMode},
        "failure_mode_percentages": {
            mode.value: distribution[mode.value]["percentage"] for mode in FailureMode
        },
        "distribution": distribution,
        "representative_examples": [failure.to_dict() for failure in failures[:5]],
        "suggested_synthetic_data_targets": suggested_targets(distribution),
        "failures": failures,
    }
