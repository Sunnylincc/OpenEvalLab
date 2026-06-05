"""Template-based synthetic prompt generation from failure modes."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from openevallab.analysis import FailureMode, FailureRecord


@dataclass(frozen=True)
class SyntheticExample:
    id: str
    task_type: str
    prompt: str
    gold_answer: str
    metadata: dict[str, Any] = field(default_factory=dict)


TEMPLATES = {
    FailureMode.FACTUAL_ERROR: "Answer concisely using only established facts: {prompt}",
    FailureMode.REASONING_ERROR: "Solve step by step, then give only the final answer: {prompt}",
    FailureMode.INSTRUCTION_FOLLOWING_FAILURE: "Follow the requested output format exactly. {prompt}",
    FailureMode.INCOMPLETE_ANSWER: "Provide a complete answer with the key missing detail: {prompt}",
    FailureMode.HALLUCINATION: "Answer only if supported by the prompt; avoid unsupported details: {prompt}",
    FailureMode.UNKNOWN: "Create a careful answer and state assumptions explicitly: {prompt}",
}


def generate_synthetic_prompts(failures: list[FailureRecord], *, per_failure: int = 1) -> list[SyntheticExample]:
    """Generate template-based synthetic benchmark candidates from failures."""

    synthetic: list[SyntheticExample] = []
    for failure in failures:
        template = TEMPLATES.get(failure.mode, TEMPLATES[FailureMode.UNKNOWN])
        for index in range(per_failure):
            synthetic.append(
                SyntheticExample(
                    id=f"synthetic-{failure.example_id}-{index + 1}",
                    task_type=failure.task_type,
                    prompt=template.format(prompt=failure.prompt),
                    gold_answer=failure.gold_answer,
                    metadata={
                        "source_failure_id": failure.example_id,
                        "source_failure_mode": failure.mode.value,
                        "generator": "template-v1",
                        "model_based_generation": "placeholder",
                    },
                )
            )
    return synthetic
