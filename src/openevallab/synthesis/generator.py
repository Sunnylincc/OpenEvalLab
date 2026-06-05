"""Template-based synthetic benchmark generation from failure modes."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
from typing import Any

from openevallab.analysis import FailureMode, FailureRecord


@dataclass(frozen=True)
class SyntheticExample:
    id: str
    task_type: str
    prompt: str
    gold_answer: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_jsonl(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False)


TEMPLATE_BANK = {
    FailureMode.REASONING_ERROR: [
        ("reasoning", "A recipe needs 3 cups of flour per loaf. If you bake 4 loaves and already used 5 cups, how many more cups are needed?", "7"),
        ("reasoning", "A lab labels samples A, B, and C. A was collected before B, and C after B. Which sample was collected first?", "A"),
    ],
    FailureMode.FACTUAL_ERROR: [
        ("fact_checking", "Answer with the common scientific term: What gas do humans primarily exhale after cellular respiration?", "carbon dioxide"),
        ("fact_checking", "Answer with the organ name: Which organ filters blood to produce urine?", "kidney"),
    ],
    FailureMode.INCOMPLETE_ANSWER: [
        ("answer_completion", "Give a complete answer: Water freezes at what temperature in Celsius?", "0°C"),
        ("answer_completion", "Give a complete answer: What is the largest planet in the Solar System?", "Jupiter"),
    ],
    FailureMode.HALLUCINATION: [
        ("grounded_answer", "Use only the prompt: The passage says Nora owns two cats. How many cats does Nora own?", "two"),
        ("grounded_answer", "Use only the prompt: The box contains red and blue pens. Name one pen color in the box.", "red"),
    ],
    FailureMode.INSTRUCTION_FOLLOWING_FAILURE: [
        ("instruction_following", "Return only YES or NO: Is 8 greater than 3?", "YES"),
        ("instruction_following", "Return only the number: How many days are in a standard week?", "7"),
    ],
    FailureMode.UNKNOWN: [
        ("robust_qa", "Answer carefully: What is 10 minus 4?", "6"),
        ("robust_qa", "Answer carefully: What color do you get by mixing red and white paint?", "pink"),
    ],
}


def _failure_mode_counts(failures: list[FailureRecord]) -> list[FailureMode]:
    if not failures:
        return [FailureMode.REASONING_ERROR, FailureMode.FACTUAL_ERROR, FailureMode.INCOMPLETE_ANSWER]
    counts: dict[FailureMode, int] = {}
    for failure in failures:
        counts[failure.mode] = counts.get(failure.mode, 0) + 1
    return [mode for mode, _ in sorted(counts.items(), key=lambda item: (-item[1], item[0].value))]


def generate_synthetic_prompts(
    failures: list[FailureRecord],
    *,
    num_examples: int = 10,
) -> list[SyntheticExample]:
    """Generate valid benchmark-schema JSONL examples from observed failures."""

    if num_examples < 1:
        return []
    modes = _failure_mode_counts(failures)
    generated: list[SyntheticExample] = []
    index = 0
    while len(generated) < num_examples:
        mode = modes[index % len(modes)]
        template = TEMPLATE_BANK[mode][(index // len(modes)) % len(TEMPLATE_BANK[mode])]
        task_type, prompt, gold_answer = template
        generated.append(
            SyntheticExample(
                id=f"synthetic_{len(generated) + 1:04d}",
                task_type=task_type,
                prompt=prompt,
                gold_answer=gold_answer,
                metadata={
                    "source": "openevallab_template_generator",
                    "target_failure_mode": mode.value,
                    "generator": "template-v1",
                },
            )
        )
        index += 1
    return generated


def write_synthetic_jsonl(examples: list[SyntheticExample], path: str | Path) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(example.to_jsonl() for example in examples) + "\n", encoding="utf-8")
