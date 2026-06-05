"""Markdown experiment report renderer."""

from __future__ import annotations

from openevallab.analysis import FailureRecord
from openevallab.evaluator import EvaluationResult, summarize_scores
from openevallab.synthesis import generate_synthetic_prompts


def _percent(value: float) -> str:
    return f"{value * 100:.1f}%"


def render_markdown_report(
    *,
    model_name: str,
    benchmark_name: str,
    results: list[EvaluationResult],
    failure_analysis: dict,
    max_examples: int = 5,
) -> str:
    """Render a reproducible experiment report in Markdown."""

    score_summary = summarize_scores(results)
    failures: list[FailureRecord] = failure_analysis.get("failures", [])
    synthetic_targets = generate_synthetic_prompts(failures[:max_examples])

    lines = [
        f"# OpenEvalLab Report: {benchmark_name}",
        "",
        "## Run Metadata",
        f"- Model: `{model_name}`",
        f"- Benchmark: `{benchmark_name}`",
        f"- Examples: {int(score_summary.get('num_examples', 0))}",
        "",
        "## Score Summary",
    ]
    for metric, value in score_summary.items():
        if metric == "num_examples":
            continue
        lines.append(f"- {metric}: {_percent(value)}")

    lines.extend(["", "## Failure Mode Distribution"])
    distribution = failure_analysis.get("distribution", {})
    for mode, stats in distribution.items():
        lines.append(f"- {mode}: {stats.get('count', 0)} ({_percent(stats.get('rate', 0.0))})")

    lines.extend(["", "## Representative Examples"])
    if not failures:
        lines.append("No failed examples were observed.")
    for failure in failures[:max_examples]:
        lines.extend(
            [
                f"### {failure.example_id} — {failure.mode.value}",
                f"- Prompt: {failure.prompt}",
                f"- Gold answer: {failure.gold_answer}",
                f"- Prediction: {failure.prediction}",
                f"- Analysis: {failure.rationale}",
            ]
        )

    lines.extend(["", "## Suggested Data Synthesis Targets"])
    if not synthetic_targets:
        lines.append("No synthesis targets suggested because no failures were observed.")
    for target in synthetic_targets:
        lines.append(f"- `{target.id}` ({target.metadata['source_failure_mode']}): {target.prompt}")

    return "\n".join(lines) + "\n"
