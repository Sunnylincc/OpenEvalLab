"""Markdown experiment report renderer."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from openevallab.analysis import analyze_failures
from openevallab.evaluator import EvaluationResult, summarize_scores


def _percent(value: float) -> str:
    return f"{value * 100:.1f}%"


def render_markdown_report(
    *,
    model_name: str,
    benchmark_path: str,
    results: list[EvaluationResult],
    failure_analysis: dict[str, Any] | None = None,
    generated_at: datetime | None = None,
    max_examples: int = 5,
) -> str:
    """Render a readable, reproducible experiment report in Markdown."""

    failure_analysis = failure_analysis or analyze_failures(results)
    score_summary = summarize_scores(results)
    generated_at = generated_at or datetime.now(timezone.utc)
    failures = failure_analysis.get("failures", [])
    representative = failure_analysis.get("representative_examples", [])

    lines = [
        "# OpenEvalLab Evaluation Report",
        "",
        "## Run Summary",
        f"- Generated: {generated_at.isoformat(timespec='seconds')}",
        f"- Model: `{model_name}`",
        f"- Benchmark: `{benchmark_path}`",
        f"- Number of examples: {score_summary['num_examples']}",
        f"- Mean score: {score_summary['mean_score']:.3f}",
        f"- Pass rate: {_percent(float(score_summary['pass_rate']))}",
        "",
        "## Failure Mode Distribution",
        "",
        "| Failure mode | Count | Percentage |",
        "| --- | ---: | ---: |",
    ]
    distribution = failure_analysis.get("distribution", {})
    for mode, stats in distribution.items():
        lines.append(f"| `{mode}` | {stats.get('count', 0)} | {stats.get('percentage', 0.0):.1f}% |")

    lines.extend(["", "## Representative Failed Examples"])
    if not representative:
        lines.append("No failed examples were observed.")
    for item in representative[:max_examples]:
        mode = item.get("mode", "unknown") if isinstance(item, dict) else item.mode.value
        example_id = item.get("example_id") if isinstance(item, dict) else item.example_id
        prompt = item.get("prompt") if isinstance(item, dict) else item.prompt
        gold = item.get("gold_answer") if isinstance(item, dict) else item.gold_answer
        answer = item.get("model_answer") if isinstance(item, dict) else item.model_answer
        rationale = item.get("rationale") if isinstance(item, dict) else item.rationale
        lines.extend(
            [
                f"### {example_id} — `{mode}`",
                f"- Prompt: {prompt}",
                f"- Gold answer: {gold}",
                f"- Model answer: {answer}",
                f"- Analysis: {rationale}",
            ]
        )

    lines.extend(["", "## Synthetic Data Recommendations"])
    for suggestion in failure_analysis.get("suggested_synthetic_data_targets", []):
        lines.append(f"- {suggestion}")

    lines.extend(
        [
            "",
            "## Suggested Next Steps",
            "- Inspect representative failures before changing prompts or model settings.",
            "- Generate a small synthetic set for the dominant failure modes and review it manually.",
            "- Re-run evaluation on both the original and synthetic sets to check whether changes generalize.",
        ]
    )
    return "\n".join(lines) + "\n"
