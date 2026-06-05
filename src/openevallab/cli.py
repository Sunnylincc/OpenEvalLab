"""Command-line interface for OpenEvalLab."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from openevallab.analysis import analyze_failures
from openevallab.benchmarks import load_jsonl_benchmark
from openevallab.evaluator import evaluate_benchmark, summarize_scores
from openevallab.models import MockModelClient
from openevallab.reports import render_markdown_report
from openevallab.synthesis import generate_synthetic_prompts


def _build_mock(examples, mode: str) -> MockModelClient:
    if mode == "gold":
        return MockModelClient({example.prompt: example.gold_answer for example in examples})
    return MockModelClient(default_response="I do not know.")


def eval_command(args: argparse.Namespace) -> None:
    examples = load_jsonl_benchmark(args.benchmark)
    model = _build_mock(examples, args.mock_mode)
    results = evaluate_benchmark(examples, model)
    summary = summarize_scores(results)
    print(json.dumps(summary, indent=2, sort_keys=True))


def report_command(args: argparse.Namespace) -> None:
    examples = load_jsonl_benchmark(args.benchmark)
    model = _build_mock(examples, args.mock_mode)
    results = evaluate_benchmark(examples, model)
    analysis = analyze_failures(results)
    report = render_markdown_report(
        model_name=model.model_name,
        benchmark_name=Path(args.benchmark).stem,
        results=results,
        failure_analysis=analysis,
    )
    if args.output:
        Path(args.output).write_text(report, encoding="utf-8")
    else:
        print(report)


def synthesize_command(args: argparse.Namespace) -> None:
    examples = load_jsonl_benchmark(args.benchmark)
    model = _build_mock(examples, args.mock_mode)
    results = evaluate_benchmark(examples, model)
    failures = analyze_failures(results)["failures"]
    synthetic = generate_synthetic_prompts(failures, per_failure=args.per_failure)
    for example in synthetic:
        print(json.dumps(example.__dict__, ensure_ascii=False))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="openevallab", description="Evaluate models and analyze failures.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    eval_parser = subparsers.add_parser("eval", help="Run a local benchmark evaluation.")
    eval_parser.add_argument("benchmark", help="Path to a JSONL benchmark.")
    eval_parser.add_argument("--mock-mode", choices=["default", "gold"], default="default")
    eval_parser.set_defaults(func=eval_command)

    report_parser = subparsers.add_parser("report", help="Generate a Markdown report.")
    report_parser.add_argument("benchmark", help="Path to a JSONL benchmark.")
    report_parser.add_argument("--mock-mode", choices=["default", "gold"], default="default")
    report_parser.add_argument("--output", "-o", help="Optional output Markdown path.")
    report_parser.set_defaults(func=report_command)

    synth_parser = subparsers.add_parser("synthesize", help="Generate synthetic prompt candidates.")
    synth_parser.add_argument("benchmark", help="Path to a JSONL benchmark.")
    synth_parser.add_argument("--mock-mode", choices=["default", "gold"], default="default")
    synth_parser.add_argument("--per-failure", type=int, default=1)
    synth_parser.set_defaults(func=synthesize_command)
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
