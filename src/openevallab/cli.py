"""Command-line interface for OpenEvalLab."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Any

from openevallab.analysis import analyze_failures
from openevallab.benchmarks import load_jsonl_benchmark
from openevallab.evaluator import (
    evaluate_benchmark,
    load_results,
    results_from_payload,
    results_payload,
    save_results,
)
from openevallab.metrics import METRICS
from openevallab.models import MockModelClient, OpenAICompatibleClient
from openevallab.reports import render_markdown_report
from openevallab.synthesis import generate_synthetic_prompts, write_synthetic_jsonl


def _fail(message: str) -> None:
    print(f"Error: {message}", file=sys.stderr)
    raise SystemExit(2)


def _model_client(name: str):
    if name == "mock":
        return MockModelClient()
    if name in {"openai", "openai-compatible"}:
        try:
            return OpenAICompatibleClient(model_name=name)
        except ValueError as exc:
            _fail(str(exc))
    _fail("Unsupported model name. Use '--model mock' for local deterministic runs.")


def _load_benchmark_or_exit(path: str):
    benchmark_path = Path(path)
    if not benchmark_path.exists():
        _fail(f"Benchmark file not found: {benchmark_path}")
    try:
        examples = load_jsonl_benchmark(benchmark_path)
    except ValueError as exc:
        _fail(str(exc))
    if not examples:
        _fail(f"Benchmark is empty: {benchmark_path}")
    return examples


def eval_command(args: argparse.Namespace) -> None:
    examples = _load_benchmark_or_exit(args.benchmark)
    model = _model_client(args.model)
    try:
        results = evaluate_benchmark(examples, model, metric=args.metric)
    except ValueError as exc:
        _fail(str(exc))
    payload = results_payload(results=results, model_name=model.model_name, benchmark_path=args.benchmark)
    save_results(payload, args.out)
    aggregate = payload["aggregate_metrics"]
    print("OpenEvalLab evaluation complete")
    print(f"  Benchmark: {args.benchmark}")
    print(f"  Model: {model.model_name}")
    print(f"  Examples: {aggregate['num_examples']}")
    print(f"  Mean score: {aggregate['mean_score']:.3f}")
    print(f"  Pass rate: {aggregate['pass_rate'] * 100:.1f}%")
    print(f"  Results: {args.out}")


def _load_results_or_exit(path: str) -> tuple[dict[str, Any], list]:
    try:
        payload = load_results(path)
        return payload, results_from_payload(payload)
    except ValueError as exc:
        _fail(str(exc))


def analyze_command(args: argparse.Namespace) -> None:
    payload, results = _load_results_or_exit(args.results)
    analysis = analyze_failures(results)
    report = render_markdown_report(
        model_name=payload.get("model_name", "unknown"),
        benchmark_path=payload.get("benchmark_path", "unknown"),
        results=results,
        failure_analysis=analysis,
    )
    output_path = Path(args.out)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(report, encoding="utf-8")
    print("OpenEvalLab analysis complete")
    print(f"  Failures: {analysis['total_failures']}")
    print(f"  Report: {output_path}")


def synthesize_command(args: argparse.Namespace) -> None:
    _, results = _load_results_or_exit(args.results)
    analysis = analyze_failures(results)
    synthetic = generate_synthetic_prompts(analysis["failures"], num_examples=args.num_examples)
    write_synthetic_jsonl(synthetic, args.out)
    print("OpenEvalLab synthetic data generation complete")
    print(f"  Examples written: {len(synthetic)}")
    print(f"  Output: {args.out}")


def demo_command(args: argparse.Namespace) -> None:
    benchmark = args.benchmark or "data/sample_reasoning.jsonl"
    results_path = Path(args.results_out)
    report_path = Path(args.report_out)
    examples = _load_benchmark_or_exit(benchmark)
    model = MockModelClient()
    results = evaluate_benchmark(examples, model, metric="contains_answer")
    payload = results_payload(results=results, model_name=model.model_name, benchmark_path=benchmark)
    save_results(payload, results_path)
    analysis = analyze_failures(results)
    report = render_markdown_report(
        model_name=model.model_name,
        benchmark_path=benchmark,
        results=results,
        failure_analysis=analysis,
    )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report, encoding="utf-8")
    aggregate = payload["aggregate_metrics"]
    print("Welcome to OpenEvalLab — demo complete")
    print(f"  Evaluated {aggregate['num_examples']} examples with the local mock model.")
    print(f"  Mean score: {aggregate['mean_score']:.3f}")
    print(f"  Pass rate: {aggregate['pass_rate'] * 100:.1f}%")
    print(f"  Results JSON: {results_path}")
    print(f"  Markdown report: {report_path}")
    print("Next: try `openevallab synthesize --results results/demo_results.json --out data/synthetic_demo.jsonl --num-examples 10`.")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="openevallab",
        description="Evaluate LLM outputs, analyze failures, synthesize targeted data, and write reports.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    demo_parser = subparsers.add_parser("demo", help="Run a no-key local demo and write results/report files.")
    demo_parser.add_argument("--benchmark", help="Optional benchmark path for the demo.")
    demo_parser.add_argument("--results-out", default="results/demo_results.json")
    demo_parser.add_argument("--report-out", default="reports/demo_report.md")
    demo_parser.set_defaults(func=demo_command)

    eval_parser = subparsers.add_parser("eval", help="Evaluate a JSONL benchmark.")
    eval_parser.add_argument("--model", default="mock", help="Model client to use: mock or openai-compatible.")
    eval_parser.add_argument("--benchmark", required=True, help="Path to a JSONL benchmark.")
    eval_parser.add_argument("--out", required=True, help="Path for results JSON.")
    eval_parser.add_argument("--metric", choices=sorted(METRICS), default="contains_answer")
    eval_parser.set_defaults(func=eval_command)

    analyze_parser = subparsers.add_parser("analyze", help="Analyze a results JSON file and write a report.")
    analyze_parser.add_argument("--results", required=True, help="Path to results JSON from `openevallab eval`.")
    analyze_parser.add_argument("--out", required=True, help="Path for Markdown report.")
    analyze_parser.set_defaults(func=analyze_command)

    synth_parser = subparsers.add_parser("synthesize", help="Generate JSONL examples from observed failures.")
    synth_parser.add_argument("--results", required=True, help="Path to results JSON from `openevallab eval`.")
    synth_parser.add_argument("--out", required=True, help="Path for synthetic JSONL benchmark.")
    synth_parser.add_argument("--num-examples", type=int, default=20)
    synth_parser.set_defaults(func=synthesize_command)
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
