# OpenEvalLab

OpenEvalLab is a small Python toolkit for evaluating language-model outputs, inspecting failure modes, generating targeted synthetic benchmark examples, and writing reproducible Markdown reports.

It is designed for research workflows where a single score is not enough. A useful evaluation run should make it easy to see **what failed**, **why it failed**, and **what data to create next**.

## Install

```bash
git clone <repo>
cd OpenEvalLab
pip install -e .
```

The base package has no runtime dependencies. The demo uses a deterministic mock model and does not require an API key.

## 30-second demo

```bash
openevallab demo
```

This writes:

```text
results/demo_results.json
reports/demo_report.md
```

The terminal output summarizes the benchmark, mean score, pass rate, and paths to the generated artifacts.

## CLI usage

Evaluate a JSONL benchmark with the local mock model:

```bash
openevallab eval \
  --model mock \
  --benchmark data/sample_reasoning.jsonl \
  --out results/reasoning_results.json
```

Analyze a result file and write a Markdown report:

```bash
openevallab analyze \
  --results results/reasoning_results.json \
  --out reports/reasoning_report.md
```

Generate synthetic JSONL examples targeted at observed failures:

```bash
openevallab synthesize \
  --results results/reasoning_results.json \
  --out data/synthetic_reasoning.jsonl \
  --num-examples 20
```

Available metrics:

- `exact_match`
- `normalized_exact_match`
- `contains_answer` (default)
- `heuristic_score`

## Python API usage

```python
from openevallab.analysis import analyze_failures
from openevallab.benchmarks import load_jsonl_benchmark
from openevallab.evaluator import evaluate_benchmark, summarize_scores
from openevallab.models import MockModelClient
from openevallab.reports import render_markdown_report

examples = load_jsonl_benchmark("data/sample_reasoning.jsonl")
model = MockModelClient()
results = evaluate_benchmark(examples, model, metric="contains_answer")
analysis = analyze_failures(results)

print(summarize_scores(results))
print(render_markdown_report(
    model_name=model.model_name,
    benchmark_path="data/sample_reasoning.jsonl",
    results=results,
    failure_analysis=analysis,
))
```

## Benchmark schema

Benchmarks are JSON Lines files. Each line must be a JSON object with these fields:

```json
{
  "id": "example_001",
  "task_type": "reasoning",
  "prompt": "Question text here",
  "gold_answer": "Expected answer here",
  "metadata": {
    "source": "demo",
    "difficulty": "easy"
  }
}
```

Validation checks that required fields are present, string fields are non-empty, and `metadata` is an object. Errors include the line number when possible.

Included sample datasets:

- `data/sample_reasoning.jsonl`
- `data/sample_biomed_qa.jsonl`

The biomedical examples are generic, safe educational QA items and avoid unpublished or specialized claims.

## Result file format

`openevallab eval` writes a JSON file with run metadata, aggregate metrics, and one record per example:

```json
{
  "schema_version": "1.0",
  "model_name": "mock",
  "benchmark_path": "data/sample_reasoning.jsonl",
  "aggregate_metrics": {
    "num_examples": 3,
    "mean_score": 1.0,
    "pass_rate": 1.0
  },
  "results": [
    {
      "id": "example_001",
      "prompt": "Question text here",
      "gold_answer": "Expected answer here",
      "model_answer": "Expected answer here",
      "score": 1.0,
      "metric": "contains_answer",
      "passed": true,
      "task_type": "reasoning",
      "metadata": {}
    }
  ]
}
```

## Example report

Reports are Markdown files with:

- title and generation time
- model name and benchmark path
- number of examples, mean score, and pass rate
- failure-mode table
- representative failed examples
- synthetic data recommendations
- suggested next steps

Run `openevallab demo` and open `reports/demo_report.md` to see a generated report.

## Project structure

```text
src/openevallab/
  analysis/      failure-mode classification and aggregate statistics
  benchmarks/    JSONL schema and loading utilities
  metrics/       exact, normalized, contains-answer, and heuristic scoring
  models/        BaseModelClient, mock client, and OpenAI-compatible placeholder
  reports/       Markdown report rendering
  synthesis/     template-based synthetic benchmark generation
  cli.py         command-line interface

data/             small sample benchmarks
examples/         Python API examples
tests/            unit and CLI tests
```

## Design notes

- The default path is local and deterministic: `openevallab demo` uses `MockModelClient`.
- `OpenAICompatibleClient` is a configuration placeholder for future provider integrations and reads `OPENAI_API_KEY` when used.
- Failure analysis and synthetic data generation are intentionally heuristic in this first version so they remain auditable and easy to replace.

## Development

```bash
make install
make test
make demo
make clean
```

Or run tests directly:

```bash
pytest
```

## Roadmap

- Add optional working provider clients while keeping the base install dependency-light.
- Add task-specific metric registries and benchmark cards.
- Add model-assisted failure classification behind explicit configuration.
- Add report comparison across runs and models.
- Add artifact directories for reproducible experiment bundles.

## Contributing

Contributions are welcome. Please keep changes focused, documented, and covered by tests. Useful contributions include new benchmark loaders, task-specific metrics, domain-specific failure analyzers, report templates, and well-documented example workflows.
