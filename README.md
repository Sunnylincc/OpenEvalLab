# OpenEvalLab

OpenEvalLab is a lightweight, extensible research toolkit for evaluating language models, analyzing failure modes, generating targeted synthetic data, and producing reproducible experiment reports.

The project is intentionally small in its first release: it favors transparent interfaces, plain JSONL benchmarks, deterministic local demos, and composable modules over heavyweight orchestration. The goal is to make evaluation loops easy to inspect, reproduce, and extend.

## Motivation

Language-model evaluation should not stop at a single aggregate score. Practical research workflows need to answer follow-up questions:

- Which examples failed, and why?
- Are failures concentrated in reasoning, factuality, instruction following, or incompleteness?
- What new examples should be generated to stress-test the observed weaknesses?
- Can a report be regenerated later from the same benchmark and evaluation configuration?

OpenEvalLab provides a compact foundation for that loop: **benchmark → model responses → metrics → failure analysis → synthetic data targets → Markdown report**.

## Installation

Install the package in editable mode from the repository root:

```bash
python -m pip install -e .
```

For test development:

```bash
python -m pip install -e '.[dev]'
pytest
```

## Quickstart

Run a deterministic evaluation that uses the included mock model and sample reasoning benchmark:

```bash
openevallab eval data/sample_reasoning.jsonl --mock-mode gold
```

Generate a report from failed mock-model outputs:

```bash
openevallab report data/sample_biomed_qa.jsonl --output biomed_report.md
```

Generate synthetic prompt candidates from observed failures:

```bash
openevallab synthesize data/sample_biomed_qa.jsonl --per-failure 2
```

You can also run the Python examples:

```bash
python examples/quickstart_eval.py
python examples/synthesize_from_errors.py
python examples/generate_report.py
```

## CLI Examples

### Evaluate a benchmark

```bash
openevallab eval data/sample_reasoning.jsonl
```

The command prints a JSON score summary containing the number of examples, pass rate, and average metric scores.

### Generate a Markdown report

```bash
openevallab report data/sample_reasoning.jsonl -o report.md
```

Reports include model metadata, benchmark name, score summary, failure-mode distribution, representative examples, and suggested synthesis targets.

### Create synthetic data candidates

```bash
openevallab synthesize data/sample_biomed_qa.jsonl --per-failure 1
```

The current generator is template-based. A model-based generation hook is intentionally left as an extension point so researchers can plug in their own generation policies.

## Benchmark Schema

Benchmarks are JSON Lines files with one object per example. Each record must include:

| Field | Type | Description |
| --- | --- | --- |
| `id` | string | Stable example identifier. |
| `task_type` | string | Task category, such as `biomed_qa` or `multi_step_arithmetic`. |
| `prompt` | string | Prompt sent to the model. |
| `gold_answer` | string | Reference answer used by metrics. |
| `metadata` | object | Free-form metadata for domains, difficulty, provenance, tags, or split names. |

Example:

```json
{"id":"reasoning-001","task_type":"arithmetic_reasoning","prompt":"If a train leaves at 2 PM and travels for 3 hours, what time does it arrive?","gold_answer":"5 PM","metadata":{"domain":"reasoning","difficulty":"easy"}}
```

## Package Structure

```text
src/openevallab/
  analysis/      failure-mode classification and aggregate statistics
  benchmarks/    JSONL schema and loading utilities
  metrics/       exact match, contains answer, and judge placeholders
  models/        BaseModelClient, mock client, and OpenAI-compatible adapter placeholder
  reports/       Markdown experiment reports
  synthesis/     failure-driven synthetic prompt generation
  cli.py         openevallab command-line interface
```

## Model Interfaces

OpenEvalLab defines a small `BaseModelClient` interface with a single `generate` method. The included `MockModelClient` is deterministic and suitable for demos, examples, and unit tests. `OpenAICompatibleClient` is provided as a placeholder adapter for researchers who want to wire in OpenAI-compatible chat completion APIs without changing the evaluation loop.

## Metrics

The first release includes:

- `exact_match`: normalized string equality.
- `contains_answer`: checks whether the normalized gold answer appears in the prediction.
- `llm_judge_placeholder`: explicit placeholder for future semantic judging with a configured model and rubric.

## Failure Modes

The rule-based analyzer classifies failed examples into:

- factual error
- reasoning error
- instruction-following failure
- incomplete answer
- hallucination
- unknown

The heuristics are deliberately transparent. They are designed to be replaced or augmented by domain-specific analyzers as projects mature.

## Roadmap

- Add configurable OpenAI-compatible and local inference clients.
- Support richer metric registries and task-specific scoring.
- Add model-based synthetic data generation policies.
- Persist run artifacts as versioned experiment directories.
- Add HTML report rendering and comparison reports across models.
- Support benchmark cards with provenance, licensing, and intended-use metadata.

## Contributing

Contributions are welcome. Good first contributions include:

- New benchmark loaders or exporters.
- Additional transparent metrics.
- Domain-specific failure classifiers.
- Report templates.
- Examples that reproduce published evaluation workflows.

Please keep changes small, documented, and covered by tests. OpenEvalLab is meant to remain easy to audit and easy to adapt for research settings.
