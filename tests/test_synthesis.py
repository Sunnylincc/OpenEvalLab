import json

from openevallab.analysis import analyze_failures
from openevallab.benchmarks import BenchmarkExample, load_jsonl_benchmark
from openevallab.evaluator import evaluate_benchmark
from openevallab.models import MockModelClient
from openevallab.synthesis import generate_synthetic_prompts, write_synthetic_jsonl


def test_generate_synthetic_jsonl_matches_benchmark_schema(tmp_path):
    result = evaluate_benchmark(
        [BenchmarkExample("x", "reasoning", "Question?", "Answer", {})],
        MockModelClient({"Question?": "wrong because bad reasoning"}),
    )[0]
    synthetic = generate_synthetic_prompts(analyze_failures([result])["failures"], num_examples=3)
    assert len(synthetic) == 3
    record = json.loads(synthetic[0].to_jsonl())
    assert {"id", "task_type", "prompt", "gold_answer", "metadata"}.issubset(record)
    path = tmp_path / "synthetic.jsonl"
    write_synthetic_jsonl(synthetic, path)
    assert len(load_jsonl_benchmark(path)) == 3
