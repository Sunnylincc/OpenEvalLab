"""Generate template-based synthetic prompts from observed errors."""

from openevallab.analysis import analyze_failures
from openevallab.benchmarks import load_jsonl_benchmark
from openevallab.evaluator import evaluate_benchmark
from openevallab.models import MockModelClient
from openevallab.synthesis import generate_synthetic_prompts

examples = load_jsonl_benchmark("data/sample_biomed_qa.jsonl")
model = MockModelClient(default_response="The answer is mitochondria because it powers cells.")
results = evaluate_benchmark(examples, model)
analysis = analyze_failures(results)

for example in generate_synthetic_prompts(analysis["failures"]):
    print(example)
