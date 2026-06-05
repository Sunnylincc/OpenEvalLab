"""Failure-driven synthetic data generation."""

from openevallab.synthesis.generator import (
    SyntheticExample,
    generate_synthetic_prompts,
    write_synthetic_jsonl,
)

__all__ = ["SyntheticExample", "generate_synthetic_prompts", "write_synthetic_jsonl"]
