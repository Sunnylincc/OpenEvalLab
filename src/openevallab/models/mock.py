"""Deterministic model client for demos, tests, and first-run workflows."""

from __future__ import annotations

from collections.abc import Callable, Mapping
import re

from openevallab.models.base import BaseModelClient


class MockModelClient(BaseModelClient):
    """A deterministic local baseline that requires no API key.

    The client can be backed by exact prompt-to-answer mappings, a callback, or a
    small heuristic baseline used by the demo and examples.
    """

    def __init__(
        self,
        responses: Mapping[str, str] | None = None,
        *,
        default_response: str | None = None,
        response_fn: Callable[[str], str] | None = None,
        model_name: str = "mock",
    ) -> None:
        self.responses = dict(responses or {})
        self.default_response = default_response
        self.response_fn = response_fn
        self.model_name = model_name

    def generate(self, prompt: str) -> str:
        if self.response_fn is not None:
            return self.response_fn(prompt)
        if prompt in self.responses:
            return self.responses[prompt]
        if self.default_response is not None:
            return self.default_response
        return self._heuristic_response(prompt)

    def _heuristic_response(self, prompt: str) -> str:
        lowered = prompt.lower()
        if "train leaves" in lowered and "3 hours" in lowered:
            return "5 PM"
        if "bloops" in lowered and "lazzies" in lowered:
            return "yes"
        if "12 apples" in lowered and "gives away 5" in lowered:
            return "11"
        if "genetic information" in lowered:
            return "DNA"
        if "carrying oxygen" in lowered:
            return "red blood cells"
        if "produces insulin" in lowered:
            return "pancreas"
        numbers = [int(match) for match in re.findall(r"-?\d+", prompt)]
        if len(numbers) >= 2 and any(token in lowered for token in ["plus", "sum", "add"]):
            return str(sum(numbers))
        return "I do not know."
