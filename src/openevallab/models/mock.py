"""Deterministic model client for demos and tests."""

from __future__ import annotations

from collections.abc import Callable, Mapping

from openevallab.models.base import BaseModelClient, ModelResponse


class MockModelClient(BaseModelClient):
    """A small deterministic client backed by fixed responses or a callback."""

    def __init__(
        self,
        responses: Mapping[str, str] | None = None,
        *,
        default_response: str = "I do not know.",
        response_fn: Callable[[str], str] | None = None,
        model_name: str = "mock-model",
    ) -> None:
        self.responses = dict(responses or {})
        self.default_response = default_response
        self.response_fn = response_fn
        self.model_name = model_name

    def generate(self, prompt: str, *, metadata: dict | None = None) -> ModelResponse:
        if self.response_fn is not None:
            text = self.response_fn(prompt)
        else:
            text = self.responses.get(prompt, self.default_response)
        return ModelResponse(text=text, metadata={"client": "mock", "input_metadata": metadata or {}})
