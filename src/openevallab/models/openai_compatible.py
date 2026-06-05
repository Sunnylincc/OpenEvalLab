"""Placeholder for OpenAI-compatible chat completion clients."""

from __future__ import annotations

from openevallab.models.base import BaseModelClient, ModelResponse


class OpenAICompatibleClient(BaseModelClient):
    """Minimal adapter for OpenAI-compatible APIs.

    The first OpenEvalLab release keeps networked model execution explicit: instantiate this
    class with an API base URL and key, then replace ``generate`` with a provider-specific call
    or install the optional ``openai`` dependency and extend this adapter.
    """

    def __init__(self, *, model_name: str, api_key: str | None = None, base_url: str | None = None) -> None:
        self.model_name = model_name
        self.api_key = api_key
        self.base_url = base_url

    def generate(self, prompt: str, *, metadata: dict | None = None) -> ModelResponse:
        raise NotImplementedError(
            "OpenAICompatibleClient is a placeholder. Use MockModelClient for local runs or "
            "subclass this adapter to call your OpenAI-compatible provider."
        )
