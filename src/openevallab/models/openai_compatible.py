"""OpenAI-compatible model client placeholder."""

from __future__ import annotations

import os

from openevallab.models.base import BaseModelClient


class OpenAICompatibleClient(BaseModelClient):
    """Adapter stub for OpenAI-compatible APIs.

    The toolkit is fully usable with ``MockModelClient``. This class validates the
    expected configuration and provides a clear extension point without making
    the base install depend on a network client library.
    """

    def __init__(self, *, model_name: str = "openai-compatible", api_key: str | None = None) -> None:
        self.model_name = model_name
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError(
                "OPENAI_API_KEY is not set. Use '--model mock' for local runs, or set OPENAI_API_KEY "
                "before using the OpenAI-compatible client."
            )

    def generate(self, prompt: str) -> str:
        raise NotImplementedError(
            "OpenAICompatibleClient is a configuration placeholder in this lightweight release. "
            "Subclass it or add an optional provider dependency to make network calls."
        )
