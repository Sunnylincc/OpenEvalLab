"""Base model client protocol."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class ModelResponse:
    """A normalized model response."""

    text: str
    raw: Any | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


class BaseModelClient(ABC):
    """Interface implemented by all model clients."""

    model_name: str

    @abstractmethod
    def generate(self, prompt: str, *, metadata: dict[str, Any] | None = None) -> ModelResponse:
        """Generate a response for one prompt."""
