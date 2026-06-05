"""Base model client interface."""

from __future__ import annotations

from abc import ABC, abstractmethod


class BaseModelClient(ABC):
    """Small interface implemented by all model clients."""

    model_name: str

    @abstractmethod
    def generate(self, prompt: str) -> str:
        """Generate a response for one prompt."""
