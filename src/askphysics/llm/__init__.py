"""LLM clients. The pipeline only depends on the ``LLMClient`` protocol."""

from askphysics.llm.base import LLMClient
from askphysics.llm.fake import FakeLLMClient

__all__ = ["FakeLLMClient", "LLMClient"]
