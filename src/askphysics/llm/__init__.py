"""LLM clients. The pipeline only depends on the ``LLMClient`` protocol.

``FermiClient`` lives in ``askphysics.llm.fermi_client`` and is not imported here: it
needs torch, and the website imports this package without it.
"""

from askphysics.llm.base import LLMClient, Roster, client_name
from askphysics.llm.fake import FakeLLMClient

__all__ = ["FakeLLMClient", "LLMClient", "Roster", "client_name"]
