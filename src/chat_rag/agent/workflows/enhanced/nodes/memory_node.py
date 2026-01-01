import logging
from typing import Dict, Any

from ..state import AgentState

logger = logging.getLogger(__name__)


class MemoryNode:
    """Handles context resolution."""

    def __init__(self, memory: Any):
        self.memory = memory

    async def resolve_context(self, state: AgentState) -> Dict:
        """Resolve references using session memory."""
        original_query = state["messages"][-1].content

        resolved_query = original_query
        if self.memory and self.memory.has_context:
            try:
                resolved_query = self.memory.resolve_query(original_query)
            except Exception as e:
                logger.warning(f"Query resolution failed: {e}")

        return {
            "retrieval": {
                **state.get("retrieval", {}),
                "original_query": original_query,
                "resolved_query": resolved_query,
            }
        }
