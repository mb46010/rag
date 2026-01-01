import logging
from typing import Dict, Any
from ..state import AgentState

logger = logging.getLogger(__name__)


class CommonOptimizations:
    """Pre-canned responses for optimization."""

    async def handle_chitchat(self, state: AgentState) -> Dict:
        return {
            "final_response": (
                "Hello! I'm your HR assistant. I can help you with:\n"
                "• HR policies (WFH, expenses, leave, etc.)\n"
                "• Your PTO balance and employee info\n"
                "• Benefits and workplace questions\n\n"
                "What would you like to know?"
            )
        }

    async def handle_out_of_scope(self, state: AgentState) -> Dict:
        return {
            "final_response": (
                "I'm an HR assistant and can help with HR policies, benefits, "
                "time off, and workplace questions. For other topics, please "
                "reach out to the appropriate team or department."
            )
        }
