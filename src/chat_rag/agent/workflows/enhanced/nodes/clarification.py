import logging
import json
from typing import Dict, Any, Optional

from chat_rag.agent.clarification import ClarificationDetector
from ..state import AgentState

logger = logging.getLogger(__name__)


class ClarificationNode:
    """Handles ambiguity detection and clarification."""

    def __init__(
        self, detector: ClarificationDetector, prompts: Dict[str, str], enable_memory: bool = True, memory: Any = None
    ):
        self.detector = detector
        self.prompts = prompts
        self.enable_memory = enable_memory
        self.memory = memory

    async def check_ambiguity(self, state: AgentState) -> Dict:
        """Check if query needs clarification."""
        if not self.detector:
            return {"response": {**state.get("response", {}), "needs_clarification": False}}

        query = state["retrieval"].get("resolved_query") or state["retrieval"].get("original_query")

        # Get user context (profile info)
        user_context_str = None
        user_state = state.get("user", {})
        if user_state.get("profile"):
            # Pass full profile info to help ambiguity detection skip known fields
            user_context_str = json.dumps(user_state["profile"])

        # Get conversation context
        context_summary = None
        if self.enable_memory and self.memory:
            context_summary = self.memory.get_context_summary()

        try:
            result = await self.detector.analyze(
                query=query,
                user_country=user_context_str,
                conversation_context=context_summary,
                system_prompt_override=self.prompts.get("ambiguity_detection"),
            )

            logger.info(f"Clarification result: {result.needs_clarification}, type={result.ambiguity_type.value}")

            return {
                "response": {
                    **state.get("response", {}),
                    "needs_clarification": result.needs_clarification,
                    "clarifying_question": result.clarifying_question,
                }
            }
        except Exception as e:
            logger.error(f"Ambiguity detection failed: {e}")
            return {
                "response": {
                    **state.get("response", {}),
                    "needs_clarification": False,  # Fail open
                    "error": str(e),
                }
            }

    async def ask_clarification(self, state: AgentState) -> Dict:
        """Return clarifying question as response."""
        response_state = state.get("response", {})
        question = response_state.get("clarifying_question", "Could you please provide more details?")

        # Store in memory that we asked for clarification
        if self.enable_memory and self.memory:
            self.memory.context.pending_clarification = question

        return {"final_response": question}
