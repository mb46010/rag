import logging
from typing import Dict, Any

from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI

from ..state import AgentState

logger = logging.getLogger(__name__)


class RoutingNode:
    """Handles intent classification."""

    def __init__(self, llm: ChatOpenAI, prompts: Dict[str, str]):
        self.llm = llm
        self.prompts = prompts

    async def classify_intent(self, state: AgentState) -> Dict:
        """Classify query intent."""
        query = state["retrieval"].get("resolved_query") or state["retrieval"].get("original_query")

        prompt_template = self.prompts.get("classification", "{query}")
        prompt = prompt_template.format(query=query)

        try:
            response = await self.llm.ainvoke([HumanMessage(content=prompt)])
            intent_raw = response.content.strip().lower().replace('"', "")

            valid_intents = ["policy_only", "personal_only", "hybrid"]
            intent = intent_raw if intent_raw in valid_intents else "hybrid"

            return {"response": {**state.get("response", {}), "intent": intent}}
        except Exception as e:
            logger.error(f"Intent classification failed: {e}")
            return {
                "response": {
                    **state.get("response", {}),
                    "intent": "hybrid",  # Default safe intent
                    "error": str(e),
                }
            }
