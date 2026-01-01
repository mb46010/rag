import logging
from typing import Dict, Any

from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI

from ..state import AgentState, QueryIntent

logger = logging.getLogger(__name__)


class RoutingNode:
    """Handles intent classification."""

    def __init__(self, llm: ChatOpenAI, prompts: Dict[str, str]):
        self.llm = llm
        self.prompts = prompts

    async def classify_intent(self, state: AgentState) -> Dict:
        """Classify query intent."""
        query = state["messages"][-1].content

        prompt_template = self.prompts.get("classification", "")
        prompt = prompt_template.format(query=query)

        try:
            response = await self.llm.ainvoke([HumanMessage(content=prompt)])
            intent_str = response.content.strip().lower().replace('"', "")

            try:
                intent = QueryIntent(intent_str)
            except ValueError:
                logger.warning(f"Unknown intent '{intent_str}', defaulting to hybrid")
                intent = QueryIntent.HYBRID

            return {"response": {"intent": intent.value}}
        except Exception as e:
            logger.error(f"Intent classification failed: {e}")
            return {"response": {"intent": "hybrid"}, "error": f"Classification failed: {e}"}
