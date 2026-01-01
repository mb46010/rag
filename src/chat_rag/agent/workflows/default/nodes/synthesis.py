import logging
from typing import Dict, Any

from langchain_core.messages import SystemMessage, HumanMessage
from langchain_openai import ChatOpenAI

from ..state import AgentState

logger = logging.getLogger(__name__)


class SynthesisNode:
    """Handles response generation."""

    def __init__(self, llm: ChatOpenAI, prompts: Dict[str, str]):
        self.llm = llm
        self.prompts = prompts

    async def synthesize_response(self, state: AgentState) -> Dict:
        """Generate final response."""
        query = state["messages"][-1].content
        context = self._build_context(state)

        prompt_template = self.prompts.get("synthesis", "")
        # Fallback
        if not prompt_template:
            prompt_template = "Context: {context}\n\nQuery: {query}\n\nAnswer:"

        prompt = prompt_template.format(context=context, query=query)

        try:
            response = await self.llm.ainvoke(
                [
                    SystemMessage(content="You are a helpful HR assistant."),
                    HumanMessage(content=prompt),
                ]
            )
            return {"final_response": response.content}
        except Exception as e:
            logger.error(f"Synthesis failed: {e}")
            return {"error": f"Synthesis failed: {e}"}

    def _build_context(self, state: AgentState) -> str:
        context_parts = []
        user_state = state.get("user", {})

        # Profile
        profile = user_state.get("profile")
        if profile and isinstance(profile, dict) and "error" not in profile:
            context_parts.append(
                "**Employee Profile:**\n"
                f"• Name: {profile.get('name', 'N/A')} {profile.get('surname', '')}\n"
                f"• Role: {profile.get('role', 'N/A')}\n"
                f"• Country: {profile.get('country', 'N/A')}\n"
                f"• Tenure: {profile.get('tenure', 'N/A')}\n"
                f"• Band: {profile.get('band', 'N/A')}"
            )

        # PTO
        pto = user_state.get("pto_balance")
        if pto and isinstance(pto, dict) and "error" not in pto:
            context_parts.append(f"**Time Off Balance:**\n• Days remaining: {pto.get('number_of_days_left', 'N/A')}")

        # Policies
        retrieval_state = state.get("retrieval", {})
        policies = retrieval_state.get("policy_results", [])
        if policies:
            policy_text = "\n\n".join(
                [f"[{p['source']}] (confidence: {p['confidence']})\n{p['text']}" for p in policies]
            )
            context_parts.append(f"**Relevant Policies:**\n{policy_text}")

        return "\n\n---\n\n".join(context_parts) if context_parts else "No context available."
