import logging
from typing import Dict, Any, List

from langchain_core.messages import SystemMessage, HumanMessage
from langchain_openai import ChatOpenAI

from ..state import AgentState

logger = logging.getLogger(__name__)


class SynthesisNode:
    """Handles response generation."""

    def __init__(self, llm: ChatOpenAI, prompts: Dict[str, str], enable_memory: bool = True, memory: Any = None):
        self.llm = llm
        self.prompts = prompts
        self.enable_memory = enable_memory
        self.memory = memory

    async def synthesize(self, state: AgentState) -> Dict:
        """Generate final response based on gathered context."""
        query = state["retrieval"].get("resolved_query") or state["retrieval"].get("original_query")

        context = self._build_context(state)

        prompt_template = self.prompts.get("synthesis", "")
        # Fallback if prompt missing
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

            content = response.content

            # Memory update (side effect, could be separate node but keeping here for now)
            if self.enable_memory and self.memory:
                self._update_memory(state, query, content)

            return {"final_response": content}

        except Exception as e:
            logger.error(f"Synthesis failed: {e}")
            return {"error": f"Synthesis failed: {e}"}

    def _build_context(self, state: AgentState) -> str:
        context_parts = []

        user_state = state.get("user", {})
        if user_state.get("profile"):
            context_parts.append(f"Employee: {user_state['profile']}")

        if user_state.get("pto_balance"):
            context_parts.append(f"PTO Balance: {user_state['pto_balance']}")

        retrieval_state = state.get("retrieval", {})
        chunks = retrieval_state.get("chunks", [])
        if chunks:
            policy_text = "\n\n".join([f"[{c['source']}]\n{c['text']}" for c in chunks])
            context_parts.append(f"Policies:\n{policy_text}")

        return "\n\n---\n\n".join(context_parts) or "No context available."

    def _update_memory(self, state: AgentState, query: str, response: str):
        try:
            chunks = state.get("retrieval", {}).get("chunks", [])
            topics = [c["source"].split(" > ")[0] for c in chunks]
            sources = [c["source"] for c in chunks]

            # Simple entity extraction not available here directly unless we call helper or LLM
            # For now, just adding messages
            # Previous implementation used memory.extract_entities_from_query(query) which calls LLM
            # Just mimicking basic update

            self.memory.add_user_message(query, topics=topics[:1])
            self.memory.add_assistant_message(response, topics=topics[:1], sources=sources)
        except Exception as e:
            logger.warning(f"Failed to update memory: {e}")
