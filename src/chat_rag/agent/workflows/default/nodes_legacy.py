import logging
import asyncio
import json
import os
from typing import Dict, List, Optional, Any
from tenacity import retry, stop_after_attempt, wait_exponential

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langchain_mcp_adapters.client import MultiServerMCPClient

from chat_rag.agent.config import AgentConfig
from chat_rag.retriever import EnhancedHybridRetriever, RetrievalConfig, HyDERetriever
from .state import AgentState, QueryIntent

logger = logging.getLogger(__name__)


class WorkflowNodes:
    """Implementations of workflow nodes."""

    def __init__(
        self,
        llm: ChatOpenAI,
        retriever: EnhancedHybridRetriever,
        config: AgentConfig,
        user_email: str,
        prompts: Dict[str, str],
        mcp_client: Optional[MultiServerMCPClient] = None,
    ):
        self.llm = llm
        self.retriever = retriever
        self.config = config
        self.user_email = user_email
        self.prompts = prompts
        self.mcp_client = mcp_client
        self._mcp_tools = None

    async def _get_mcp_client(self) -> MultiServerMCPClient:
        """Lazily initialize MCP client."""
        if self.mcp_client is None:
            self.mcp_client = MultiServerMCPClient(
                {
                    "hr_services": {
                        "url": f"{self.config.mcp_url}/sse",
                        "transport": "sse",
                    }
                }
            )
            self._mcp_tools = await self.mcp_client.get_tools()
            logger.info(f"Initialized MCP client with {len(self._mcp_tools)} tools")
        elif self._mcp_tools is None:
            self._mcp_tools = await self.mcp_client.get_tools()
        return self.mcp_client

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
    )
    async def call_mcp_tool(self, tool_name: str, args: Dict) -> Dict:
        """Call MCP tool with retry logic."""
        await self._get_mcp_client()

        for tool in self._mcp_tools:
            if tool.name == tool_name:
                result = await tool.ainvoke(args)

                # Handle MCP list response (e.g. [TextContent(text='{...}')])
                if isinstance(result, list):
                    for item in result:
                        # Case 1: Item is an object with a 'text' attribute (TextContent)
                        if hasattr(item, "text"):
                            try:
                                import json

                                return json.loads(item.text)
                            except (json.JSONDecodeError, TypeError):
                                continue
                        # Case 2: Item is already a dict
                        elif isinstance(item, dict):
                            return item

                    # If we got a list but couldn't extracting anything, return empty or error
                    logger.warning(f"MCP tool {tool_name} returned a list but no valid JSON/dict found: {result}")
                    return {"error": "Invalid tool response format"}

                return result

        raise ValueError(f"Tool {tool_name} not found")

    async def classify_intent(self, state: AgentState) -> Dict:
        """Classify query intent for routing."""
        query = state["messages"][-1].content

        prompt_template = self.prompts.get("classification", "")
        prompt = prompt_template.format(query=query)
        response = await self.llm.ainvoke([HumanMessage(content=prompt)])

        intent_str = response.content.strip().lower().replace('"', "")

        # Validate intent
        try:
            intent = QueryIntent(intent_str)
        except ValueError:
            logger.warning(f"Unknown intent '{intent_str}', defaulting to hybrid")
            intent = QueryIntent.HYBRID

        logger.info(f"Classified intent: {intent.value}")
        return {"intent": intent.value}

    async def handle_chitchat(self, state: AgentState) -> Dict:
        """Handle casual conversation."""
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
        """Handle out-of-scope queries."""
        return {
            "final_response": (
                "I'm an HR assistant and can help with HR policies, benefits, "
                "time off, and workplace questions. For other topics, please "
                "reach out to the appropriate team or department."
            )
        }

    async def fetch_personal_data(self, state: AgentState) -> Dict:
        """Fetch employee profile and PTO balance via MCP."""
        results = {}

        # Fetch profile
        try:
            profile = await self.call_mcp_tool("get_employee_profile_tool", {"email": self.user_email})
            results["employee_profile"] = profile
            logger.info(f"Fetched profile for {self.user_email}")
        except Exception as e:
            logger.error(f"Failed to fetch profile: {e}")
            results["employee_profile"] = {"error": str(e)}

        # Fetch PTO balance
        try:
            balance = await self.call_mcp_tool("get_time_off_balance_tool", {"email": self.user_email})
            results["time_off_balance"] = balance
            logger.info(f"Fetched PTO balance for {self.user_email}")
        except Exception as e:
            logger.error(f"Failed to fetch PTO: {e}")
            results["time_off_balance"] = {"error": str(e)}

        return results

    async def search_policies(self, state: AgentState) -> Dict:
        """Search policy documents using RAG."""
        query = state["messages"][-1].content

        # Use employee country if available
        filters = {"active": True}
        if state.get("employee_profile") and isinstance(state["employee_profile"], dict):
            country = state["employee_profile"].get("country")
            if country:
                filters["country"] = country

        try:
            # Use aretrieve to support HyDE (which requires async for LLM calls)
            # EnhancedHybridRetriever also has an aretrieve wrapper for compatibility.
            chunks = await self.retriever.aretrieve(query, filters=filters, top_k=3)

            policy_results = [
                {
                    "text": c.text,
                    "source": f"{c.document_name} > {c.section_path_str}",
                    "confidence": c.confidence_level,
                    "score": c.score,
                }
                for c in chunks
            ]

            logger.info(f"Retrieved {len(policy_results)} policy chunks")
            return {"policy_results": policy_results}

        except Exception as e:
            logger.error(f"Policy search failed: {e}")
            return {"policy_results": [], "error": str(e)}

    async def parallel_fetch(self, state: AgentState) -> Dict:
        """Fetch personal data and policies in parallel."""
        if not self.config.enable_parallel_fetch:
            # Sequential fallback
            personal_result = await self.fetch_personal_data(state)
            state_with_profile = {**state, **personal_result}
            policy_result = await self.search_policies(state_with_profile)
            return {**personal_result, **policy_result}

        # Parallel execution
        # First get profile to use country filter for policies
        try:
            profile = await self.call_mcp_tool("get_employee_profile_tool", {"email": self.user_email})
        except Exception as e:
            profile = {"error": str(e)}

        # Now run PTO and policy search in parallel
        state_with_profile = {**state, "employee_profile": profile}

        async def fetch_pto():
            try:
                return await self.call_mcp_tool("get_time_off_balance_tool", {"email": self.user_email})
            except Exception as e:
                return {"error": str(e)}

        async def fetch_policies():
            return await self.search_policies(state_with_profile)

        pto_task = asyncio.create_task(fetch_pto())
        policy_task = asyncio.create_task(fetch_policies())

        pto_result, policy_result = await asyncio.gather(pto_task, policy_task, return_exceptions=True)

        # Handle exceptions
        if isinstance(pto_result, Exception):
            pto_result = {"error": str(pto_result)}
        if isinstance(policy_result, Exception):
            policy_result = {"policy_results": [], "error": str(policy_result)}

        return {
            "employee_profile": profile,
            "time_off_balance": pto_result,
            **policy_result,
        }

    async def synthesize_response(self, state: AgentState) -> Dict:
        """Generate final response from gathered context."""
        query = state["messages"][-1].content

        # Build context sections
        context_parts = []

        # Employee profile
        if state.get("employee_profile") and "error" not in state["employee_profile"]:
            profile = state["employee_profile"]
            context_parts.append(
                "**Employee Profile:**\n"
                f"• Name: {profile.get('name', 'N/A')} {profile.get('surname', '')}\n"
                f"• Role: {profile.get('role', 'N/A')}\n"
                f"• Country: {profile.get('country', 'N/A')}\n"
                f"• Tenure: {profile.get('tenure', 'N/A')}\n"
                f"• Band: {profile.get('band', 'N/A')}"
            )

        # Time off balance
        if state.get("time_off_balance") and "error" not in state["time_off_balance"]:
            balance = state["time_off_balance"]
            context_parts.append(
                f"**Time Off Balance:**\n• Days remaining: {balance.get('number_of_days_left', 'N/A')}"
            )

        # Policy results
        if state.get("policy_results"):
            policy_text = "\n\n".join(
                [f"[{p['source']}] (confidence: {p['confidence']})\n{p['text']}" for p in state["policy_results"]]
            )
            context_parts.append(f"**Relevant Policies:**\n{policy_text}")

        context = "\n\n---\n\n".join(context_parts) if context_parts else "No context available."

        # Generate response
        prompt_template = self.prompts.get("synthesis", "")
        prompt = prompt_template.format(context=context, query=query)

        response = await self.llm.ainvoke(
            [
                SystemMessage(content="You are a helpful HR assistant."),
                HumanMessage(content=prompt),
            ]
        )

        return {"final_response": response.content}
