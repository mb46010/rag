import logging
import json
import asyncio
from typing import Dict, List, Optional, Any
from tenacity import retry, stop_after_attempt, wait_exponential

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langchain_mcp_adapters.client import MultiServerMCPClient

from chat_rag.agent.config import AgentConfig
from chat_rag.retriever import EnhancedHybridRetriever
from chat_rag.agent.clarification import ClarificationDetector
from chat_rag.agent.highlighting import SourceHighlighter, format_highlighted_sources
from chat_rag.agent.memory import SessionMemory

from .state import EnhancedAgentState

logger = logging.getLogger(__name__)


class EnhancedWorkflowNodes:
    """Implementations of enhanced workflow nodes."""

    def __init__(
        self,
        llm: ChatOpenAI,
        retriever: EnhancedHybridRetriever,
        config: AgentConfig,
        user_email: str,
        prompts: Dict[str, str],
        enable_hyde: bool,
        enable_clarification: bool,
        enable_highlighting: bool,
        enable_memory: bool,
        enable_cost_tracking: bool,
        clarification_detector: Optional[ClarificationDetector] = None,
        highlighter: Optional[SourceHighlighter] = None,
        memory: Optional[SessionMemory] = None,
        mcp_client: Optional[MultiServerMCPClient] = None,
    ):
        self.llm = llm
        self.retriever = retriever
        self.config = config
        self.user_email = user_email
        self.prompts = prompts

        # Feature flags
        self.enable_hyde = enable_hyde
        self.enable_clarification = enable_clarification
        self.enable_highlighting = enable_highlighting
        self.enable_memory = enable_memory
        self.enable_cost_tracking = enable_cost_tracking

        # Components
        self.clarification_detector = clarification_detector
        self.highlighter = highlighter
        self.memory = memory

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
                        if hasattr(item, "text"):
                            try:
                                data = json.loads(item.text)
                                return data
                            except (json.JSONDecodeError, TypeError):
                                continue
                        elif isinstance(item, dict):
                            return item

                    logger.warning(f"MCP tool {tool_name} returned a list but no valid JSON/dict found: {result}")
                    return {"error": "Invalid tool response format"}

                return result

        raise ValueError(f"Tool {tool_name} not found")

    async def resolve_context(self, state: EnhancedAgentState) -> Dict:
        """Resolve references using session memory."""
        original_query = state["messages"][-1].content

        if self.enable_memory and self.memory and self.memory.has_context:
            resolved_query = self.memory.resolve_query(original_query)
        else:
            resolved_query = original_query

        return {
            "original_query": original_query,
            "resolved_query": resolved_query,
        }

    async def fetch_profile(self, state: EnhancedAgentState) -> Dict:
        """Fetch employee profile via MCP early in the flow."""
        try:
            raw_profile = await self.call_mcp_tool("get_employee_profile_tool", {"email": self.user_email})

            # Robust extraction: MCP might return {"employee_profile": {"text": {...}}}
            profile = raw_profile
            if isinstance(raw_profile, dict) and "employee_profile" in raw_profile:
                inner = raw_profile["employee_profile"]
                if isinstance(inner, dict) and "text" in inner:
                    profile = inner["text"]
                else:
                    profile = inner

            logger.info(f"Early profile fetch success for {self.user_email}: {json.dumps(profile)}")
            return {"employee_profile": profile}
        except Exception as e:
            logger.error(f"Early profile fetch failed: {e}")
            return {"employee_profile": {"error": str(e)}}

    async def check_ambiguity(self, state: EnhancedAgentState) -> Dict:
        """Check if query needs clarification."""
        if not self.enable_clarification or not self.clarification_detector:
            return {"needs_clarification": False}

        query = state.get("resolved_query", state["messages"][-1].content)

        # Get user context (profile info)
        user_context = None
        if state.get("employee_profile"):
            # Pass full profile info to help ambiguity detection skip known fields
            profile = state["employee_profile"]
            user_context = json.dumps(profile)
            logger.info(f"Using profile context for ambiguity detection: {user_context}")

        # Get conversation context
        context = None
        if self.enable_memory and self.memory:
            context = self.memory.get_context_summary()

        result = await self.clarification_detector.analyze(
            query=query,
            user_country=user_context,  # Passed as 'country' in prompt, but contains full info now
            conversation_context=context,
            system_prompt_override=self.prompts.get("ambiguity_detection"),
        )

        logger.info(
            f"Clarification result: needs_clarification={result.needs_clarification}, type={result.ambiguity_type.value}"
        )

        if result.needs_clarification:
            logger.info(f"Clarification needed: {result.ambiguity_type.value}")

        return {
            "needs_clarification": result.needs_clarification,
            "clarifying_question": result.clarifying_question,
        }

    async def ask_clarification(self, state: EnhancedAgentState) -> Dict:
        """Return clarifying question as response."""
        question = state.get("clarifying_question", "Could you please provide more details?")

        # Store in memory that we asked for clarification
        if self.enable_memory and self.memory:
            self.memory.context.pending_clarification = question

        return {"final_response": question}

    async def classify_intent(self, state: EnhancedAgentState) -> Dict:
        """Classify query intent."""
        query = state.get("resolved_query", state["messages"][-1].content)

        prompt_template = self.prompts.get("classification", "")
        prompt = prompt_template.format(query=query)

        response = await self.llm.ainvoke([HumanMessage(content=prompt)])
        intent = response.content.strip().lower().replace('"', "")

        # Track tokens if enabled
        if self.enable_cost_tracking and hasattr(response, "usage_metadata"):
            # Note: actual token tracking would need callback integration
            pass

        return {"intent": intent if intent in ["policy_only", "personal_only", "hybrid"] else "hybrid"}

    async def gather_data(self, state: EnhancedAgentState) -> Dict:
        """Gather data based on intent (with HyDE if enabled)."""
        query = state.get("resolved_query", state["messages"][-1].content)
        intent = state.get("intent", "hybrid")

        results = {"hyde_used": False}

        # Fetch personal data if needed
        if intent in ["personal_only", "hybrid"]:
            # Profile might be already fetched by early fetch_profile node
            profile = state.get("employee_profile")
            if not profile or "error" in profile:
                try:
                    profile = await self.call_mcp_tool("get_employee_profile_tool", {"email": self.user_email})
                    results["employee_profile"] = profile
                    logger.info(f"Fetched profile for {self.user_email} (fallback in gather_data)")
                except Exception as e:
                    logger.error(f"Failed to fetch profile in gather_data: {e}")
                    results["employee_profile"] = {"error": str(e)}

            # Fetch PTO balance
            try:
                balance = await self.call_mcp_tool("get_time_off_balance_tool", {"email": self.user_email})
                results["time_off_balance"] = balance
                logger.info(f"Fetched PTO balance for {self.user_email}")
            except Exception as e:
                logger.error(f"Failed to fetch PTO: {e}")
                results["time_off_balance"] = {"error": str(e)}

        # Search policies if needed
        if intent in ["policy_only", "hybrid"]:
            filters = {"active": True}
            profile_for_filter = results.get("employee_profile") or state.get("employee_profile")
            if profile_for_filter and isinstance(profile_for_filter, dict):
                country = profile_for_filter.get("country")
                if country:
                    filters["country"] = country

            # Use HyDE-enhanced retrieval if available
            if self.enable_hyde and hasattr(self.retriever, "aretrieve"):
                chunks = await self.retriever.aretrieve(query, filters=filters, top_k=3)
                results["hyde_used"] = any(getattr(c, "hyde_enhanced", False) for c in chunks)
            else:
                chunks = self.retriever.retrieve(query, filters=filters, top_k=3)

            results["policy_results"] = [
                {
                    "text": c.text,
                    "source": f"{c.document_name} > {c.section_path_str}",
                    "confidence": c.confidence_level,
                    "score": c.score,
                    "chunk_id": c.chunk_id,
                }
                for c in chunks
            ]

            # Store chunks for highlighting
            results["_chunks"] = chunks

        return results

    async def synthesize(self, state: EnhancedAgentState) -> Dict:
        """Generate response."""
        query = state.get("resolved_query", state["messages"][-1].content)

        # Build context
        context_parts = []

        if state.get("employee_profile"):
            profile = state["employee_profile"]
            context_parts.append(f"Employee: {profile}")

        if state.get("time_off_balance"):
            context_parts.append(f"PTO Balance: {state['time_off_balance']}")

        if state.get("policy_results"):
            policy_text = "\n\n".join([f"[{p['source']}]\n{p['text']}" for p in state["policy_results"]])
            context_parts.append(f"Policies:\n{policy_text}")

        context = "\n\n---\n\n".join(context_parts) or "No context available."

        prompt_template = self.prompts.get("synthesis", "")
        prompt = prompt_template.format(context=context, query=query)

        response = await self.llm.ainvoke(
            [
                SystemMessage(content="You are a helpful HR assistant."),
                HumanMessage(content=prompt),
            ]
        )

        # Update memory
        if self.enable_memory and self.memory:
            entities = self.memory.extract_entities_from_query(query)
            topics = [p["source"].split(" > ")[0] for p in (state.get("policy_results") or [])]

            self.memory.add_user_message(query, topics=topics[:1], entities=entities)
            self.memory.add_assistant_message(
                response.content,
                topics=topics[:1],
                entities=entities,
                sources=[p["source"] for p in (state.get("policy_results") or [])],
            )

        return {"final_response": response.content}

    async def highlight_sources(self, state: EnhancedAgentState) -> Dict:
        """Add source highlighting to response."""
        if not self.enable_highlighting or not self.highlighter:
            return {}

        if not state.get("policy_results") or not state.get("_chunks"):
            return {}

        query = state.get("resolved_query", state["messages"][-1].content)
        answer = state.get("final_response", "")
        chunks = state.get("_chunks", [])

        highlighted = self.highlighter.highlight_sync(query, answer, chunks)
        formatted = format_highlighted_sources(highlighted)

        # Append to response
        if formatted:
            return {"highlighted_sources": formatted}

        return {}
