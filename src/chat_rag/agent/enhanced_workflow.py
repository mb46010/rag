"""Enhanced LangGraph Workflow Agent with production features.

Features:
- HyDE for improved retrieval
- Clarifying questions for ambiguous queries
- Source highlighting for citations
- Multi-turn session memory
- Cost tracking
"""

import logging
import os
import uuid
import asyncio
import json
from typing import TypedDict, Annotated, Optional, List, Dict, Any

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langchain_mcp_adapters.client import MultiServerMCPClient
from tenacity import retry, stop_after_attempt, wait_exponential

from chat_rag.retriever import EnhancedHybridRetriever, RetrievalConfig
from chat_rag.retriever.hyde import HyDERetriever
from chat_rag.observability import CostTracker, QueryMetrics
from langfuse.langchain import CallbackHandler
from langfuse import observe

from .config import AgentConfig
from .clarification import ClarificationDetector, ClarificationResult
from .highlighting import SourceHighlighter, format_highlighted_sources
from .memory import SessionMemory

logger = logging.getLogger(__name__)


class EnhancedAgentState(TypedDict):
    """State with additional fields for new features."""

    messages: Annotated[list, add_messages]
    user_email: str
    query_id: str

    # Original and resolved query
    original_query: str
    resolved_query: str

    # Intent and routing
    intent: Optional[str]
    needs_clarification: bool
    clarifying_question: Optional[str]

    # Data gathered
    employee_profile: Optional[Dict]
    time_off_balance: Optional[Dict]
    policy_results: Optional[List[Dict]]

    # Enhanced features
    highlighted_sources: Optional[str]
    hyde_used: bool

    # Output
    final_response: Optional[str]
    error: Optional[str]


class EnhancedHRWorkflowAgent:
    """Production-ready HR workflow agent with:

    1. **HyDE**: Generates hypothetical answers for better retrieval on vague queries
    2. **Clarifying Questions**: Detects ambiguity and asks for clarification
    3. **Source Highlighting**: Shows exact spans supporting the answer
    4. **Session Memory**: Handles follow-ups and reference resolution
    5. **Cost Tracking**: Monitors token usage and latency

    Flow:
    ```
    User Query
        │
        ▼
    ┌─────────────────┐
    │ Resolve Context │ ← Session Memory
    └────────┬────────┘
             │
             ▼
    ┌─────────────────┐
    │ Check Ambiguity │
    └────────┬────────┘
             │
        ┌────┴────┐
        │ Ambiguous?
        │    │
        ▼    ▼
    [Ask]  [Continue]
             │
             ▼
    ┌─────────────────┐
    │ Classify Intent │
    └────────┬────────┘
             │
             ▼
    ┌─────────────────┐
    │ Gather Data     │ ← HyDE + RAG + MCP
    └────────┬────────┘
             │
             ▼
    ┌─────────────────┐
    │ Synthesize      │
    └────────┬────────┘
             │
             ▼
    ┌─────────────────┐
    │ Highlight       │ ← Source spans
    └────────┬────────┘
             │
             ▼
    Response + Citations
    ```
    """  # noqa: D415

    def __init__(
        self,
        user_email: str,
        retriever: EnhancedHybridRetriever = None,
        config: AgentConfig = None,
        enable_hyde: bool = True,
        enable_clarification: bool = True,
        enable_highlighting: bool = True,
        enable_memory: bool = True,
        enable_cost_tracking: bool = True,
    ):
        """Initialize the enhanced HR workflow agent."""
        self.user_email = user_email
        self.config = config or AgentConfig()

        # Feature flags
        self.enable_hyde = enable_hyde
        self.enable_clarification = enable_clarification
        self.enable_highlighting = enable_highlighting
        self.enable_memory = enable_memory
        self.enable_cost_tracking = enable_cost_tracking

        # Initialize LLM
        self.llm = ChatOpenAI(
            model=self.config.model_name,
            temperature=self.config.temperature,
        )

        # Initialize retriever
        if retriever:
            self.retriever = retriever
            self.owns_retriever = False
        else:
            self.retriever = EnhancedHybridRetriever(RetrievalConfig(enable_reranking=True))
            self.owns_retriever = True

        if enable_hyde:
            self.retriever = HyDERetriever(
                base_retriever=self.retriever,
                llm=ChatOpenAI(model="gpt-4o-mini", temperature=0.7),
            )
        else:
            # Already initialized in self.retriever above
            pass

        # Initialize feature modules
        if enable_clarification:
            self.clarification_detector = ClarificationDetector()

        if enable_highlighting:
            self.highlighter = SourceHighlighter(use_llm=False)  # Heuristic for speed

        if enable_memory:
            self.memory = SessionMemory(max_turns=10)

        if enable_cost_tracking:
            self.cost_tracker = CostTracker()

        # MCP client (lazy init)
        self._mcp_client = None
        self._mcp_tools = None

        # Load prompts
        self.prompts = self._load_prompts()

        # Build graph
        self.graph = self._build_graph()

        logger.info(f"EnhancedHRWorkflowAgent initialized for {user_email}")
        logger.info(
            f"Features: HyDE={enable_hyde}, Clarification={enable_clarification}, "
            f"Highlighting={enable_highlighting}, Memory={enable_memory}"
        )

    def _load_prompts(self) -> Dict[str, str]:
        """Load prompts from the enhanced_workflow prompts directory."""
        prompts = {}
        base_path = os.path.dirname(__file__)
        prompts_dir = os.path.join(base_path, "prompts", "enhanced_workflow")

        for name in ["classification", "synthesis", "ambiguity_detection"]:
            path = os.path.join(prompts_dir, f"{name}.json")
            try:
                with open(path, "r") as f:
                    data = json.load(f)
                    prompts[name] = data["prompt"]
                    logger.info(f"Loaded enhanced workflow prompt '{name}' (version {data.get('version', 'unknown')})")
            except Exception as e:
                logger.error(f"Failed to load prompt '{name}' from {path}: {e}")
                prompts[name] = ""

        return prompts

    async def _get_mcp_client(self) -> MultiServerMCPClient:
        """Lazily initialize MCP client."""
        if self._mcp_client is None:
            self._mcp_client = MultiServerMCPClient(
                {
                    "hr_services": {
                        "url": f"{self.config.mcp_url}/sse",
                        "transport": "sse",
                    }
                }
            )
            self._mcp_tools = await self._mcp_client.get_tools()
            logger.info(f"Initialized MCP client with {len(self._mcp_tools)} tools")
        return self._mcp_client

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
    )
    async def _call_mcp_tool(self, tool_name: str, args: Dict) -> Dict:
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
                                # If it's a dict with a single key that matches the tool's data type,
                                # we might want to flatten it, but for now let's just return the dict.
                                # The node implementation will handle specific flattening.
                                return data
                            except (json.JSONDecodeError, TypeError):
                                continue
                        elif isinstance(item, dict):
                            return item

                    logger.warning(f"MCP tool {tool_name} returned a list but no valid JSON/dict found: {result}")
                    return {"error": "Invalid tool response format"}

                return result

        raise ValueError(f"Tool {tool_name} not found")

    async def _fetch_profile(self, state: EnhancedAgentState) -> Dict:
        """Fetch employee profile via MCP early in the flow."""
        try:
            raw_profile = await self._call_mcp_tool("get_employee_profile_tool", {"email": self.user_email})

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

    def _build_graph(self) -> StateGraph:
        """Build enhanced workflow graph."""
        graph = StateGraph(EnhancedAgentState)

        # Nodes
        graph.add_node("resolve_context", self._resolve_context)
        graph.add_node("fetch_profile", self._fetch_profile)
        graph.add_node("check_ambiguity", self._check_ambiguity)
        graph.add_node("ask_clarification", self._ask_clarification)
        graph.add_node("classify_intent", self._classify_intent)
        graph.add_node("gather_data", self._gather_data)
        graph.add_node("synthesize", self._synthesize)
        graph.add_node("highlight_sources", self._highlight_sources)

        # Edges
        graph.add_edge(START, "resolve_context")
        graph.add_edge("resolve_context", "fetch_profile")
        graph.add_edge("fetch_profile", "check_ambiguity")

        # Conditional: ambiguous → ask, else continue
        graph.add_conditional_edges(
            "check_ambiguity",
            lambda s: "ask" if s.get("needs_clarification") else "continue",
            {
                "ask": "ask_clarification",
                "continue": "classify_intent",
            },
        )

        graph.add_edge("ask_clarification", END)
        graph.add_edge("classify_intent", "gather_data")
        graph.add_edge("gather_data", "synthesize")
        graph.add_edge("synthesize", "highlight_sources")
        graph.add_edge("highlight_sources", END)

        return graph.compile()

    # ─────────────────────────────────────────────────────────────
    # Node implementations
    # ─────────────────────────────────────────────────────────────

    async def _resolve_context(self, state: EnhancedAgentState) -> Dict:
        """Resolve references using session memory."""
        original_query = state["messages"][-1].content

        if self.enable_memory and self.memory.has_context:
            resolved_query = self.memory.resolve_query(original_query)
        else:
            resolved_query = original_query

        return {
            "original_query": original_query,
            "resolved_query": resolved_query,
        }

    async def _check_ambiguity(self, state: EnhancedAgentState) -> Dict:
        """Check if query needs clarification."""
        if not self.enable_clarification:
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
        if self.enable_memory:
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

    async def _ask_clarification(self, state: EnhancedAgentState) -> Dict:
        """Return clarifying question as response."""
        question = state.get("clarifying_question", "Could you please provide more details?")

        # Store in memory that we asked for clarification
        if self.enable_memory:
            self.memory.context.pending_clarification = question

        return {"final_response": question}

    async def _classify_intent(self, state: EnhancedAgentState) -> Dict:
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

    async def _gather_data(self, state: EnhancedAgentState) -> Dict:
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
                    profile = await self._call_mcp_tool("get_employee_profile_tool", {"email": self.user_email})
                    results["employee_profile"] = profile
                    logger.info(f"Fetched profile for {self.user_email} (fallback in gather_data)")
                except Exception as e:
                    logger.error(f"Failed to fetch profile in gather_data: {e}")
                    results["employee_profile"] = {"error": str(e)}
            else:
                # Already have it, but might need to ensure it's in results for synthesis node if it expects it there
                # Actually synthesis node uses state.get("employee_profile"), so it's fine.
                pass

            # Fetch PTO balance
            try:
                balance = await self._call_mcp_tool("get_time_off_balance_tool", {"email": self.user_email})
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

    async def _synthesize(self, state: EnhancedAgentState) -> Dict:
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
        if self.enable_memory:
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

    async def _highlight_sources(self, state: EnhancedAgentState) -> Dict:
        """Add source highlighting to response."""
        if not self.enable_highlighting:
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

    # ─────────────────────────────────────────────────────────────
    # Public interface
    # ─────────────────────────────────────────────────────────────

    @observe(as_type="agent")
    async def arun(self, message: str) -> Dict[str, Any]:
        """Execute workflow with full tracking."""
        query_id = str(uuid.uuid4())[:8]

        initial_state: EnhancedAgentState = {
            "messages": [HumanMessage(content=message)],
            "user_email": self.user_email,
            "query_id": query_id,
            "original_query": message,
            "resolved_query": message,
            "intent": None,
            "needs_clarification": False,
            "clarifying_question": None,
            "employee_profile": None,
            "time_off_balance": None,
            "policy_results": None,
            "highlighted_sources": None,
            "hyde_used": False,
            "final_response": None,
            "error": None,
        }

        # Initialize Langfuse handler
        langfuse_handler = CallbackHandler()

        # Track cost if enabled
        if self.enable_cost_tracking:
            with self.cost_tracker.track_query(query_id, self.user_email, message) as metrics:
                result = await self.graph.ainvoke(initial_state, config={"callbacks": [langfuse_handler]})
                metrics.hyde_used = result.get("hyde_used", False)
                metrics.chunks_retrieved = len(result.get("policy_results") or [])
        else:
            result = await self.graph.ainvoke(initial_state, config={"callbacks": [langfuse_handler]})

        # Build response
        output = result.get("final_response", "Unable to generate response.")

        # Append highlighted sources if available
        if result.get("highlighted_sources"):
            output += result["highlighted_sources"]

        return {
            "output": output,
            "query_id": query_id,
            "intent": result.get("intent"),
            "hyde_used": result.get("hyde_used", False),
            "clarification_requested": result.get("needs_clarification", False),
            "sources_count": len(result.get("policy_results") or []),
        }

    @observe(as_type="agent")
    async def astream(self, message: str):
        """
        Stream workflow execution with progress updates.

        Yields:
            Dict with type ('progress' or 'complete') and data
        """
        query_id = str(uuid.uuid4())[:8]

        initial_state: EnhancedAgentState = {
            "messages": [HumanMessage(content=message)],
            "user_email": self.user_email,
            "query_id": query_id,
            "original_query": message,
            "resolved_query": message,
            "intent": None,
            "needs_clarification": False,
            "clarifying_question": None,
            "employee_profile": None,
            "time_off_balance": None,
            "policy_results": None,
            "highlighted_sources": None,
            "hyde_used": False,
            "final_response": None,
            "error": None,
        }

        step_names = {
            "resolve_context": "🧠 Resolving context...",
            "check_ambiguity": "🔍 Checking for ambiguity...",
            "ask_clarification": "💬 Asking for clarification...",
            "classify_intent": "🏷️ Classifying intent...",
            "gather_data": "📊 Gathering information...",
            "synthesize": "✍️ Preparing response...",
            "highlight_sources": "📽️ Highlighting sources...",
        }

        # Initialize Langfuse handler
        langfuse_handler = CallbackHandler()

        node_output = {}

        # Determine if we should track cost
        metrics_context = (
            self.cost_tracker.track_query(query_id, self.user_email, message) if self.enable_cost_tracking else None
        )

        try:
            # We use a dummy context if tracking is disabled
            class DummyMetrics:
                hyde_used = False
                chunks_retrieved = 0

                def __enter__(self):
                    return self

                def __exit__(self, *args):
                    pass

            context = metrics_context or DummyMetrics()

            with context as metrics:
                async for event in self.graph.astream(
                    initial_state, stream_mode="updates", config={"callbacks": [langfuse_handler]}
                ):
                    node_name = list(event.keys())[0]
                    node_output = event[node_name] or {}

                    if node_name == "gather_data" and not isinstance(metrics, DummyMetrics):
                        metrics.hyde_used = node_output.get("hyde_used", False)
                        metrics.chunks_retrieved = len(node_output.get("policy_results") or [])

                    yield {
                        "type": "progress",
                        "step": node_name,
                        "message": step_names.get(node_name, f"Processing: {node_name}"),
                        "data": node_output,
                    }
        except Exception as e:
            logger.error(f"astream error: {e}", exc_info=True)
            node_output = {"error": str(e), "final_response": f"Error: {str(e)}"}

        # Final response
        final_response = node_output.get("final_response", "Unable to generate response.")
        if node_output.get("highlighted_sources"):
            final_response += node_output["highlighted_sources"]

        yield {"type": "complete", "response": final_response}

    def get_cost_summary(self) -> Dict:
        """Get cost tracking summary."""
        if self.enable_cost_tracking:
            return self.cost_tracker.get_summary()
        return {"tracking_enabled": False}

    async def close(self):
        """Clean up resources."""
        if self.owns_retriever and hasattr(self.retriever, "close"):
            self.retriever.close()
            logger.info("Closed agent-owned retriever")

        if self.enable_cost_tracking:
            logger.info(f"Session cost summary: {self.get_cost_summary()}")
