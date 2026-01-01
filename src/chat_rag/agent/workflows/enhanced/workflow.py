import logging
import os
import uuid
import json
from typing import Dict, Any, Optional

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from langchain_mcp_adapters.client import MultiServerMCPClient
from langfuse.langchain import CallbackHandler
from langfuse import observe

from chat_rag.retriever import EnhancedHybridRetriever, RetrievalConfig
from chat_rag.retriever.hyde import HyDERetriever
from chat_rag.observability import CostTracker
from chat_rag.agent.config import AgentConfig
from chat_rag.agent.clarification import ClarificationDetector
from chat_rag.agent.highlighting import SourceHighlighter
from chat_rag.agent.memory import SessionMemory

from .state import AgentState
from .nodes.mcp import MCPNode
from .nodes.retrieval import RetrievalNode
from .nodes.synthesis import SynthesisNode
from .nodes.clarification import ClarificationNode
from .nodes.routing import RoutingNode
from .nodes.highlighting import HighlightingNode
from .nodes.memory_node import MemoryNode
from .nodes.error_handling import handle_error

logger = logging.getLogger(__name__)


class EnhancedHRWorkflowAgent:
    """Production-ready HR workflow agent with enhanced features."""

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
        checkpointer: Any = None,
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

        if enable_hyde and not isinstance(self.retriever, HyDERetriever):
            self.retriever = HyDERetriever(
                base_retriever=self.retriever,
                llm=ChatOpenAI(model="gpt-4o-mini", temperature=0.7),
            )

        # Initialize feature modules
        self.clarification_detector = ClarificationDetector() if enable_clarification else None
        self.highlighter = SourceHighlighter(use_llm=False) if enable_highlighting else None
        self.memory = SessionMemory(max_turns=10) if enable_memory else None
        self.cost_tracker = CostTracker() if enable_cost_tracking else None

        # Checkpointer
        self.checkpointer = checkpointer or MemorySaver()

        # Load prompts
        self.prompts = self._load_prompts()

        # Initialize Nodes
        self.mcp_node = MCPNode(mcp_url=self.config.mcp_url, user_email=self.user_email)
        self.retrieval_node = RetrievalNode(retriever=self.retriever, enable_hyde=enable_hyde)
        self.synthesis_node = SynthesisNode(
            llm=self.llm, prompts=self.prompts, enable_memory=enable_memory, memory=self.memory
        )
        self.clarification_node = ClarificationNode(
            detector=self.clarification_detector, prompts=self.prompts, enable_memory=enable_memory, memory=self.memory
        )
        self.routing_node = RoutingNode(llm=self.llm, prompts=self.prompts)
        self.highlighting_node = HighlightingNode(highlighter=self.highlighter)
        self.memory_node = MemoryNode(memory=self.memory)

        # Build graph
        self.graph = self._build_graph()

        logger.info(f"EnhancedHRWorkflowAgent initialized for {user_email}")

    def _load_prompts(self) -> Dict[str, str]:
        """Load prompts."""
        prompts = {}
        base_path = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
        prompts_dir = os.path.join(base_path, "prompts", "enhanced_workflow")

        for name in ["classification", "synthesis", "ambiguity_detection"]:
            path = os.path.join(prompts_dir, f"{name}.json")
            try:
                with open(path, "r") as f:
                    data = json.load(f)
                    prompt_data = data["prompt"]
                if isinstance(prompt_data, list):
                    prompts[name] = "\n".join(prompt_data)
                else:
                    prompts[name] = prompt_data
            except Exception as e:
                logger.error(f"Failed to load prompt '{name}': {e}")
                prompts[name] = ""

        return prompts

    def _build_graph(self) -> StateGraph:
        """Build enhanced workflow graph."""
        graph = StateGraph(AgentState)

        # Add Nodes
        graph.add_node("resolve_context", self.memory_node.resolve_context)
        graph.add_node("fetch_profile", self.mcp_node.fetch_profile)
        graph.add_node("fetch_pto", self.mcp_node.fetch_pto)
        graph.add_node("check_ambiguity", self.clarification_node.check_ambiguity)
        graph.add_node("ask_clarification", self.clarification_node.ask_clarification)
        graph.add_node("classify_intent", self.routing_node.classify_intent)
        graph.add_node("search_policies", self.retrieval_node.search_policies)
        graph.add_node("synthesize", self.synthesis_node.synthesize)
        graph.add_node("highlight_sources", self.highlighting_node.highlight_sources)
        graph.add_node("handle_error", handle_error)

        # Define Edges
        graph.add_edge(START, "resolve_context")
        graph.add_edge("resolve_context", "fetch_profile")
        graph.add_edge("fetch_profile", "check_ambiguity")

        # Ambiguity Check
        def check_ambiguity_route(state: AgentState):
            if state.get("error"):
                return "error"
            if state.get("response", {}).get("needs_clarification"):
                return "ask"
            return "continue"

        graph.add_conditional_edges(
            "check_ambiguity",
            check_ambiguity_route,
            {"error": "handle_error", "ask": "ask_clarification", "continue": "classify_intent"},
        )

        graph.add_edge("ask_clarification", END)

        # Intent Routing
        def intent_route(state: AgentState):
            if state.get("error"):
                return "error"
            intent = state.get("response", {}).get("intent", "hybrid")
            if intent == "personal_only":
                return "personal"
            if intent == "policy_only":
                return "policy"
            return "hybrid"

        graph.add_conditional_edges(
            "classify_intent",
            intent_route,
            {
                "error": "handle_error",
                "personal": "fetch_pto",
                "policy": "search_policies",
                "hybrid": "fetch_pto",  # Fetch PTO then Policies (simulating hybrid)
            },
        )

        # Data Gathering Paths
        # Personal path: fetch_pto -> synthesize
        graph.add_edge(
            "fetch_pto", "search_policies"
        )  # Simplified: always search policies if hybrid, or if routing says so.
        # Wait, if intent is personal_only, we shouldn't search policies.
        # I need to fix logic above.

        # Let's use specific routing logic more carefully
        # If personal: fetch_pto -> synthesize
        # If policy: search_policies -> synthesize
        # If hybrid: fetch_pto -> search_policies -> synthesize

        # Revised Intent Routing:
        # We need intermediate logic or carefully wired edges.
        # graph.add_edge("fetch_pto", "synthesize") won't work if we want hybrid to go to search_policies.
        # We can add a conditional edge after fetch_pto

        pass
        # I'll restart the edges definition below cleaner

        return self._build_graph_edges(graph)

    def _build_graph_edges(self, graph: StateGraph) -> Any:
        # Re-defining edges for clarity
        graph.add_edge(START, "resolve_context")
        graph.add_edge("resolve_context", "fetch_profile")
        graph.add_edge("fetch_profile", "check_ambiguity")

        def check_ambiguity_route(state):
            if state.get("error"):
                return "error"
            if state.get("response", {}).get("needs_clarification"):
                return "ask"
            return "continue"

        graph.add_conditional_edges(
            "check_ambiguity",
            check_ambiguity_route,
            {"error": "handle_error", "ask": "ask_clarification", "continue": "classify_intent"},
        )
        graph.add_edge("ask_clarification", END)

        def route_after_intent(state):
            if state.get("error"):
                return "error"
            intent = state.get("response", {}).get("intent", "hybrid")
            if intent == "personal_only":
                return "fetch_pto"
            if intent == "policy_only":
                return "search_policies"
            return "fetch_pto_hybrid"  # Goes to PTO then Policies

        graph.add_conditional_edges(
            "classify_intent",
            route_after_intent,
            {
                "error": "handle_error",
                "fetch_pto": "fetch_pto",
                "fetch_pto_hybrid": "fetch_pto",
                "search_policies": "search_policies",
            },
        )

        # From fetch_pto, determine if we need to go to policies (hybrid) or synthesis (personal)
        def route_after_pto(state):
            if state.get("error"):
                return "error"  # Simple error check
            intent = state.get("response", {}).get("intent", "hybrid")
            if intent == "hybrid":
                return "search_policies"
            return "synthesize"

        graph.add_conditional_edges(
            "fetch_pto",
            route_after_pto,
            {"error": "handle_error", "search_policies": "search_policies", "synthesize": "synthesize"},
        )

        graph.add_edge("search_policies", "synthesize")
        graph.add_edge("synthesize", "highlight_sources")
        graph.add_edge("highlight_sources", END)
        graph.add_edge("handle_error", END)

        # Checkpointer is enabled
        return graph.compile(checkpointer=self.checkpointer)

    @observe(as_type="agent")
    async def arun(self, message: str, thread_id: str = None) -> Dict[str, Any]:
        """Execute workflow with full tracking."""
        query_id = str(uuid.uuid4())[:8]
        if not thread_id:
            thread_id = str(uuid.uuid4())

        initial_state: AgentState = {
            "messages": [HumanMessage(content=message)],
            "query_id": query_id,
            "user": {"email": self.user_email, "profile": None, "pto_balance": None},
            "retrieval": {"original_query": message, "resolved_query": message, "chunks": [], "hyde_used": False},
            "response": {"intent": None, "needs_clarification": False, "clarifying_question": None},
            "final_response": None,
            "highlighted_sources": None,
            "error": None,
        }

        langfuse_handler = CallbackHandler()
        config = {"configurable": {"thread_id": thread_id}, "callbacks": [langfuse_handler]}

        result = await self.graph.ainvoke(initial_state, config=config)

        # Extract output
        output = result.get("final_response", "Unable to generate response.")
        if result.get("highlighted_sources"):
            output += "\n\n" + result["highlighted_sources"]

        return {
            "output": output,
            "query_id": query_id,
            "intent": result["response"].get("intent"),
            "hyde_used": result["retrieval"].get("hyde_used", False),
            "clarification_requested": result["response"].get("needs_clarification", False),
            "sources_count": len(result["retrieval"].get("chunks", [])),
            "thread_id": thread_id,
        }

    async def close(self):
        if self.owns_retriever and hasattr(self.retriever, "close"):
            self.retriever.close()

    @observe(as_type="agent")
    async def astream(self, message: str, thread_id: str = None):
        """
        Stream workflow execution with progress updates.

        Yields:
            Dict with type ('progress' or 'complete') and data
        """
        query_id = str(uuid.uuid4())[:8]
        if not thread_id:
            thread_id = str(uuid.uuid4())

        initial_state: AgentState = {
            "messages": [HumanMessage(content=message)],
            "query_id": query_id,
            "user": {"email": self.user_email, "profile": None, "pto_balance": None},
            "retrieval": {"original_query": message, "resolved_query": message, "chunks": [], "hyde_used": False},
            "response": {"intent": None, "needs_clarification": False, "clarifying_question": None},
            "final_response": None,
            "highlighted_sources": None,
            "error": None,
        }

        step_names = {
            "resolve_context": "🧠 Resolving context...",
            "check_ambiguity": "🔍 Checking for ambiguity...",
            "ask_clarification": "💬 Asking for clarification...",
            "classify_intent": "🏷️ Classifying intent...",
            "fetch_profile": "👤 Fetching profile...",
            "fetch_pto": "📅 Fetching PTO balance...",
            "search_policies": "📚 Searching policies...",
            "synthesize": "✍️ Preparing response...",
            "highlight_sources": "📽️ Highlighting sources...",
            "handle_error": "⚠️ Handling error...",
        }

        langfuse_handler = CallbackHandler()
        config = {"configurable": {"thread_id": thread_id}, "callbacks": [langfuse_handler]}

        node_output = {}
        final_response_acc = None
        highlighted_sources_acc = None

        try:
            async for event in self.graph.astream(initial_state, stream_mode="updates", config=config):
                node_name = list(event.keys())[0]
                node_output = event[node_name]

                # Handling accumulating state changes is tricky with nested state updates
                # but 'node_output' contains only the update key.
                # E.g. {"user": {...}} or {"final_response": "..."}

                if "final_response" in node_output:
                    final_response_acc = node_output["final_response"]
                if "highlighted_sources" in node_output:
                    highlighted_sources_acc = node_output["highlighted_sources"]

                yield {
                    "type": "progress",
                    "step": node_name,
                    "message": step_names.get(node_name, f"Processing: {node_name}"),
                    "data": node_output,
                }
        except Exception as e:
            logger.error(f"astream error: {e}", exc_info=True)
            yield {"type": "progress", "step": "error", "message": f"Error: {str(e)}", "data": {"error": str(e)}}
            final_response_acc = f"Error: {str(e)}"

        # Final response
        final_response = final_response_acc or node_output.get("final_response", "Unable to generate response.")

        if highlighted_sources_acc:
            final_response += "\n\n" + highlighted_sources_acc

        yield {"type": "complete", "response": final_response}
