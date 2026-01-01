import logging
import os
import uuid
import json
from typing import Dict, Any

from langgraph.graph import StateGraph, START, END
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

from .state import EnhancedAgentState
from .nodes import EnhancedWorkflowNodes

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
        self.clarification_detector = ClarificationDetector() if enable_clarification else None
        self.highlighter = SourceHighlighter(use_llm=False) if enable_highlighting else None
        self.memory = SessionMemory(max_turns=10) if enable_memory else None
        self.cost_tracker = CostTracker() if enable_cost_tracking else None

        # Load prompts
        self.prompts = self._load_prompts()

        # Initialize Nodes helper
        self.nodes = EnhancedWorkflowNodes(
            llm=self.llm,
            retriever=self.retriever,
            config=self.config,
            user_email=self.user_email,
            prompts=self.prompts,
            enable_hyde=enable_hyde,
            enable_clarification=enable_clarification,
            enable_highlighting=enable_highlighting,
            enable_memory=enable_memory,
            enable_cost_tracking=enable_cost_tracking,
            clarification_detector=self.clarification_detector,
            highlighter=self.highlighter,
            memory=self.memory,
        )

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
        # Path adjustment: src/chat_rag/agent/workflows/enhanced/workflow.py -> src/chat_rag/agent
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
                logger.info(f"Loaded enhanced workflow prompt '{name}' (version {data.get('version', 'unknown')})")
            except Exception as e:
                logger.error(f"Failed to load prompt '{name}' from {path}: {e}")
                prompts[name] = ""

        return prompts

    def _build_graph(self) -> StateGraph:
        """Build enhanced workflow graph."""
        graph = StateGraph(EnhancedAgentState)

        # Nodes
        graph.add_node("resolve_context", self.nodes.resolve_context)
        graph.add_node("fetch_profile", self.nodes.fetch_profile)
        graph.add_node("check_ambiguity", self.nodes.check_ambiguity)
        graph.add_node("ask_clarification", self.nodes.ask_clarification)
        graph.add_node("classify_intent", self.nodes.classify_intent)
        graph.add_node("gather_data", self.nodes.gather_data)
        graph.add_node("synthesize", self.nodes.synthesize)
        graph.add_node("highlight_sources", self.nodes.highlight_sources)

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
