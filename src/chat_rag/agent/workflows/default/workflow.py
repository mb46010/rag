import logging
import asyncio
import os
import json
import uuid
from typing import Dict, List, Optional, Any

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from langchain_mcp_adapters.client import MultiServerMCPClient
from langfuse.langchain import CallbackHandler
from langfuse import observe

from chat_rag.retriever import EnhancedHybridRetriever, RetrievalConfig, HyDERetriever
from chat_rag.agent.config import AgentConfig
from .state import AgentState, QueryIntent
from .nodes.mcp import MCPNode
from .nodes.retrieval import RetrievalNode
from .nodes.routing import RoutingNode
from .nodes.synthesis import SynthesisNode
from .nodes.parallel import ParallelFetchNode
from .nodes.common import CommonOptimizations
from .nodes.error_handling import handle_error

logger = logging.getLogger(__name__)


class HRWorkflowAgent:
    """
    Structured workflow agent for HR queries.
    """

    def __init__(
        self,
        user_email: str,
        retriever: EnhancedHybridRetriever = None,
        config: AgentConfig = None,
        checkpointer: Any = None,
    ):
        self.user_email = user_email
        self.config = config or AgentConfig()

        # Initialize retriever
        if retriever:
            self.retriever = retriever
            self.owns_retriever = False
        else:
            self.retriever = EnhancedHybridRetriever(RetrievalConfig(enable_reranking=True))
            self.owns_retriever = True

        # Wrap with HyDE if enabled
        if getattr(self.retriever, "config", None) and getattr(self.retriever.config, "enable_hyde", False):
            if not isinstance(self.retriever, HyDERetriever):
                logger.info("HyDE enabled in retriever config - wrapping with HyDERetriever")
                self.retriever = HyDERetriever(self.retriever)

        # Initialize LLM
        self.llm = ChatOpenAI(
            model=self.config.model_name,
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
        )

        # Load prompts
        self.prompts = self._load_prompts()

        # Checkpointer
        self.checkpointer = checkpointer or MemorySaver()

        # Initialize Nodes
        self.mcp_node = MCPNode(self.config.mcp_url, self.user_email)
        self.retrieval_node = RetrievalNode(self.retriever)
        self.routing_node = RoutingNode(self.llm, self.prompts)
        self.synthesis_node = SynthesisNode(self.llm, self.prompts)
        self.parallel_node = ParallelFetchNode(self.mcp_node, self.retrieval_node, self.config.enable_parallel_fetch)
        self.common_node = CommonOptimizations()

        # Build workflow graph
        self.graph = self._build_graph()

        logger.info(f"HRWorkflowAgent initialized for {user_email}")

    def _load_prompts(self) -> Dict[str, str]:
        """Load prompts from the workflow prompts directory."""
        prompts = {}
        base_path = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))  # src/chat_rag/agent
        prompts_dir = os.path.join(base_path, "prompts", "workflow")

        for name in ["classification", "synthesis"]:
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
        """Build the LangGraph workflow."""
        graph = StateGraph(AgentState)

        # Add nodes
        graph.add_node("classify_intent", self.routing_node.classify_intent)
        graph.add_node("handle_chitchat", self.common_node.handle_chitchat)
        graph.add_node("handle_out_of_scope", self.common_node.handle_out_of_scope)
        graph.add_node("fetch_personal_data", self.mcp_node.fetch_personal_data)
        graph.add_node("search_policies", self.retrieval_node.search_policies)
        graph.add_node("parallel_fetch", self.parallel_node.parallel_fetch)
        graph.add_node("synthesize", self.synthesis_node.synthesize_response)
        graph.add_node("handle_error", handle_error)

        # Define edges
        graph.add_edge(START, "classify_intent")

        # Route based on intent
        def route_intent(state):
            if state.get("error"):
                return "handle_error"
            intent = state.get("response", {}).get("intent", "hybrid")
            # Map intents to nodes
            mapping = {
                "policy_only": "search_policies",
                "personal_only": "fetch_personal_data",
                "hybrid": "parallel_fetch",
                "chitchat": "handle_chitchat",
                "out_of_scope": "handle_out_of_scope",
            }
            return mapping.get(intent, "parallel_fetch")

        graph.add_conditional_edges(
            "classify_intent",
            route_intent,
            {
                "search_policies": "search_policies",
                "fetch_personal_data": "fetch_personal_data",
                "parallel_fetch": "parallel_fetch",
                "handle_chitchat": "handle_chitchat",
                "handle_out_of_scope": "handle_out_of_scope",
                "handle_error": "handle_error",
            },
        )

        # Standard flow -> synthesize (with error check)
        def route_to_synthesis_or_error(state):
            if state.get("error"):
                return "handle_error"
            return "synthesize"

        for node in ["fetch_personal_data", "search_policies", "parallel_fetch"]:
            graph.add_conditional_edges(
                node, route_to_synthesis_or_error, {"handle_error": "handle_error", "synthesize": "synthesize"}
            )

        # Terminal nodes
        graph.add_edge("handle_chitchat", END)
        graph.add_edge("handle_out_of_scope", END)
        graph.add_edge("synthesize", END)
        graph.add_edge("handle_error", END)

        return graph.compile(checkpointer=self.checkpointer)

    @observe(as_type="agent")
    async def arun(self, message: str, thread_id: str = None) -> Dict[str, Any]:
        """Execute the workflow."""
        if not thread_id:
            thread_id = str(uuid.uuid4())

        initial_state: AgentState = {
            "messages": [HumanMessage(content=message)],
            "user": {"email": self.user_email, "profile": None, "pto_balance": None},
            "retrieval": {"policy_results": None},
            "response": {"intent": None},
            "final_response": None,
            "error": None,
        }

        langfuse_handler = CallbackHandler()
        config = {"configurable": {"thread_id": thread_id}, "callbacks": [langfuse_handler]}

        result = await self.graph.ainvoke(initial_state, config=config)

        # Flatten for output compatibility
        user_state = result.get("user", {})
        retrieval_state = result.get("retrieval", {})
        response_state = result.get("response", {})

        return {
            "output": result.get("final_response", "Unable to generate response."),
            "intent": response_state.get("intent"),
            # No intermediate steps formatted here yet, can refer to node execution if needed.
            "sources": {
                "profile": user_state.get("profile") is not None and "error" not in (user_state.get("profile") or {}),
                "pto": user_state.get("pto_balance") is not None
                and "error" not in (user_state.get("pto_balance") or {}),
                "policies": len(retrieval_state.get("policy_results") or []),
            },
            "error": result.get("error"),
            "thread_id": thread_id,
        }

    @observe(as_type="agent")
    async def astream(self, message: str, thread_id: str = None):
        """Stream workflow execution."""
        if not thread_id:
            thread_id = str(uuid.uuid4())

        initial_state: AgentState = {
            "messages": [HumanMessage(content=message)],
            "user": {"email": self.user_email, "profile": None, "pto_balance": None},
            "retrieval": {"policy_results": None},
            "response": {"intent": None},
            "final_response": None,
            "error": None,
        }

        step_names = {
            "classify_intent": "🔍 Analyzing your question...",
            "fetch_personal_data": "📋 Fetching your profile...",
            "search_policies": "📚 Searching policies...",
            "parallel_fetch": "⚡ Gathering information...",
            "synthesize": "✍️ Preparing response...",
            "handle_chitchat": "👋 ",
            "handle_out_of_scope": "ℹ️ ",
            "handle_error": "⚠️ Handling error...",
        }

        langfuse_handler = CallbackHandler()
        config = {"configurable": {"thread_id": thread_id}, "callbacks": [langfuse_handler]}

        final_response_acc = None
        node_output = {}

        try:
            async for event in self.graph.astream(initial_state, stream_mode="updates", config=config):
                node_name = list(event.keys())[0]
                node_output = event[node_name]

                if not node_output:
                    continue

                if "final_response" in node_output:
                    final_response_acc = node_output["final_response"]

                yield {
                    "type": "progress",
                    "step": node_name,
                    "message": step_names.get(node_name, f"Processing: {node_name}"),
                    "data": node_output,
                }
        except Exception as e:
            logger.error(f"astream error: {e}")
            yield {"type": "progress", "step": "error", "message": f"Error: {str(e)}", "data": {"error": str(e)}}
            final_response_acc = f"Error: {str(e)}"

        final_response = final_response_acc or node_output.get("final_response")
        if final_response:
            yield {"type": "complete", "response": final_response}

    async def close(self):
        if self.owns_retriever and hasattr(self.retriever, "close"):
            self.retriever.close()
