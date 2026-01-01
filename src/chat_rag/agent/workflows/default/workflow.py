import logging
import asyncio
import os
import json
from typing import Dict, List, Optional, Any

from langgraph.graph import StateGraph, START, END
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from langchain_mcp_adapters.client import MultiServerMCPClient
from langfuse.langchain import CallbackHandler
from langfuse import observe

from chat_rag.retriever import EnhancedHybridRetriever, RetrievalConfig, HyDERetriever
from chat_rag.agent.config import AgentConfig
from .state import AgentState, QueryIntent
from .nodes import WorkflowNodes

logger = logging.getLogger(__name__)


class HRWorkflowAgent:
    """
    Structured workflow agent for HR queries.

    Flow:
    1. Classify query → determine intent
    2. Route based on intent:
       - policy_only → RAG search → synthesize
       - personal_only → MCP tools → synthesize
       - hybrid → parallel fetch (MCP + RAG) → synthesize
       - chitchat/out_of_scope → direct response
    3. Synthesize response with citations
    """

    def __init__(
        self,
        user_email: str,
        retriever: EnhancedHybridRetriever = None,
        config: AgentConfig = None,
    ):
        """
        Initialize the workflow agent.

        Args:
            user_email: Email of the current user
            retriever: Optional pre-configured retriever
            config: Agent configuration
        """
        self.user_email = user_email
        self.config = config or AgentConfig()

        # Initialize retriever
        if retriever:
            self.retriever = retriever
            self.owns_retriever = False
        else:
            self.retriever = EnhancedHybridRetriever(RetrievalConfig(enable_reranking=True))
            self.owns_retriever = True

        # Wrap with HyDE if enabled in the retriever's config
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

        # Initialize Nodes helper
        self.nodes = WorkflowNodes(
            llm=self.llm,
            retriever=self.retriever,
            config=self.config,
            user_email=self.user_email,
            prompts=self.prompts,
        )

        # Build workflow graph
        self.graph = self._build_graph()

        logger.info(f"HRWorkflowAgent initialized for {user_email}")

    def _load_prompts(self) -> Dict[str, str]:
        """Load prompts from the workflow prompts directory."""
        prompts = {}
        # Note: We need to point to the correct directory relative to the package structure or absolute path
        # Originally it was os.path.dirname(__file__) -> prompts -> workflow
        # Now __file__ is inside src/chat_rag/agent/workflows/default/
        # So we need to go up two levels to get to agent/prompts/workflow?
        # No, prompts seem to be in src/chat_rag/agent/prompts/workflow

        # Let's adjust the path finding logic.
        # Current file: .../src/chat_rag/agent/workflows/default/workflow.py
        # Prompts: .../src/chat_rag/agent/prompts/workflow

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
                logger.info(f"Loaded workflow prompt '{name}' (version {data.get('version', 'unknown')})")
            except Exception as e:
                logger.error(f"Failed to load prompt '{name}' from {path}: {e}")
                # Fallback to empty string or raise?
                prompts[name] = ""

        return prompts

    def _build_graph(self) -> StateGraph:
        """Build the LangGraph workflow."""
        graph = StateGraph(AgentState)

        # Add nodes
        graph.add_node("classify_intent", self.nodes.classify_intent)
        graph.add_node("handle_chitchat", self.nodes.handle_chitchat)
        graph.add_node("handle_out_of_scope", self.nodes.handle_out_of_scope)
        graph.add_node("fetch_personal_data", self.nodes.fetch_personal_data)
        graph.add_node("search_policies", self.nodes.search_policies)
        graph.add_node("parallel_fetch", self.nodes.parallel_fetch)
        graph.add_node("synthesize", self.nodes.synthesize_response)

        # Define edges
        graph.add_edge(START, "classify_intent")

        # Route based on intent
        graph.add_conditional_edges(
            "classify_intent",
            self._route_by_intent,
            {
                "policy_only": "search_policies",
                "personal_only": "fetch_personal_data",
                "hybrid": "parallel_fetch",
                "chitchat": "handle_chitchat",
                "out_of_scope": "handle_out_of_scope",
            },
        )

        # Personal data flow → synthesize
        graph.add_edge("fetch_personal_data", "synthesize")

        # Policy flow → synthesize
        graph.add_edge("search_policies", "synthesize")

        # Parallel flow → synthesize
        graph.add_edge("parallel_fetch", "synthesize")

        # Terminal nodes
        graph.add_edge("handle_chitchat", END)
        graph.add_edge("handle_out_of_scope", END)
        graph.add_edge("synthesize", END)

        return graph.compile()

    def _route_by_intent(self, state: AgentState) -> str:
        """Route to appropriate node based on intent."""
        return state.get("intent", "hybrid")

    # -------------------------
    # Public interface
    # -------------------------

    @observe(as_type="agent")
    async def arun(self, message: str) -> Dict[str, Any]:
        """
        Execute the workflow.

        Args:
            message: User's message

        Returns:
            Dict with output, intent, and sources metadata
        """
        initial_state: AgentState = {
            "messages": [HumanMessage(content=message)],
            "user_email": self.user_email,
            "intent": None,
            "employee_profile": None,
            "time_off_balance": None,
            "policy_results": None,
            "error": None,
            "final_response": None,
        }

        # Initialize Langfuse handler for LangChain components
        langfuse_handler = CallbackHandler()

        # Run graph with callbacks
        result = await self.graph.ainvoke(initial_state, config={"callbacks": [langfuse_handler]})

        return {
            "output": result.get("final_response", "Unable to generate response."),
            "intent": result.get("intent"),
            "intermediate_steps": self._format_intermediate_steps(result),
            "sources": {
                "profile": result.get("employee_profile") is not None
                and "error" not in (result.get("employee_profile") or {}),
                "pto": result.get("time_off_balance") is not None
                and "error" not in (result.get("time_off_balance") or {}),
                "policies": len(result.get("policy_results") or []),
            },
            "error": result.get("error"),
        }

    @observe(as_type="agent")
    async def astream(self, message: str):
        """
        Stream workflow execution with progress updates.

        Yields:
            Dict with type ('progress' or 'complete') and data
        """
        initial_state: AgentState = {
            "messages": [HumanMessage(content=message)],
            "user_email": self.user_email,
            "intent": None,
            "employee_profile": None,
            "time_off_balance": None,
            "policy_results": None,
            "error": None,
            "final_response": None,
        }

        step_names = {
            "classify_intent": "🔍 Analyzing your question...",
            "fetch_personal_data": "📋 Fetching your profile...",
            "search_policies": "📚 Searching policies...",
            "parallel_fetch": "⚡ Gathering information...",
            "synthesize": "✍️ Preparing response...",
            "handle_chitchat": "👋 ",
            "handle_out_of_scope": "ℹ️ ",
        }

        # Initialize Langfuse handler
        langfuse_handler = CallbackHandler()

        async for event in self.graph.astream(
            initial_state, stream_mode="updates", config={"callbacks": [langfuse_handler]}
        ):
            node_name = list(event.keys())[0]
            node_output = event[node_name]

            yield {
                "type": "progress",
                "step": node_name,
                "message": step_names.get(node_name, f"Processing: {node_name}"),
                "data": node_output,
            }

        # Final response
        # Note: the last event yielded might not contain the final response if we loop,
        # but in this DAG it should be fine. We can check the node_output of the last step.
        # But 'node_output' is from the loop which is local scope? No, Python scoping in loops...
        # Wait, 'node_output' will hold the value of the last iteration.

        if "final_response" in node_output:
            yield {"type": "complete", "response": node_output["final_response"]}

    def _format_intermediate_steps(self, state: AgentState) -> List[Dict]:
        """Format state into intermediate steps for compatibility."""
        steps = []

        if state.get("intent"):
            steps.append(
                {
                    "tool": "classify_intent",
                    "input": state["messages"][-1].content if state.get("messages") else "",
                    "output": state["intent"],
                }
            )

        if state.get("employee_profile"):
            steps.append(
                {
                    "tool": "get_employee_profile",
                    "input": self.user_email,
                    "output": str(state["employee_profile"]),
                }
            )

        if state.get("time_off_balance"):
            steps.append(
                {
                    "tool": "get_time_off_balance",
                    "input": self.user_email,
                    "output": str(state["time_off_balance"]),
                }
            )

        if state.get("policy_results"):
            steps.append(
                {
                    "tool": "search_policies",
                    "input": state["messages"][-1].content if state.get("messages") else "",
                    "output": f"Found {len(state['policy_results'])} relevant policies",
                }
            )

        return steps

    async def close(self):
        """Clean up resources."""
        if self.owns_retriever and hasattr(self.retriever, "close"):
            self.retriever.close()
            logger.info("Closed agent-owned retriever")
        logger.info("Agent closed")
