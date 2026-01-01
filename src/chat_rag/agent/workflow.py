"""
LangGraph Workflow Agent for HR Assistant.

This replaces the ReAct agent with a structured workflow that:
1. Classifies query intent
2. Routes to appropriate data gathering
3. Executes tools in parallel where possible
4. Synthesizes response with citations

Benefits over ReAct:
- Fewer LLM calls (2-3 vs 4-5)
- More predictable behavior
- Better error handling
- Parallel execution support
"""

import logging
import asyncio
from typing import TypedDict, Annotated, Optional, List, Dict, Any
from enum import Enum

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langchain_mcp_adapters.client import MultiServerMCPClient
from tenacity import retry, stop_after_attempt, wait_exponential

from chat_rag.retriever import EnhancedHybridRetriever, RetrievalConfig
from .config import AgentConfig

logger = logging.getLogger(__name__)


class QueryIntent(Enum):
    """Classified query intents for routing."""

    POLICY_ONLY = "policy_only"  # General policy questions
    PERSONAL_ONLY = "personal_only"  # User's specific data
    HYBRID = "hybrid"  # Needs both personal data AND policies
    CHITCHAT = "chitchat"  # Greetings, casual conversation
    OUT_OF_SCOPE = "out_of_scope"  # Not HR-related


class AgentState(TypedDict):
    """State that flows through the workflow graph."""

    messages: Annotated[list, add_messages]
    user_email: str
    intent: Optional[str]
    employee_profile: Optional[Dict]
    time_off_balance: Optional[Dict]
    policy_results: Optional[List[Dict]]
    error: Optional[str]
    final_response: Optional[str]


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

    CLASSIFICATION_PROMPT = """Classify this HR-related query into one category:

Query: "{query}"

Categories:
- "policy_only": General policy questions that don't need personal employee data
  Examples: "What's the WFH policy?", "How do expense reports work?", "What are the office hours?"

- "personal_only": Questions about the user's specific data that don't need policy lookup
  Examples: "How many PTO days do I have?", "What's my job title?", "What team am I on?"

- "hybrid": Questions that need BOTH personal data AND policy information
  Examples: "Am I eligible for parental leave?", "Based on my tenure, how much PTO do I get?"

- "chitchat": Greetings or casual conversation
  Examples: "Hello", "Thanks!", "How are you?"

- "out_of_scope": Not HR-related at all
  Examples: "What's the weather?", "Help me write code"

Respond with ONLY the category name, nothing else."""

    SYNTHESIS_PROMPT = """You are an HR assistant. Answer the user's question based on the provided context.

{context}

User Question: {query}

Instructions:
- Be concise and accurate
- Cite policies as: "According to [Policy Name]..."
- For personal data, be specific to the user
- If information is missing, clearly state what's unavailable
- Never make up policy details or employee information
- If confidence is low, mention that the user should verify with HR"""

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
        else:
            self.retriever = EnhancedHybridRetriever(
                RetrievalConfig(enable_reranking=True)
            )

        # Initialize LLM
        self.llm = ChatOpenAI(
            model=self.config.model_name,
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
        )

        # MCP client (initialized lazily)
        self._mcp_client = None
        self._mcp_tools = None

        # Build workflow graph
        self.graph = self._build_graph()

        logger.info(f"HRWorkflowAgent initialized for {user_email}")

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

    def _build_graph(self) -> StateGraph:
        """Build the LangGraph workflow."""
        graph = StateGraph(AgentState)

        # Add nodes
        graph.add_node("classify_intent", self._classify_intent)
        graph.add_node("handle_chitchat", self._handle_chitchat)
        graph.add_node("handle_out_of_scope", self._handle_out_of_scope)
        graph.add_node("fetch_personal_data", self._fetch_personal_data)
        graph.add_node("search_policies", self._search_policies)
        graph.add_node("parallel_fetch", self._parallel_fetch)
        graph.add_node("synthesize", self._synthesize_response)

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

    # -------------------------
    # Node implementations
    # -------------------------

    async def _classify_intent(self, state: AgentState) -> Dict:
        """Classify query intent for routing."""
        query = state["messages"][-1].content

        prompt = self.CLASSIFICATION_PROMPT.format(query=query)
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

    def _route_by_intent(self, state: AgentState) -> str:
        """Route to appropriate node based on intent."""
        return state.get("intent", "hybrid")

    async def _handle_chitchat(self, state: AgentState) -> Dict:
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

    async def _handle_out_of_scope(self, state: AgentState) -> Dict:
        """Handle out-of-scope queries."""
        return {
            "final_response": (
                "I'm an HR assistant and can help with HR policies, benefits, "
                "time off, and workplace questions. For other topics, please "
                "reach out to the appropriate team or department."
            )
        }

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
                return result

        raise ValueError(f"Tool {tool_name} not found")

    async def _fetch_personal_data(self, state: AgentState) -> Dict:
        """Fetch employee profile and PTO balance via MCP."""
        results = {}

        # Fetch profile
        try:
            profile = await self._call_mcp_tool(
                "get_employee_profile_tool", {"email": self.user_email}
            )
            results["employee_profile"] = profile
            logger.info(f"Fetched profile for {self.user_email}")
        except Exception as e:
            logger.error(f"Failed to fetch profile: {e}")
            results["employee_profile"] = {"error": str(e)}

        # Fetch PTO balance
        try:
            balance = await self._call_mcp_tool(
                "get_time_off_balance_tool", {"email": self.user_email}
            )
            results["time_off_balance"] = balance
            logger.info(f"Fetched PTO balance for {self.user_email}")
        except Exception as e:
            logger.error(f"Failed to fetch PTO: {e}")
            results["time_off_balance"] = {"error": str(e)}

        return results

    async def _search_policies(self, state: AgentState) -> Dict:
        """Search policy documents using RAG."""
        query = state["messages"][-1].content

        # Use employee country if available
        filters = {"active": True}
        if state.get("employee_profile") and isinstance(state["employee_profile"], dict):
            country = state["employee_profile"].get("country")
            if country:
                filters["country"] = country

        try:
            chunks = self.retriever.retrieve(query, filters=filters, top_k=3)

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

    async def _parallel_fetch(self, state: AgentState) -> Dict:
        """Fetch personal data and policies in parallel."""
        if not self.config.enable_parallel_fetch:
            # Sequential fallback
            personal_result = await self._fetch_personal_data(state)
            state_with_profile = {**state, **personal_result}
            policy_result = await self._search_policies(state_with_profile)
            return {**personal_result, **policy_result}

        # Parallel execution
        # First get profile to use country filter for policies
        try:
            profile = await self._call_mcp_tool(
                "get_employee_profile_tool", {"email": self.user_email}
            )
        except Exception as e:
            profile = {"error": str(e)}

        # Now run PTO and policy search in parallel
        state_with_profile = {**state, "employee_profile": profile}

        async def fetch_pto():
            try:
                return await self._call_mcp_tool(
                    "get_time_off_balance_tool", {"email": self.user_email}
                )
            except Exception as e:
                return {"error": str(e)}

        async def fetch_policies():
            return await self._search_policies(state_with_profile)

        pto_task = asyncio.create_task(fetch_pto())
        policy_task = asyncio.create_task(fetch_policies())

        pto_result, policy_result = await asyncio.gather(
            pto_task, policy_task, return_exceptions=True
        )

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

    async def _synthesize_response(self, state: AgentState) -> Dict:
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
                "**Time Off Balance:**\n"
                f"• Days remaining: {balance.get('number_of_days_left', 'N/A')}"
            )

        # Policy results
        if state.get("policy_results"):
            policy_text = "\n\n".join(
                [
                    f"[{p['source']}] (confidence: {p['confidence']})\n{p['text']}"
                    for p in state["policy_results"]
                ]
            )
            context_parts.append(f"**Relevant Policies:**\n{policy_text}")

        context = (
            "\n\n---\n\n".join(context_parts)
            if context_parts
            else "No context available."
        )

        # Generate response
        prompt = self.SYNTHESIS_PROMPT.format(context=context, query=query)

        response = await self.llm.ainvoke(
            [
                SystemMessage(content="You are a helpful HR assistant."),
                HumanMessage(content=prompt),
            ]
        )

        return {"final_response": response.content}

    # -------------------------
    # Public interface
    # -------------------------

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

        result = await self.graph.ainvoke(initial_state)

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

        async for event in self.graph.astream(initial_state, stream_mode="updates"):
            node_name = list(event.keys())[0]
            node_output = event[node_name]

            yield {
                "type": "progress",
                "step": node_name,
                "message": step_names.get(node_name, f"Processing: {node_name}"),
                "data": node_output,
            }

        # Final response
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
        if self.retriever:
            self.retriever.close()
        logger.info("Agent closed")


# Keep the old class name for backwards compatibility
HRAssistantAgent = HRWorkflowAgent
