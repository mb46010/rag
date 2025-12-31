"""
LangChain agent with MCP tools and RAG retriever.

The agent can:
- Retrieve employee profiles via MCP
- Get time-off balances via MCP
- Search policies using RAG retriever
"""

import logging
from typing import Optional, Dict, Any, List
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, ToolMessage, AIMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent
from langchain.tools import tool
from langchain_mcp_adapters.client import MultiServerMCPClient
from langfuse.langchain import CallbackHandler
from langfuse import observe

from chat_rag.retriever import HybridRetriever
from .prompt import SYSTEM_PROMPT

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class HRAssistantAgent:
    """HR Assistant agent with MCP tools and RAG."""

    def __init__(
        self, user_email: str, mcp_url: str = "http://localhost:9000", model: str = "gpt-4", temperature: float = 0.0
    ):
        """
        Initialize HR Assistant agent.

        Args:
            user_email: User's email (from session)
            mcp_url: MCP server URL
            model: OpenAI model name (e.g., "gpt-4", "gpt-4o")
            temperature: LLM temperature
        """
        self.user_email = user_email
        self.mcp_url = mcp_url
        self.model_name = model
        self.temperature = temperature
        logger.info(f"Initializing HRAssistantAgent for user: {user_email}")

        # Initialize retriever
        self.retriever = HybridRetriever()

        # MCP client and agent will be initialized in async context
        self.mcp_client = None
        self.agent = None
        self.thread_id = None

    async def _initialize_mcp_tools(self) -> List:
        """Initialize MCP client and get tools."""
        if self.mcp_client is None:
            logger.info(f"Initializing MCP client for {self.mcp_url}")
            self.mcp_client = MultiServerMCPClient(
                {
                    "hr_services": {
                        "url": f"{self.mcp_url}/sse",
                        "transport": "sse",
                    }
                }
            )
            # Get MCP tools
            mcp_tools = await self.mcp_client.get_tools()
            logger.info(f"Got {len(mcp_tools)} MCP tools")

            # Add retriever tool using @tool decorator
            @tool
            def search_policies(query: str, country: Optional[str] = None) -> str:
                """Search HR policies and retrieve relevant information. Use country filter when you know the employee's country."""
                filters = {}
                if country:
                    filters["country"] = country

                results = self.retriever.retrieve(query, filters=filters)

                return "\n\n".join(
                    [f"Source: {r.document_name} (Section: {r.section_path_str})\n{r.text}" for r in results]
                )

            self.tools = mcp_tools + [search_policies]
            logger.info(f"Total tools: {len(self.tools)}")

        return self.tools

    async def _get_agent(self):
        """Get or create agent."""
        if self.agent is None:
            logger.info("Creating agent...")

            # Initialize tools
            tools = await self._initialize_mcp_tools()

            # Initialize module
            model = ChatOpenAI(model=self.model_name, temperature=self.temperature)

            # Create agent with LangGraph
            self.agent = create_react_agent(model=model, tools=tools, checkpointer=MemorySaver(), prompt=SYSTEM_PROMPT)

            # Generate thread ID for conversation memory
            import uuid

            self.thread_id = str(uuid.uuid4())

            logger.info(f"Agent created with model: {self.model_name}")

        return self.agent

    @observe(as_type="agent")
    async def arun(self, message: str, chat_history: Optional[list] = None) -> Dict[str, Any]:
        """
        Run agent asynchronously.

        Args:
            message: User message
            chat_history: Optional chat history (not used with MemorySaver, kept for compatibility)

        Returns:
            Agent response with output and intermediate steps
        """
        logger.info(f"Running agent for message: {message}")

        # Get agent (initializes on first call)
        agent = await self._get_agent()

        # Inject user email context
        context_message = f"[User context: email={self.user_email}]\n\n{message}"

        # Run agent
        inputs = {"messages": [HumanMessage(content=context_message)]}

        # Initialize Langfuse handler
        langfuse_handler = CallbackHandler()

        config = {
            "configurable": {"thread_id": self.thread_id},
            "callbacks": [langfuse_handler],
        }

        response_text = ""
        intermediate_steps = []

        # We use ainvoke for simplicity to get the final state,
        # but you could use astream to get tokens if needed.
        result = await agent.ainvoke(inputs, config)

        messages = result["messages"]
        if messages:
            last_message = messages[-1]
            if isinstance(last_message, AIMessage):
                response_text = last_message.content

        # Extract intermediate steps (Tool calls and outputs)
        # We look for pairs of AIMessage(tool_calls) and ToolMessage
        for i, msg in enumerate(messages):
            if isinstance(msg, AIMessage) and msg.tool_calls:
                for tool_call in msg.tool_calls:
                    # Find corresponding ToolMessage
                    tool_output = "No output found"
                    # Look ahead for the tool result
                    for next_msg in messages[i + 1 :]:
                        if isinstance(next_msg, ToolMessage) and next_msg.tool_call_id == tool_call["id"]:
                            tool_output = next_msg.content
                            break

                    intermediate_steps.append(
                        {"tool": tool_call["name"], "input": str(tool_call["args"]), "output": str(tool_output)}
                    )

        logger.info("Agent execution complete")
        return {"output": response_text, "intermediate_steps": intermediate_steps}

    async def close(self):
        """Close resources."""
        # Note: MultiServerMCPClient doesn't have a close() method
        # It handles cleanup automatically
        if self.retriever:
            self.retriever.close()
            logger.info("Agent resources closed")


if __name__ == "__main__":
    import os
    import asyncio
    from dotenv import load_dotenv

    load_dotenv()

    # Ensure OPENAI_API_KEY is set
    if not os.getenv("OPENAI_API_KEY"):
        raise ValueError("OPENAI_API_KEY environment variable not set")

    async def test_agent():
        """Test the agent."""
        logger.info("Testing HRAssistantAgent...")

        agent = HRAssistantAgent(user_email="john.doe@company.com")

        try:
            # Test query
            response = await agent.arun("How many PTO days do I have left?")

            print("\n" + "=" * 80)
            print("AGENT RESPONSE:")
            print("=" * 80)
            print(response["output"])
            print("=" * 80)
            print("STEPS:")
            for step in response["intermediate_steps"]:
                print(f"Tool: {step['tool']}")
                print(f"Input: {step['input']}")
                print(f"Output: {step['output'][:100]}...")  # truncate
                print("-" * 20)

        finally:
            await agent.close()

    # Run the test
    asyncio.run(test_agent())
