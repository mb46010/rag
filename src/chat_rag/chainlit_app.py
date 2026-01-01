"""
Chainlit chatbot app for HR Assistant.

Provides chat interface for employees to ask HR policy questions.
Uses the workflow-based agent with streaming progress updates.
"""

import logging
from typing import Optional
import chainlit as cl
from chainlit.input_widget import TextInput
from dotenv import load_dotenv

from chat_rag.agent import HRWorkflowAgent, AgentConfig
from chat_rag.retriever import EnhancedHybridRetriever, RetrievalConfig

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


@cl.on_chat_start
async def on_chat_start():
    """Handle chat session start - get user email and initialize agent."""
    logger.info("New chat session started")

    # Mock login - get user email
    settings = await cl.ChatSettings(
        [
            TextInput(
                id="user_email",
                label="Employee Email",
                initial="john.doe@company.com",
                placeholder="Enter your email address",
            )
        ]
    ).send()

    user_email = settings.get("user_email", "john.doe@company.com")
    logger.info(f"User logged in: {user_email}")

    # Store user email in session
    cl.user_session.set("user_email", user_email)

    # Initialize components
    try:
        logger.info(f"Initializing agent for {user_email}")

        # Initialize retriever with enhanced features
        retriever_config = RetrievalConfig(
            enable_reranking=True,
            enable_context_window=True,
            enable_rrf=False,  # Use standard hybrid search
            debug_to_file=True,
            retrieval_output_dir="output/retrieval",
        )
        retriever = EnhancedHybridRetriever(retriever_config)

        # Initialize workflow agent
        agent_config = AgentConfig(
            model_name="gpt-4o",
            temperature=0.1,
            enable_streaming=True,
            enable_parallel_fetch=True,
        )
        agent = HRWorkflowAgent(
            user_email=user_email,
            retriever=retriever,
            config=agent_config,
        )

        cl.user_session.set("agent", agent)
        cl.user_session.set("retriever", retriever)

        # Welcome message
        await cl.Message(
            content=f"""Welcome to the HR Assistant! 👋

You're logged in as: **{user_email}**

I can help you with:
• PTO and leave policies
• Expense reimbursement policies
• Work from home policies
• Your employee information
• And other HR-related questions

Ask me anything about HR policies!"""
        ).send()

    except Exception as e:
        logger.error(f"Error initializing agent: {e}", exc_info=True)
        await cl.Message(
            content=f"""Error initializing agent: {str(e)}

Please ensure:
1. Weaviate is running (http://localhost:8080)
2. MCP server is running (http://localhost:9000)
3. Documents have been ingested
4. OPENAI_API_KEY is set in your environment"""
        ).send()


@cl.on_settings_update
async def on_settings_update(settings):
    """Handle settings update (email change)."""
    user_email = settings.get("user_email")
    logger.info(f"Settings updated: user_email={user_email}")

    # Close old agent
    old_agent = cl.user_session.get("agent")
    if old_agent:
        await old_agent.close()

    # Update session
    cl.user_session.set("user_email", user_email)

    # Create new agent with existing retriever
    try:
        retriever = cl.user_session.get("retriever")
        if not retriever:
            retriever = EnhancedHybridRetriever(RetrievalConfig(enable_reranking=True))
            cl.user_session.set("retriever", retriever)

        agent = HRWorkflowAgent(
            user_email=user_email,
            retriever=retriever,
        )
        cl.user_session.set("agent", agent)

        await cl.Message(content=f"Switched to user: **{user_email}**").send()

    except Exception as e:
        logger.error(f"Error updating agent: {e}")
        await cl.Message(content=f"Error updating agent: {str(e)}").send()


@cl.on_message
async def on_message(message: cl.Message):
    """Handle incoming chat message with streaming progress."""
    user_email = cl.user_session.get("user_email")
    agent: Optional[HRWorkflowAgent] = cl.user_session.get("agent")

    if not agent:
        await cl.Message(content="Agent not initialized. Please refresh the page.").send()
        return

    logger.info(f"Processing message from {user_email}: {message.content}")

    # Create response message
    msg = cl.Message(content="")
    await msg.send()

    try:
        # Stream progress updates
        async for event in agent.astream(message.content):
            if event["type"] == "progress":
                msg.content = event["message"]
                await msg.update()

            elif event["type"] == "complete":
                msg.content = event["response"]
                await msg.update()

        # Update chat history
        chat_history = cl.user_session.get("chat_history", [])
        chat_history.append({"role": "user", "content": message.content})
        chat_history.append({"role": "assistant", "content": msg.content})
        cl.user_session.set("chat_history", chat_history)

        logger.info("Message processed successfully")

    except Exception as e:
        logger.error(f"Error processing message: {e}", exc_info=True)
        msg.content = f"""An error occurred: {str(e)}

Please try again or rephrase your question. If the problem persists,
check that all services are running properly."""
        await msg.update()


@cl.on_chat_end
async def on_chat_end():
    """Handle chat session end - cleanup resources."""
    logger.info("Chat session ending")

    # Close agent
    agent: Optional[HRWorkflowAgent] = cl.user_session.get("agent")
    if agent:
        await agent.close()
        logger.info("Agent closed")


if __name__ == "__main__":
    logger.info("Starting Chainlit app...")
    logger.info("Run with: chainlit run src/chat_rag/chainlit_app.py -w")
