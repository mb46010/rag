"""
Chainlit chatbot app for HR Assistant.

Provides chat interface for employees to ask HR policy questions.
"""

import logging
from typing import Optional
import chainlit as cl
from chainlit.input_widget import TextInput
from dotenv import load_dotenv

from chat_rag.agent import HRAssistantAgent

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


@cl.on_chat_start
async def on_chat_start():
    """Handle chat session start - get user email."""
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

    # Initialize agent
    try:
        logger.info(f"Initializing agent for {user_email}")
        agent = HRAssistantAgent(user_email=user_email)
        cl.user_session.set("agent", agent)

        # Welcome message
        await cl.Message(
            content=f"""Welcome to the HR Assistant! 👋

You're logged in as: **{user_email}**

I can help you with:
- PTO and leave policies
- Expense reimbursement policies
- Work from home policies
- Code of conduct
- And other HR-related questions

Ask me anything about HR policies!"""
        ).send()

    except Exception as e:
        logger.error(f"Error initializing agent: {e}")
        await cl.Message(
            content=f"Error initializing agent: {str(e)}\n\nPlease ensure:\n1. Weaviate is running (http://localhost:8080)\n2. MCP server is running (http://localhost:9000)\n3. Documents have been ingested\n4. OPENAI_API_KEY is set"
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

    # Create new agent
    try:
        agent = HRAssistantAgent(user_email=user_email)
        cl.user_session.set("agent", agent)

        await cl.Message(content=f"Switched to user: **{user_email}**").send()
    except Exception as e:
        logger.error(f"Error updating agent: {e}")
        await cl.Message(content=f"Error updating agent: {str(e)}").send()


@cl.on_message
async def on_message(message: cl.Message):
    """Handle incoming chat message."""
    user_email = cl.user_session.get("user_email")
    agent: Optional[HRAssistantAgent] = cl.user_session.get("agent")

    if not agent:
        await cl.Message(content="Agent not initialized. Please refresh the page.").send()
        return

    logger.info(f"Processing message from {user_email}: {message.content}")

    # Create a message to show thinking
    msg = cl.Message(content="")
    await msg.send()

    try:
        # Get chat history
        chat_history = cl.user_session.get("chat_history", [])

        # Run agent
        logger.info("Running agent...")
        result = await agent.arun(message=message.content, chat_history=chat_history)

        # Extract response
        output = result.get("output", "No response generated.")
        intermediate_steps = result.get("intermediate_steps", [])

        # Log intermediate steps
        logger.info(f"Agent completed with {len(intermediate_steps)} steps")

        # Format response with tool usage info
        response = output

        # Show tool calls if any
        # Show tool calls if any
        if intermediate_steps:
            tool_info = "\n\n---\n**Tools Used:**\n"
            for step in intermediate_steps:
                # Handle legacy tuple format (action, observation)
                if isinstance(step, (tuple, list)) and len(step) == 2:
                    action, observation = step
                    tool_name = getattr(action, "tool", str(action))
                    tool_input = getattr(action, "tool_input", "")
                # Handle dict format
                elif isinstance(step, dict):
                    tool_name = step.get("tool", "Unknown Tool")
                    tool_input = step.get("input", "")
                else:
                    logger.warning(f"Unknown step format: {step}")
                    continue

                tool_info += f"- `{tool_name}`: {tool_input}\n"

            response += tool_info

        # Update message
        msg.content = response
        await msg.update()

        # Update chat history
        chat_history.append({"role": "user", "content": message.content})
        chat_history.append({"role": "assistant", "content": output})
        cl.user_session.set("chat_history", chat_history)

        logger.info("Message processed successfully")

    except Exception as e:
        logger.error(f"Error processing message: {e}", exc_info=True)
        msg.content = f"An error occurred: {str(e)}\n\nPlease try again or rephrase your question."
        await msg.update()


@cl.on_chat_end
async def on_chat_end():
    """Handle chat session end."""
    logger.info("Chat session ending")

    # Close agent
    agent: Optional[HRAssistantAgent] = cl.user_session.get("agent")
    if agent:
        await agent.close()
        logger.info("Agent closed")


if __name__ == "__main__":
    logger.info("Starting Chainlit app...")
    logger.info("Run with: chainlit run src/chainlit_app.py -w")
