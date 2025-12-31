# HR Assistant Chatbot - RAG Demo

AI assistant that can answer questions about HR policies and procedures using RAG (Retrieval-Augmented Generation) and MCP (Model Context Protocol).

## Overview

This demo showcases a chatbot that:
- Answers HR policy questions using RAG
- Retrieves employee information via MCP tools
- Uses hybrid search (BM25 + Vector similarity) for accurate retrieval
- Provides citations to source policies

**Note**: This is a tech demo. It omits authentication, logging, scaling considerations, and security features like CORS and rate limiting.

## Architecture

```
[User] -> [Chainlit Chatbot] -> [AI Agent with MCP tools and RAG] -> [Answer] -> [User]
```

## Tech Stack

- **Python** - Programming language
- **Chainlit** - Chatbot UI framework
- **LangChain** - Agent orchestration with MCP tools and RAG
- **OpenAI** - LLM and embeddings (requires `OPENAI_API_KEY`)
- **FastAPI** - Web framework for MCP server
- **FastMCP** - MCP server implementation
- **LlamaIndex** - RAG ingestion and retrieval
- **Weaviate** - Local vector store (BYOV mode)
- **SQLite** - Ingestion metadata storage

## Prerequisites

- Python 3.10+
- Docker (for Weaviate)
- OpenAI API key

## Setup

### 1. Clone and Setup Environment

Please follow the instructions in [INSTALL.md](INSTALL.md) to set up the environment with `uv`.

```bash
uv lock
uv sync
```

### 2. Configure Environment Variables

Create a `.env` file in the project root:

```env
OPENAI_API_KEY=your_openai_api_key_here
```

### 3. Start Weaviate Vector Store

```bash
# Start Weaviate with Docker
bash scripts/start_weaviate.sh

# Weaviate will run on http://localhost:8080
```

### 4. Ingest HR Policies

```bash
# Run ingestion pipeline
uv run python -m chat_rag.ingestion

# This will:
# - Read policy documents from docs/
# - Split into chunks with section paths
# - Generate embeddings
# - Store in Weaviate
```

### 5. Start MCP Server

In a separate terminal:

```bash
# Start MCP server
uv run python -m chat_rag.mcp_server

# Server will run on http://localhost:9000
```

### 6. Start Chainlit App

In another terminal:

```bash
# Start Chainlit app
uv run chainlit run src/chat_rag/chainlit_app.py -w

# App will run on http://localhost:8000
```

## Usage

1. Open your browser to `http://localhost:8000`
2. Enter a mock email (e.g., `john.doe@company.com`)
3. Ask HR-related questions like:
   - "How many PTO days do I have?"
   - "What's the expense reimbursement policy?"
   - "What are the work from home rules?"

The assistant will:
- Retrieve your employee profile
- Search relevant policy chunks
- Provide answers with citations

## Project Structure

```
chat_rag/
├── docs/                      # HR policy documents (JSON)
├── ingested/
│   └── contracts/            # Ingestion metadata (optional)
├── scripts/
│   └── start_weaviate.sh     # Weaviate startup script
├── src/
│   ├── chainlit_app.py       # Chainlit chatbot UI
│   ├── agent.py              # LangChain agent with tools
│   ├── mcp_server.py         # FastMCP server
│   ├── ingestion.py          # Document ingestion pipeline
│   └── retriever.py          # Hybrid retriever
├── tests/                    # Pytest tests
├── requirements.txt          # Python dependencies
├── .env                      # Environment variables
└── README.md                 # This file
```

## MCP Tools

The agent has access to:

- `get_employee_profile(email: str)` - Get employee information
- `get_time_off_balance(email: str)` - Get remaining PTO days
- `search_policies(query: str)` - Search policy documents

## Development

### Running Tests

```bash
uv run pytest tests/ -v
```

### Adding New Policies

1. Add JSON file to `docs/` with structure:
```json
{
    "metadata": {
        "document_id": "pol_xxx",
        "document_name": "Policy Name",
        "topic": "Topic",
        "country": "CH",
        "active": true,
        "last_modified": "2024-01-01"
    },
    "content": "# Markdown content..."
}
```

2. Re-run ingestion:
```bash
uv run python -m chat_rag.ingestion
```

### Debugging

The application includes extensive logging. Check console output for:
- Agent reasoning steps
- Tool calls and responses
- Retrieval results
- LLM prompts and responses

## Troubleshooting

**Weaviate connection issues:**
```bash
# Check if Weaviate is running
curl http://localhost:8080/v1/.well-known/ready

# Restart Weaviate
docker restart weaviate
```

**MCP server connection issues:**
```bash
# Check if server is running
curl http://localhost:9000/health
```

**Missing embeddings:**
```bash
# Re-run ingestion
uv run python -m chat_rag.ingestion
```

## License

MIT
