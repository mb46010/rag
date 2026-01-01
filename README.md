# HR Assistant Chatbot - RAG Demo v2.0

AI assistant that answers HR policy questions using RAG (Retrieval-Augmented Generation) and MCP (Model Context Protocol).

## What's New in v2.0

### 🚀 Performance Improvements
- **Workflow-based agent** replaces ReAct loop - 50%+ latency reduction
- **Parallel tool execution** for hybrid queries
- **Query-adaptive alpha** for optimal BM25/vector balance
- **Batch context fetching** - eliminates N+1 query pattern

### 📊 Better Retrieval
- **Calibrated reranking** with confidence scores
- **Optional RRF fusion** for stable ranking
- **Confidence-based filtering** with low-confidence warnings
- **Query type classification** (factual/conceptual/procedural)

### 🔍 Observability
- **Retrieval evaluation framework** with Recall@K, MRR metrics
- **Streaming progress updates** in chat UI
- **Enhanced logging** with query classification info

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        User Query                               │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Intent Classification                        │
│         (policy_only / personal_only / hybrid / chitchat)       │
└─────────────────────────────────────────────────────────────────┘
                              │
            ┌─────────────────┼─────────────────┐
            ▼                 ▼                 ▼
     ┌──────────┐      ┌──────────┐      ┌──────────┐
     │  Policy  │      │ Personal │      │  Hybrid  │
     │  Only    │      │  Only    │      │ (Parallel)│
     └────┬─────┘      └────┬─────┘      └────┬─────┘
          │                 │                 │
          ▼                 ▼                 ▼
    ┌───────────┐    ┌───────────┐    ┌───────────────┐
    │ RAG Search│    │ MCP Tools │    │ RAG + MCP     │
    │ (Enhanced)│    │           │    │ (Concurrent)  │
    └─────┬─────┘    └─────┬─────┘    └───────┬───────┘
          │                │                  │
          └────────────────┼──────────────────┘
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Response Synthesis                           │
│              (with citations and confidence)                    │
└─────────────────────────────────────────────────────────────────┘
```

## Tech Stack

- **Python 3.10+**
- **LangGraph** - Workflow orchestration (replaces ReAct)
- **LangChain** - LLM integration
- **OpenAI** - LLM (GPT-4o) and embeddings (text-embedding-3-large)
- **Chainlit** - Chat UI with streaming
- **FastMCP** - MCP server for employee data
- **Weaviate** - Vector store with hybrid search
- **sentence-transformers** - CrossEncoder reranking

## Quick Start

### 1. Setup Environment

```bash
uv lock
uv sync
```

### 2. Configure Environment

```bash
# Create .env file
echo "OPENAI_API_KEY=your_key_here" > .env
```

### 3. Start Services

```bash
# Terminal 1: Start Weaviate
bash scripts/start_weaviate.sh

# Terminal 2: Ingest documents
uv run python -m chat_rag.ingestion

# Terminal 3: Start MCP server
uv run python -m chat_rag.mcp_server

# Terminal 4: Start Chainlit
uv run chainlit run src/chat_rag/chainlit_app.py -w
```

### 4. Open Browser

Navigate to `http://localhost:8000`

## Project Structure

```
chat_rag/
├── src/chat_rag/
│   ├── agent/
│   │   ├── __init__.py
│   │   ├── config.py          # Agent configuration
│   │   ├── prompt.py          # System prompts
│   │   └── workflow.py        # LangGraph workflow agent ⭐
│   ├── retriever/
│   │   ├── __init__.py
│   │   ├── config.py          # Retrieval configuration
│   │   ├── models.py          # PolicyChunk, QueryType
│   │   ├── hybrid.py          # Enhanced hybrid retriever ⭐
│   │   ├── reranker.py        # CrossEncoder with calibration
│   │   └── tools.py           # LangChain tool wrappers
│   ├── ingestion/             # Document processing
│   ├── chainlit_app.py        # Chat UI with streaming
│   └── mcp_server.py          # Employee data MCP server
├── tests/
│   └── eval/
│       ├── retrieval_eval.py  # Evaluation framework ⭐
│       └── golden_retrieval.json
├── documents/                 # HR policy documents (JSON)
└── output/                    # Ingestion contracts
```

## Key Components

### Enhanced Hybrid Retriever

```python
from chat_rag.retriever import EnhancedHybridRetriever, RetrievalConfig

config = RetrievalConfig(
    enable_reranking=True,      # CrossEncoder reranking
    enable_context_window=True,  # Fetch adjacent chunks
    enable_rrf=False,           # Use standard hybrid (or RRF)
)

retriever = EnhancedHybridRetriever(config)

# Query-adaptive: alpha adjusts based on query type
results = retriever.retrieve(
    "How many PTO days do new employees get?",
    filters={"country": "CH"},
    top_k=3,
)

for chunk in results:
    print(f"[{chunk.confidence_level}] {chunk.document_name}: {chunk.text[:100]}...")
```

### Workflow Agent

```python
from chat_rag.agent import HRWorkflowAgent, AgentConfig

agent = HRWorkflowAgent(
    user_email="john.doe@company.com",
    config=AgentConfig(
        model_name="gpt-4o",
        enable_parallel_fetch=True,
    ),
)

# Streaming execution
async for event in agent.astream("How many PTO days do I have?"):
    if event["type"] == "progress":
        print(f"Step: {event['step']}")
    elif event["type"] == "complete":
        print(f"Response: {event['response']}")
```

### Retrieval Evaluation

```python
from tests.eval import RetrievalEvaluator
from pathlib import Path

evaluator = RetrievalEvaluator.from_file(
    retriever,
    Path("tests/eval/golden_retrieval.json")
)

# Full evaluation
metrics = evaluator.evaluate()
print(metrics)
# Recall@1: 0.850
# Recall@3: 0.950
# MRR: 0.892

# Evaluate by tag
factual_metrics = evaluator.evaluate_by_tag("factual")
```

## Configuration Options

### RetrievalConfig

| Option | Default | Description |
|--------|---------|-------------|
| `enable_reranking` | `True` | Use CrossEncoder reranking |
| `enable_rrf` | `False` | Use RRF instead of Weaviate hybrid |
| `alpha_factual` | `0.3` | Alpha for factual queries (more BM25) |
| `alpha_conceptual` | `0.7` | Alpha for conceptual queries (more vector) |
| `candidate_pool_multiplier` | `4` | Candidates = top_k × multiplier |

### AgentConfig

| Option | Default | Description |
|--------|---------|-------------|
| `model_name` | `gpt-4o` | LLM model |
| `enable_parallel_fetch` | `True` | Parallel MCP + RAG for hybrid |
| `max_retries` | `3` | MCP tool retry attempts |

## Development

### Running Tests

```bash
# Unit tests
uv run pytest tests/ -v

# Retrieval evaluation
uv run python -c "
from chat_rag.retriever import EnhancedHybridRetriever
from tests.eval import RetrievalEvaluator
from pathlib import Path

retriever = EnhancedHybridRetriever()
evaluator = RetrievalEvaluator.from_file(retriever, Path('tests/eval/golden_retrieval.json'))
print(evaluator.evaluate())
"
```

### Adding Test Cases

Edit `tests/eval/golden_retrieval.json`:

```json
{
  "query": "Your test query",
  "expected_doc_ids": ["pol_xxx"],
  "tags": ["factual", "topic"],
  "description": "What this tests"
}
```

## Troubleshooting

**Low retrieval scores?**
- Check if reranking is enabled
- Try adjusting alpha for your query type
- Verify documents are ingested

**Slow responses?**
- Enable parallel fetch in AgentConfig
- Reduce candidate_pool_multiplier
- Check MCP server latency

**Missing context?**
- Ensure `enable_context_window=True`
- Verify chunk_index is correct in Weaviate

## License

MIT
