# Changelog

All notable changes to the HR Assistant Chatbot are documented here.

## [2.0.0] - 2024-XX-XX

### Added

#### Retriever Improvements
- **Query-adaptive alpha**: Automatically adjusts BM25/vector balance based on query type
  - Factual queries (α=0.3): favor BM25 for specific facts, numbers
  - Conceptual queries (α=0.7): favor vector for explanations
  - Procedural queries (α=0.5): balanced for how-to questions
- **Calibrated reranking**: CrossEncoder scores converted to 0-1 confidence via sigmoid
- **Confidence levels**: Chunks tagged as high/medium/low confidence
- **Optional RRF fusion**: Reciprocal Rank Fusion as alternative to score-based hybrid
- **Batch context fetching**: Single query fetches all adjacent chunks (fixes N+1)
- **RetrievalConfig**: Centralized configuration for all retrieval settings
- **RetrievalResult**: Rich result object with query metadata

#### Agent Improvements
- **Workflow-based architecture**: LangGraph workflow replaces ReAct loop
  - 2-3 LLM calls instead of 4-5
  - ~50% latency reduction
- **Intent classification**: Routes queries to optimal execution path
  - `policy_only`: RAG only
  - `personal_only`: MCP tools only
  - `hybrid`: Parallel execution
  - `chitchat`/`out_of_scope`: Direct response
- **Parallel tool execution**: MCP and RAG run concurrently for hybrid queries
- **Streaming progress**: Real-time step updates in Chainlit UI
- **Retry logic**: Tenacity-based retries for MCP tool calls
- **AgentConfig**: Centralized configuration

#### Evaluation Framework
- **RetrievalEvaluator**: Measure retrieval quality against golden test cases
- **Metrics**: Recall@1/3/5, MRR, Precision@3
- **Tag-based evaluation**: Filter test cases by tags (factual, PTO, etc.)
- **Golden test cases**: Sample test file with HR policy queries

#### Observability
- **Enhanced logging**: Query type, alpha used, confidence levels
- **Langfuse integration**: Ready for production tracing
- **Intermediate steps**: Formatted for Chainlit display

### Changed
- `HybridRetriever` → `EnhancedHybridRetriever` (alias kept for compatibility)
- `HRAssistantAgent` → `HRWorkflowAgent` (alias kept for compatibility)
- Chainlit app uses streaming by default
- Reranker enabled by default

### Fixed
- N+1 query pattern for context windows
- Score normalization across BM25 and vector results
- Low-confidence results returned without warning

### Migration Guide

#### Retriever

```python
# Before
from chat_rag.retriever import HybridRetriever
retriever = HybridRetriever()
results = retriever.retrieve(query, country="CH")

# After
from chat_rag.retriever import EnhancedHybridRetriever, RetrievalConfig
config = RetrievalConfig(enable_reranking=True)
retriever = EnhancedHybridRetriever(config)
results = retriever.retrieve(query, filters={"country": "CH"})

# Access new features
for chunk in results:
    print(f"Confidence: {chunk.confidence_level}")
```

#### Agent

```python
# Before
from chat_rag.agent import HRAssistantAgent
agent = HRAssistantAgent(user_email="...")
result = await agent.arun("query")

# After (same interface, new implementation)
from chat_rag.agent import HRWorkflowAgent, AgentConfig
config = AgentConfig(enable_parallel_fetch=True)
agent = HRWorkflowAgent(user_email="...", config=config)

# Streaming support
async for event in agent.astream("query"):
    print(event)
```

## [1.0.0] - Initial Release

- Basic hybrid retriever with Weaviate
- ReAct agent with MCP tools
- Chainlit chat interface
- Document ingestion pipeline
