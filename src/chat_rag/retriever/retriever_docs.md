# Retriever Module Documentation

The `src.chat_rag.retriever` module implements a hybrid retrieval system using Weaviate and LlamaIndex. It is designed to retrieve policy documents based on semantic search (vector embeddings) combined with metadata filtering.

## Overview

This module provides the core functionality for searching and retrieving relevant chunks of text from the knowledge base (Weaviate). It supports:
- **Hybrid Search**: Combining vector similarity with keyword matching (BM25) via Weaviate's hybrid search capabilities.
- **Metadata Filtering**: Strict filtering based on document properties (e.g., country, active status).
- **Context Awareness**: Option to retrieve adjacent chunks (previous and next) to provide better context for the LLM.

## Components

### 1. HybridRetriever (`hybrid.py`)

The main class responsible for interacting with the Weaviate vector store.

**Initialization:**
```python
retriever = HybridRetriever(
    weaviate_url="http://localhost:8080",
    collection_name="PolicyChunk",
    embedding_model="text-embedding-3-large",
    alpha=0.5
)
```
- `alpha`: Controls the balance between vector search and keyword search. 0.5 is a balanced approach.

**Key Methods:**
- `retrieve(query: str, top_k: int, filters: Dict, ...)`: Performs the search.
    - Automatically adds `active=True` filter if not present.
    - Logs the applied filters for debugging.
    - Returns a list of `PolicyChunk` objects.
- `close()`: Closes the Weaviate client connection.

### 2. PolicyChunk (`models.py`)

A dataclass representing a retrieved unit of text.

**Attributes:**
- `text`: The content of the chunk.
- `metadata`: `document_id`, `document_name`, `section_path_str`, `topic`, `country`, etc.
- `score`: The relevance score from Weaviate.
- `previous_chunk` / `next_chunk`: Optional context text.

### 3. Agent Integration (`tools.py`)

Helper functions to expose the retriever as a tool for the AI Agent.

- `create_retriever_tool(retriever)`: Wraps the retriever instance into a callable function `search_policies(query, country)` that can be bound to a LangChain or LlamaIndex agent. This function formats the output as a string suitable for LLM consumption.

## Usage Example

```python
from chat_rag.retriever import HybridRetriever

# Initialize
retriever = HybridRetriever()

# Retrieve
filters = {"country": "IT"}
results = retriever.retrieve("sick leave policy", top_k=3, filters=filters)

# Process results
for chunk in results:
    print(f"[{chunk.score}] {chunk.document_name}: {chunk.text[:50]}...")

# Cleanup
retriever.close()
```
