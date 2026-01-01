# HyDE: Hypothetical Document Embeddings

The `HyDERetriever` in `src/chat_rag/retriever/hyde.py` implements the **Hypothetical Document Embeddings** (HyDE) technique. This method is particularly useful for improving retrieval recall on vague, short, or underspecified queries.

## 1. How HyDE Works

Standard vector retrieval searches for document chunks that are semantically similar to the user's *question*. However, questions and answers often live in different areas of the embedding space.

HyDE flips this by:
1.  **Generating a hypothetical answer** (or document excerpt) using an LLM.
2.  **Using that hypothetical answer's embedding** for retrieval.
3.  **Finding actual documents** that are similar to what a "good answer" would look like.

In our implementation, we take a hybrid approach by combining the original query with the hypothetical answer for the final retrieval.

## 2. Implementation: `HyDERetriever`

The `HyDERetriever` acts as a wrapper around the `EnhancedHybridRetriever`.

### Heuristic-Based Activation
Unlike many implementations that apply HyDE to every query, our version uses built-in heuristics to save on LLM latency and costs. It only triggers when:
-   The query is **short** (word count <= `vague_query_threshold`, default 5).
-   The query **lacks specific indicators** (e.g., "how many", "limit", "deadline").

This ensures that precise queries (which already work well with standard RAG) are not slowed down by unnecessary LLM calls.

### Custom HR Policy Prompting
The hypothetical generation is tailored for the HR domain using a specific system prompt and few-shot examples:
-   **System Prompt**: Instructs the LLM to act as an HR expert and quote directly from a formal policy document.
-   **Few-Shot Examples**: Provide the model with the expected tone and structure (e.g., vacation days, remote work policies).

### Query Enhancement
When triggered, the final query sent to the base retriever is:
```text
{original_query}

Relevant policy: {hypothetical_answer}
```

## 3. Comparison with LlamaIndex `HyDEQueryTransform`

The following table compares our custom implementation with the standard LlamaIndex approach:

| Feature | Custom `HyDERetriever` | LlamaIndex `HyDEQueryTransform` |
| :--- | :--- | :--- |
| **Logic Placement** | Wrapped around specific retriever. | Applied via `TransformQueryEngine` or `QueryBundle`. |
| **Trigger Mechanism** | **Dynamic**: Uses heuristics (word count, keywords) to decide when to run. | **Static**: Usually runs for all queries passed through the transform. |
| **Domain Adaptation** | Hardcoded HR-specific prompts and few-shot examples. | Custom prompts can be provided via `hyde_prompt`. |
| **Integration** | Direct access to `EnhancedHybridRetriever` filters and metadata. | Modular part of the LlamaIndex pipeline. |
| **Metadata Tracking** | Marks results with `hyde_enhanced=True` and stores the hypothetical string. | Harder to track which specific results were HyDE-influenced in standard flows. |
| **Sync/Async** | Async-first for generation; falls back to base retriever for sync. | Supports both depending on the query engine used. |

### Why we chose a custom implementation:
1.  **Efficiency**: By only running HyDE for "vague" queries, we significantly reduce average latency and OpenAI API costs.
2.  **Precision**: We can control exactly how the hypothetical answer is merged with the original query (e.g., appending with specific headers).
3.  **Observability**: We can easily flag and log when HyDE was used, which is critical for debugging retrieval issues in a production-like HR bot.
