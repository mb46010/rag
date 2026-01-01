# Hybrid Retriever Documentation

The `EnhancedHybridRetriever` is a production-grade retrieval system designed to balance keyword precision with semantic depth. It implements a multi-stage pipeline including query classification, adaptive search strategies, and calibrated reranking.

## 🚀 Key Strategies

### 1. Query-Adaptive Alpha (Default)
The retriever classifies every incoming query to adjust the balance between **Keyword (BM25)** and **Vector (Semantic)** search.

| Query Type | Description | Alpha | Bias |
| :--- | :--- | :--- | :--- |
| **Factual** | Specific names, dates, amounts, or limits. | 0.3 | Keyword-heavy |
| **Conceptual** | High-level explanations, themes, or "what is" questions. | 0.7 | Semantic-heavy |
| **Procedural** | "How-to" steps, processes, or registration details. | 0.5 | Balanced |
| **Default** | General or unknown query patterns. | 0.5 | Balanced |

*Lower alpha (e.g., 0.3) prioritizes exact keyword matching. Higher alpha (e.g., 0.7) prioritizes semantic similarity.*

### 2. Retrieval Methods
The system supports two methods for merging results:

*   **Standard Hybrid (Default):** Uses Weaviate's built-in score-based fusion. It is highly efficient and executes in a single database round-trip.
*   **Reciprocal Rank Fusion (RRF):** Enabled via `enable_rrf: True`. It executes separate BM25 and Vector queries, then merges them based on their rank positions rather than raw scores. This is more stable when score distributions are inconsistent.

### 3. Cross-Encoder Reranking
Once candidates are retrieved (default fetches top 20), they are passed through a `CrossEncoder` model.
*   **Calibration:** Raw scores are passed through a sigmoid function to produce a 0-1 confidence score.
*   **Confidence Levels:** Results are tagged as `high`, `medium`, or `low` confidence based on these calibrated scores.

### 4. Smart Context Fetching
To reduce LLM hallucinations, the retriever automatically fetches adjacent chunks (`previous_chunk` and `next_chunk`) using a single batch query to avoid the N+1 latency problem.

---

## ⚙️ Defaults

| Feature | Default Setting | Rationale |
| :--- | :--- | :--- |
| **Reranking** | `Enabled` | Drastically improves precision for RAG. |
| **Context Window** | `Enabled` | Provides LLM with necessary surrounding context. |
| **RRF (Reciprocal Rank Fusion)** | `Disabled` | Native Weaviate hybrid is faster and usually sufficient. |
| **Top K** | `5` | Balanced context length for LLM inputs. |
| **Candidate Pool** | `20` | Sufficient pool for effective reranking. |

---

## ⚖️ Pros & Cons

### Strategy Comparison

| Strategy | Pros | Cons |
| :--- | :--- | :--- |
| **Adaptive Alpha** | Automatically optimizes search based on user intent (e.g., finding a "claim limit" vs. "explaining a policy"). | Performance depends on the quality of the keyword-based classifier. |
| **Standard Hybrid** | Single query latency; very fast; native database optimization. | Score fusion can sometimes be biased if vector and BM25 scores have wildly different scales. |
| **RRF** | Very robust; does not require score normalization; merges disparate sources cleanly. | Slower (2+ queries); ignores the magnitude of the relevance score (only uses rank). |
| **Cross-Encoder Reranking** | State-of-the-art accuracy; provides reliable confidence scores for filtering. | Adds 50-200ms of latency depending on hardware (CPU/GPU) and model size. |

## 🛠 Configuration
Configuration is managed via the `RetrievalConfig` class in `src/chat_rag/retriever/config.py`. Key parameters include `alpha_*` values, `enable_reranking`, and `high_confidence_threshold`.


# Potential Improvements

## 🛠 Future Enhancements & Production Roadmap

### 1. Robust Alpha Selection
*   **Soft Classification:** Instead of hard-coding alpha based on a single class, transition to a probability-weighted alpha:
    `alpha = Σ (p(class) * alpha_class)`
    This prevents recall loss when queries share factual and conceptual characteristics.
*   **Fail-safe Mechanism:** If initial retrieval yields low reranker confidence or weak result diversity, automatically retry with an alternate strategy (e.g., RRF or a flipped alpha bias) and merge results.

### 2. Rigorous Score Calibration
*   **Beyond Sigmoid:** Sigmoid normalization is a mathematical mapping, not statistical calibration. 
*   **Probabilistic Scaling:** To provide true confidence levels, implement temperature scaling or isotonic regression using a labeled validation set (200-500 query-chunk pairs).
*   **Score Bands:** Until formal calibration is implemented, refer to levels as "Score Bands" (High/Medium/Low) defined empirically per model and corpus.

### 3. Observability & Diagnostics
*   **Structured Tracing:** Every retrieval should emit a diagnostic trace for monitoring and debugging:
    *   Query classification probabilities.
    *   Parameters used (Alpha, K, candidate pool size).
    *   Retrieval overlap (Intersection of BM25 and Vector results).
    *   Reranker score distribution (Top-1 score, score gap between Top-1 and Top-2, and entropy).
*   **Fusion Monitoring:** Log native Weaviate hybrid fusion metrics to detect behavior drift after database or embedding model upgrades.

### 4. Adaptive Retrieval Pipelines
*   **Dynamic Candidate Pools:** Scale the candidate pool multiplier dynamically. Increase the pool (e.g., 20 → 100) if:
    *   Vector and BM25 results show low overlap.
    *   The query is identified as short or ambiguous.
    *   The initial max reranker score falls below a critical threshold.
*   **Two-Shot Fallback:** Implement a low-latency "Quick Hybrid" pass, followed by a "Deep RRF" pass only when confidence signals are weak.

### 5. Advanced Context Management
*   **Structure-Aware Windows:** Replace blind neighbor fetching with logic that respects document boundaries:
    *   Validate that neighbors share the same section/document structure.
    *   Avoid fetching context for "header-only" or "index" chunks.
    *   Prioritize fetching the parent section or summary instead of arbitrary adjacent text.

### 6. Evaluation & Governance
*   **Offline Evaluation Harness:** Maintain a benchmark to measure:
    *   **Recall@K:** Effectiveness of the initial retrieval stage.
    *   **nDCG / MRR:** Accuracy of the reranking stage.
    *   **Latency Breakdown:** Per-stage performance tracking.
*   **Agent-Controlled Cascades:** When an agent orchestrates retrieval, it should select from fixed "Retrieval Plans" (e.g., *Standard*, *Deep Search*, *Keyword-Only*) rather than tuning individual parameters. This ensures predictable latency and prevents overfitting to specific queries.