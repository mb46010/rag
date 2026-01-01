"""
Langfuse Integration for Production Logging & Retroactive Labeling

Enables:
1. Logging all production queries with full traces
2. Exporting traces for retroactive labeling
3. Converting labeled traces to evaluation sets
4. Tracking evaluation metrics over time

This is the "Option 3" approach that builds real eval data from production.
"""

import logging
from typing import List, Dict, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import json

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────
# Note: This requires langfuse to be installed and configured
# pip install langfuse
# Set LANGFUSE_SECRET_KEY, LANGFUSE_PUBLIC_KEY, LANGFUSE_HOST
# ─────────────────────────────────────────────────────────────────


class RelevanceLabel(Enum):
    """Labels for retroactive relevance judgment."""
    HIGHLY_RELEVANT = 3
    RELEVANT = 2
    PARTIALLY_RELEVANT = 1
    NOT_RELEVANT = 0


class AnswerQualityLabel(Enum):
    """Labels for answer quality judgment."""
    EXCELLENT = 5
    GOOD = 4
    ACCEPTABLE = 3
    POOR = 2
    WRONG = 1


@dataclass
class ProductionTrace:
    """A logged production trace for labeling."""
    trace_id: str
    timestamp: datetime
    user_query: str
    resolved_query: Optional[str]
    
    # Retrieval info
    retrieved_chunks: List[Dict]  # [{chunk_id, text, score}]
    retrieval_latency_ms: float
    
    # Generation info
    generated_answer: str
    generation_latency_ms: float
    
    # Context
    user_country: Optional[str]
    intent_classified: Optional[str]
    hyde_used: bool = False
    
    # Labels (filled in during retroactive labeling)
    relevance_labels: Dict[str, int] = field(default_factory=dict)  # chunk_id -> label
    answer_quality: Optional[int] = None
    labeler_notes: str = ""
    labeled_at: Optional[datetime] = None
    labeled_by: Optional[str] = None


class LangfuseEvalIntegration:
    """
    Integration with Langfuse for logging and evaluation.
    
    Usage:
        # During inference (automatic via callbacks)
        langfuse_handler = CallbackHandler()
        result = await agent.arun(query, callbacks=[langfuse_handler])
        
        # Later: export for labeling
        integration = LangfuseEvalIntegration()
        traces = integration.export_traces_for_labeling(last_n_days=7)
        integration.save_for_labeling(traces, "traces_to_label.json")
        
        # After labeling: convert to eval set
        labeled = integration.load_labeled("traces_labeled.json")
        integration.convert_to_eval_set(labeled, "golden_from_prod.json")
    """
    
    def __init__(self):
        """Initialize Langfuse client."""
        try:
            from langfuse import Langfuse
            self.client = Langfuse()
            logger.info("Langfuse client initialized")
        except ImportError:
            logger.warning("Langfuse not installed. Run: pip install langfuse")
            self.client = None
        except Exception as e:
            logger.warning(f"Langfuse initialization failed: {e}")
            self.client = None
    
    def export_traces_for_labeling(
        self,
        last_n_days: int = 7,
        limit: int = 100,
        min_chunks_retrieved: int = 1,
        sample_strategy: str = "diverse",  # diverse, random, low_confidence
    ) -> List[ProductionTrace]:
        """
        Export recent traces for retroactive labeling.
        
        Args:
            last_n_days: Look back period
            limit: Max traces to export
            min_chunks_retrieved: Filter out queries with no retrieval
            sample_strategy: How to sample traces
                - diverse: Sample across different intents/topics
                - random: Random sample
                - low_confidence: Prioritize low-confidence retrievals
        
        Returns:
            List of ProductionTrace objects ready for labeling
        """
        if not self.client:
            logger.error("Langfuse client not available")
            return []
        
        # Calculate date range
        end_time = datetime.now()
        start_time = end_time - timedelta(days=last_n_days)
        
        # Fetch traces from Langfuse
        # Note: Actual API may differ - this is scaffolding
        try:
            traces_response = self.client.fetch_traces(
                # from_timestamp=start_time,
                # to_timestamp=end_time,
                limit=limit * 2,  # Fetch more for filtering
            )
            
            traces = []
            for trace_data in traces_response.data:
                trace = self._parse_trace(trace_data)
                if trace and len(trace.retrieved_chunks) >= min_chunks_retrieved:
                    traces.append(trace)
            
            # Apply sampling strategy
            sampled = self._apply_sampling(traces, limit, sample_strategy)
            
            logger.info(f"Exported {len(sampled)} traces for labeling")
            return sampled
            
        except Exception as e:
            logger.error(f"Failed to fetch traces: {e}")
            return []
    
    def _parse_trace(self, trace_data) -> Optional[ProductionTrace]:
        """Parse Langfuse trace into ProductionTrace object."""
        try:
            # Extract relevant info from trace
            # This depends on how you structured your traces
            
            # Get observations (spans) from trace
            observations = trace_data.observations or []
            
            # Find retrieval span
            retrieval_obs = next(
                (o for o in observations if "retriev" in (o.name or "").lower()),
                None
            )
            
            # Find generation span
            generation_obs = next(
                (o for o in observations if "generat" in (o.name or "").lower()),
                None
            )
            
            # Extract chunks from retrieval output
            chunks = []
            if retrieval_obs and retrieval_obs.output:
                # Parse chunks from output (structure depends on your implementation)
                chunks_data = retrieval_obs.output.get("chunks", [])
                for c in chunks_data:
                    chunks.append({
                        "chunk_id": c.get("chunk_id", ""),
                        "text": c.get("text", "")[:500],  # Truncate for labeling
                        "score": c.get("score", 0),
                        "source": c.get("source", ""),
                    })
            
            return ProductionTrace(
                trace_id=trace_data.id,
                timestamp=trace_data.timestamp or datetime.now(),
                user_query=trace_data.input.get("query", "") if trace_data.input else "",
                resolved_query=trace_data.input.get("resolved_query") if trace_data.input else None,
                retrieved_chunks=chunks,
                retrieval_latency_ms=retrieval_obs.latency if retrieval_obs else 0,
                generated_answer=trace_data.output.get("answer", "") if trace_data.output else "",
                generation_latency_ms=generation_obs.latency if generation_obs else 0,
                user_country=trace_data.metadata.get("country") if trace_data.metadata else None,
                intent_classified=trace_data.metadata.get("intent") if trace_data.metadata else None,
                hyde_used=trace_data.metadata.get("hyde_used", False) if trace_data.metadata else False,
            )
            
        except Exception as e:
            logger.warning(f"Failed to parse trace: {e}")
            return None
    
    def _apply_sampling(
        self,
        traces: List[ProductionTrace],
        limit: int,
        strategy: str,
    ) -> List[ProductionTrace]:
        """Apply sampling strategy to traces."""
        if len(traces) <= limit:
            return traces
        
        if strategy == "random":
            import random
            return random.sample(traces, limit)
        
        elif strategy == "low_confidence":
            # Sort by lowest retrieval score
            sorted_traces = sorted(
                traces,
                key=lambda t: min((c["score"] for c in t.retrieved_chunks), default=0)
            )
            return sorted_traces[:limit]
        
        elif strategy == "diverse":
            # Sample across different intents
            by_intent = {}
            for trace in traces:
                intent = trace.intent_classified or "unknown"
                by_intent.setdefault(intent, []).append(trace)
            
            sampled = []
            while len(sampled) < limit and any(by_intent.values()):
                for intent, intent_traces in list(by_intent.items()):
                    if intent_traces and len(sampled) < limit:
                        sampled.append(intent_traces.pop(0))
            
            return sampled
        
        return traces[:limit]
    
    def save_for_labeling(self, traces: List[ProductionTrace], filepath: str):
        """
        Save traces to JSON for labeling.
        
        The output format is designed for easy labeling:
        - Each trace shows query, chunks, and answer
        - Labeler fills in relevance_labels and answer_quality
        """
        labeling_data = {
            "metadata": {
                "exported_at": datetime.now().isoformat(),
                "trace_count": len(traces),
                "instructions": """
                    For each trace:
                    1. Read the user_query
                    2. For each retrieved chunk, rate relevance:
                       3=highly relevant, 2=relevant, 1=partial, 0=not relevant
                    3. Rate the generated answer quality (1-5)
                    4. Add any notes about issues
                    5. Set labeled=true when done
                """,
            },
            "traces": [],
        }
        
        for trace in traces:
            trace_dict = {
                "trace_id": trace.trace_id,
                "timestamp": trace.timestamp.isoformat(),
                "user_query": trace.user_query,
                "resolved_query": trace.resolved_query,
                "intent": trace.intent_classified,
                "chunks": [
                    {
                        "chunk_id": c["chunk_id"],
                        "source": c.get("source", ""),
                        "text_preview": c["text"][:300],
                        "retrieval_score": c["score"],
                        "relevance_label": None,  # TO BE FILLED
                    }
                    for c in trace.retrieved_chunks
                ],
                "generated_answer": trace.generated_answer,
                "answer_quality": None,  # TO BE FILLED (1-5)
                "labeler_notes": "",
                "labeled": False,
            }
            labeling_data["traces"].append(trace_dict)
        
        with open(filepath, "w") as f:
            json.dump(labeling_data, f, indent=2)
        
        logger.info(f"Saved {len(traces)} traces for labeling to {filepath}")
    
    def load_labeled(self, filepath: str) -> List[ProductionTrace]:
        """Load labeled traces from JSON."""
        with open(filepath) as f:
            data = json.load(f)
        
        traces = []
        for t in data.get("traces", []):
            if not t.get("labeled", False):
                continue
            
            trace = ProductionTrace(
                trace_id=t["trace_id"],
                timestamp=datetime.fromisoformat(t["timestamp"]),
                user_query=t["user_query"],
                resolved_query=t.get("resolved_query"),
                retrieved_chunks=[
                    {
                        "chunk_id": c["chunk_id"],
                        "text": c.get("text_preview", ""),
                        "score": c.get("retrieval_score", 0),
                        "source": c.get("source", ""),
                    }
                    for c in t["chunks"]
                ],
                retrieval_latency_ms=0,
                generated_answer=t["generated_answer"],
                generation_latency_ms=0,
                intent_classified=t.get("intent"),
                relevance_labels={
                    c["chunk_id"]: c["relevance_label"]
                    for c in t["chunks"]
                    if c.get("relevance_label") is not None
                },
                answer_quality=t.get("answer_quality"),
                labeler_notes=t.get("labeler_notes", ""),
                labeled_at=datetime.now(),
            )
            traces.append(trace)
        
        logger.info(f"Loaded {len(traces)} labeled traces")
        return traces
    
    def convert_to_eval_set(
        self,
        labeled_traces: List[ProductionTrace],
        output_path: str,
        min_relevance: int = 2,  # Minimum label to consider "relevant"
    ):
        """
        Convert labeled production traces to evaluation set.
        
        Args:
            labeled_traces: Traces with relevance labels
            output_path: Where to save the eval set
            min_relevance: Minimum relevance score to count as expected
        """
        test_cases = []
        
        for trace in labeled_traces:
            # Find chunks that were labeled as relevant
            relevant_chunks = [
                chunk_id
                for chunk_id, label in trace.relevance_labels.items()
                if label >= min_relevance
            ]
            
            if not relevant_chunks:
                continue  # Skip if no relevant chunks identified
            
            # Build tags from trace metadata
            tags = ["production"]
            if trace.intent_classified:
                tags.append(trace.intent_classified)
            if trace.user_country:
                tags.append(trace.user_country)
            if trace.hyde_used:
                tags.append("hyde_used")
            
            test_case = {
                "query": trace.user_query,
                "expected_chunk_ids": relevant_chunks,
                "expected_doc_ids": list(set(
                    c.get("source", "").split(" > ")[0]
                    for c in trace.retrieved_chunks
                    if c["chunk_id"] in relevant_chunks
                )),
                "tags": tags,
                "description": f"From production trace {trace.trace_id}",
                "answer_quality_baseline": trace.answer_quality,
            }
            test_cases.append(test_case)
        
        output = {
            "description": "Evaluation set from labeled production traces",
            "version": "1.0",
            "created_from": "langfuse_traces",
            "test_cases": test_cases,
        }
        
        with open(output_path, "w") as f:
            json.dump(output, f, indent=2)
        
        logger.info(f"Created eval set with {len(test_cases)} cases from production")
    
    def log_eval_results_to_langfuse(
        self,
        eval_name: str,
        metrics: Dict[str, float],
        config_snapshot: Dict[str, Any] = None,
    ):
        """
        Log evaluation results back to Langfuse for tracking over time.
        
        This creates a "score" that can be tracked across runs.
        """
        if not self.client:
            return
        
        try:
            # Create a trace for the eval run
            trace = self.client.trace(
                name=f"eval_run_{eval_name}",
                metadata={
                    "eval_name": eval_name,
                    "config": config_snapshot or {},
                },
            )
            
            # Log each metric as a score
            for metric_name, value in metrics.items():
                self.client.score(
                    trace_id=trace.id,
                    name=metric_name,
                    value=value,
                )
            
            logger.info(f"Logged eval results to Langfuse: {eval_name}")
            
        except Exception as e:
            logger.error(f"Failed to log to Langfuse: {e}")


# ─────────────────────────────────────────────────────────────────
# Langfuse Callback for Automatic Logging
# ─────────────────────────────────────────────────────────────────

def create_langfuse_callback(
    user_email: str,
    session_id: str = None,
    tags: List[str] = None,
):
    """
    Create a Langfuse callback handler for automatic trace logging.
    
    Usage:
        callback = create_langfuse_callback("user@company.com")
        result = await agent.arun(query, callbacks=[callback])
    """
    try:
        from langfuse.callback import CallbackHandler
        
        return CallbackHandler(
            user_id=user_email,
            session_id=session_id,
            tags=tags or ["hr_assistant"],
            metadata={"source": "hr_chatbot"},
        )
    except ImportError:
        logger.warning("Langfuse not available")
        return None
