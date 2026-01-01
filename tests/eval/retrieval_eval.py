"""
Retrieval Evaluation Framework.

Provides tools to measure retrieval quality against golden test cases.
Metrics include Recall@K, MRR, and Precision@K.
"""

import json
import logging
from dataclasses import dataclass, asdict, field
from typing import List, Dict, Optional, Any
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class RetrievalTestCase:
    """A test case for retrieval evaluation."""

    query: str
    expected_doc_ids: List[str]
    expected_sections: List[str] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)  # e.g., ["factual", "CH", "PTO"]
    description: str = ""


@dataclass
class RetrievalMetrics:
    """Evaluation metrics for retrieval quality."""

    recall_at_1: float = 0.0
    recall_at_3: float = 0.0
    recall_at_5: float = 0.0
    mrr: float = 0.0  # Mean Reciprocal Rank
    precision_at_3: float = 0.0
    total_cases: int = 0
    failed_cases: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return asdict(self)

    def __str__(self) -> str:
        return (
            f"Retrieval Metrics (n={self.total_cases}):\n"
            f"  Recall@1: {self.recall_at_1:.3f}\n"
            f"  Recall@3: {self.recall_at_3:.3f}\n"
            f"  Recall@5: {self.recall_at_5:.3f}\n"
            f"  MRR: {self.mrr:.3f}\n"
            f"  Precision@3: {self.precision_at_3:.3f}"
        )


@dataclass
class TestCaseResult:
    """Result of a single test case evaluation."""

    test_case: RetrievalTestCase
    retrieved_doc_ids: List[str]
    retrieved_scores: List[float]
    recall_at_1: bool
    recall_at_3: bool
    recall_at_5: bool
    reciprocal_rank: float
    precision_at_3: float


class RetrievalEvaluator:
    """
    Evaluate retrieval quality against golden test cases.

    Usage:
        evaluator = RetrievalEvaluator.from_file(retriever, "tests/eval/golden.json")
        metrics = evaluator.evaluate()
        print(metrics)
    """

    def __init__(self, retriever, test_cases: List[RetrievalTestCase]):
        """
        Initialize evaluator.

        Args:
            retriever: Retriever instance with retrieve() method
            test_cases: List of test cases to evaluate
        """
        self.retriever = retriever
        self.test_cases = test_cases

    def evaluate(
        self,
        filters: Optional[Dict[str, Any]] = None,
        top_k: int = 5,
    ) -> RetrievalMetrics:
        """
        Run evaluation and return aggregate metrics.

        Args:
            filters: Optional filters to apply to all queries
            top_k: Number of results to retrieve per query

        Returns:
            RetrievalMetrics with aggregate scores
        """
        if not self.test_cases:
            logger.warning("No test cases to evaluate")
            return RetrievalMetrics()

        metrics = RetrievalMetrics(total_cases=len(self.test_cases))
        results = []

        for case in self.test_cases:
            try:
                result = self._evaluate_case(case, filters, top_k)
                results.append(result)

                # Accumulate metrics
                if result.recall_at_1:
                    metrics.recall_at_1 += 1
                if result.recall_at_3:
                    metrics.recall_at_3 += 1
                if result.recall_at_5:
                    metrics.recall_at_5 += 1
                metrics.mrr += result.reciprocal_rank
                metrics.precision_at_3 += result.precision_at_3

            except Exception as e:
                logger.error(f"Error evaluating case '{case.query}': {e}")
                metrics.failed_cases.append(case.query)

        # Normalize metrics
        n = len(self.test_cases)
        if n > 0:
            metrics.recall_at_1 /= n
            metrics.recall_at_3 /= n
            metrics.recall_at_5 /= n
            metrics.mrr /= n
            metrics.precision_at_3 /= n

        return metrics

    def evaluate_by_tag(self, tag: str, **kwargs) -> RetrievalMetrics:
        """
        Evaluate only test cases with a specific tag.

        Args:
            tag: Tag to filter by (e.g., "factual", "PTO")
            **kwargs: Additional arguments passed to evaluate()

        Returns:
            RetrievalMetrics for filtered cases
        """
        filtered_cases = [c for c in self.test_cases if tag in c.tags]

        if not filtered_cases:
            raise ValueError(f"No test cases found with tag: {tag}")

        evaluator = RetrievalEvaluator(self.retriever, filtered_cases)
        return evaluator.evaluate(**kwargs)

    def evaluate_detailed(
        self,
        filters: Optional[Dict[str, Any]] = None,
        top_k: int = 5,
    ) -> List[TestCaseResult]:
        """
        Run evaluation and return detailed per-case results.

        Useful for debugging and identifying specific failure cases.
        """
        results = []

        for case in self.test_cases:
            try:
                result = self._evaluate_case(case, filters, top_k)
                results.append(result)
            except Exception as e:
                logger.error(f"Error evaluating case '{case.query}': {e}")

        return results

    def _evaluate_case(
        self,
        case: RetrievalTestCase,
        filters: Optional[Dict[str, Any]],
        top_k: int,
    ) -> TestCaseResult:
        """Evaluate a single test case."""
        # Retrieve results
        chunks = self.retriever.retrieve(
            case.query,
            top_k=top_k,
            filters=filters or {},
        )

        retrieved_doc_ids = [c.document_id for c in chunks]
        retrieved_scores = [c.score for c in chunks]

        # Calculate metrics
        recall_at_1 = any(
            doc_id in retrieved_doc_ids[:1] for doc_id in case.expected_doc_ids
        )
        recall_at_3 = any(
            doc_id in retrieved_doc_ids[:3] for doc_id in case.expected_doc_ids
        )
        recall_at_5 = any(
            doc_id in retrieved_doc_ids[:5] for doc_id in case.expected_doc_ids
        )

        # MRR: reciprocal of rank of first relevant result
        reciprocal_rank = 0.0
        for rank, doc_id in enumerate(retrieved_doc_ids, 1):
            if doc_id in case.expected_doc_ids:
                reciprocal_rank = 1 / rank
                break

        # Precision@3: fraction of top 3 that are relevant
        hits_at_3 = sum(
            1 for doc_id in retrieved_doc_ids[:3] if doc_id in case.expected_doc_ids
        )
        precision_at_3 = hits_at_3 / min(3, len(retrieved_doc_ids)) if retrieved_doc_ids else 0

        return TestCaseResult(
            test_case=case,
            retrieved_doc_ids=retrieved_doc_ids,
            retrieved_scores=retrieved_scores,
            recall_at_1=recall_at_1,
            recall_at_3=recall_at_3,
            recall_at_5=recall_at_5,
            reciprocal_rank=reciprocal_rank,
            precision_at_3=precision_at_3,
        )

    @classmethod
    def from_file(cls, retriever, filepath: Path) -> "RetrievalEvaluator":
        """
        Load test cases from JSON file.

        Expected format:
        {
            "test_cases": [
                {
                    "query": "How many PTO days?",
                    "expected_doc_ids": ["pol_pto"],
                    "expected_sections": ["Accrual"],
                    "tags": ["factual", "PTO"],
                    "description": "Basic PTO question"
                },
                ...
            ]
        }
        """
        with open(filepath) as f:
            data = json.load(f)

        test_cases = [RetrievalTestCase(**tc) for tc in data["test_cases"]]
        return cls(retriever, test_cases)

    @classmethod
    def from_dict(cls, retriever, data: Dict) -> "RetrievalEvaluator":
        """Load test cases from dictionary."""
        test_cases = [RetrievalTestCase(**tc) for tc in data["test_cases"]]
        return cls(retriever, test_cases)


def create_sample_test_file(filepath: Path):
    """Create a sample test cases file."""
    sample_data = {
        "test_cases": [
            {
                "query": "How many PTO days do new employees get?",
                "expected_doc_ids": ["pol_pto"],
                "expected_sections": ["Accrual"],
                "tags": ["factual", "PTO"],
                "description": "Basic PTO accrual question",
            },
            {
                "query": "What's the expense limit for meals during travel?",
                "expected_doc_ids": ["pol_expense"],
                "expected_sections": ["Travel Expenses", "Meal Expenses"],
                "tags": ["factual", "expenses"],
                "description": "Expense policy question",
            },
            {
                "query": "How does the work from home policy work?",
                "expected_doc_ids": ["pol_wfh"],
                "expected_sections": ["Overview", "Eligibility"],
                "tags": ["conceptual", "WFH"],
                "description": "WFH policy explanation",
            },
            {
                "query": "What are the steps to request parental leave?",
                "expected_doc_ids": ["pol_parental"],
                "expected_sections": ["Request Process"],
                "tags": ["procedural", "leave"],
                "description": "Parental leave process",
            },
            {
                "query": "Am I eligible for sabbatical leave?",
                "expected_doc_ids": ["pol_sabbatical", "pol_leave"],
                "expected_sections": ["Eligibility"],
                "tags": ["hybrid", "leave"],
                "description": "Sabbatical eligibility (needs tenure check)",
            },
        ]
    }

    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, "w") as f:
        json.dump(sample_data, f, indent=2)

    logger.info(f"Created sample test file: {filepath}")


if __name__ == "__main__":
    # Create sample test file when run directly
    create_sample_test_file(Path("tests/eval/golden_retrieval.json"))
