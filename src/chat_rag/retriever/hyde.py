"""
HyDE: Hypothetical Document Embeddings

Improves retrieval recall on vague or short queries by:
1. Generating a hypothetical answer using the LLM
2. Using that answer's embedding for retrieval
3. Finding documents similar to what a good answer would look like

Reference: https://arxiv.org/abs/2212.10496
"""

import logging
from typing import List, Dict, Any, Optional
from langfuse import observe
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

from .hybrid import EnhancedHybridRetriever
from .models import PolicyChunk, QueryType
from .config import RetrievalConfig

logger = logging.getLogger(__name__)


HYDE_SYSTEM_PROMPT = """You are an HR policy expert. Given an employee question, write a short excerpt (2-3 sentences) from an HR policy document that would directly answer this question.

Write as if you're quoting from an actual policy document - be specific, factual, and use formal policy language. Do not include any preamble or explanation."""

HYDE_EXAMPLES = [
    {
        "query": "How many vacation days do I get?",
        "hypothetical": "Employees are entitled to 25 days of paid annual leave per calendar year. Leave accrues at a rate of 2.08 days per month of service. Unused leave may be carried over to the following year, up to a maximum of 5 days.",
    },
    {
        "query": "Can I work from home?",
        "hypothetical": "Eligible employees may work remotely up to 2 days per week with manager approval. Remote work arrangements must be documented in the HR system. Employees must maintain core hours of 10:00-15:00 for team collaboration.",
    },
]


class HyDERetriever:
    """
    Retriever with Hypothetical Document Embeddings.

    For vague queries, generates a hypothetical answer first,
    then retrieves documents similar to that answer.
    """

    def __init__(
        self,
        base_retriever: EnhancedHybridRetriever,
        llm: ChatOpenAI = None,
        enable_for_all: bool = False,
        vague_query_threshold: int = 5,  # words
    ):
        """
        Args:
            base_retriever: The underlying retriever
            llm: LLM for generating hypotheticals (defaults to gpt-4o-mini)
            enable_for_all: If True, use HyDE for all queries
            vague_query_threshold: Use HyDE for queries with fewer words
        """
        self.base_retriever = base_retriever
        self.llm = llm or ChatOpenAI(model="gpt-4o-mini", temperature=0.7)
        self.enable_for_all = enable_for_all
        self.vague_query_threshold = vague_query_threshold

    def _should_use_hyde(self, query: str) -> bool:
        """Determine if HyDE would help this query."""
        if self.enable_for_all:
            return True

        # Short queries benefit most from HyDE
        word_count = len(query.split())
        if word_count <= self.vague_query_threshold:
            return True

        # Questions without specific keywords
        specific_indicators = ["how many", "what is the", "limit", "deadline", "rate"]
        if not any(ind in query.lower() for ind in specific_indicators):
            return True

        return False

    @observe(as_type="generation")
    async def _generate_hypothetical(self, query: str) -> str:
        """Generate a hypothetical policy excerpt that would answer the query."""
        # Build few-shot prompt
        examples_text = "\n\n".join(
            [f"Question: {ex['query']}\nPolicy excerpt: {ex['hypothetical']}" for ex in HYDE_EXAMPLES]
        )

        prompt = f"""{examples_text}

Question: {query}
Policy excerpt:"""

        response = await self.llm.ainvoke(
            [
                SystemMessage(content=HYDE_SYSTEM_PROMPT),
                HumanMessage(content=prompt),
            ]
        )

        hypothetical = response.content.strip()
        logger.info(f"Generated hypothetical ({len(hypothetical)} chars) for: {query[:50]}...")

        return hypothetical

    @observe(as_type="generation")
    async def aretrieve(
        self,
        query: str,
        top_k: int = 5,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[PolicyChunk]:
        """
        Retrieve with optional HyDE enhancement.

        Args:
            query: User's question
            top_k: Number of results
            filters: Metadata filters

        Returns:
            List of PolicyChunk with hyde_enhanced flag
        """
        use_hyde = self._should_use_hyde(query)

        if use_hyde:
            # Generate hypothetical answer
            hypothetical = await self._generate_hypothetical(query)

            # Combine original query with hypothetical for better embedding
            enhanced_query = f"{query}\n\nRelevant policy: {hypothetical}"

            logger.info(f"Using HyDE-enhanced retrieval for: {query[:50]}...")
            chunks = self.base_retriever.retrieve(
                enhanced_query,
                top_k=top_k,
                filters=filters,
            )

            # Mark chunks as HyDE-enhanced
            for chunk in chunks:
                chunk.hyde_enhanced = True
                chunk.hypothetical_used = hypothetical[:200]
        else:
            chunks = self.base_retriever.retrieve(
                query,
                top_k=top_k,
                filters=filters,
            )
            for chunk in chunks:
                chunk.hyde_enhanced = False

        return chunks

    @observe(as_type="generation")
    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[PolicyChunk]:
        """Sync version - uses base retriever without HyDE."""
        logger.warning("Sync retrieve() called - HyDE requires async. Using base retriever.")
        return self.base_retriever.retrieve(query, top_k=top_k, filters=filters)

    def close(self):
        """Close underlying retriever."""
        self.base_retriever.close()
