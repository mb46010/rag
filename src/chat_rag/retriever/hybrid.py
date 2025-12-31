import logging
import weaviate
from typing import List, Dict, Any, Optional
from weaviate.classes.query import MetadataQuery
from llama_index.core import VectorStoreIndex
from llama_index.vector_stores.weaviate import WeaviateVectorStore
from llama_index.core.schema import NodeWithScore, TextNode
from llama_index.embeddings.openai import OpenAIEmbedding

from .models import PolicyChunk
from .reranker import Reranker


logger = logging.getLogger(__name__)


class HybridRetriever:
    """
    Hybrid retriever using Weaviate and LlamaIndex.

    Performs hybrid search (BM25 + Vector) with strict metadata filtering.
    """

    def __init__(
        self,
        weaviate_url: str = "http://localhost:8080",
        collection_name: str = "PolicyChunk",
        embedding_model: str = "text-embedding-3-large",
        alpha: float = 0.5,
        reranker_model: Optional[str] = None,
    ):
        """
        Initialize the hybrid retriever.

        Args:
            weaviate_url: Weaviate server URL
            collection_name: Weaviate collection name
            embedding_model: OpenAI embedding model name
            alpha: Hybrid search weight (0=BM25 only, 1=vector only, 0.5=balanced)
        """
        self.weaviate_url = weaviate_url
        self.collection_name = collection_name
        self.alpha = alpha
        self.reranker = Reranker(model_name=reranker_model) if reranker_model else None

        logger.info(f"Initializing HybridRetriever with collection: {collection_name}")
        logger.info(f"Weaviate URL: {weaviate_url}, alpha: {alpha}")

        # Initialize Weaviate client
        self.client = weaviate.connect_to_local(host=weaviate_url.replace("http://", "").split(":")[0])

        # Initialize embedding model
        self.embed_model = OpenAIEmbedding(model=embedding_model)
        logger.info(f"Using embedding model: {embedding_model}")

        # Initialize vector store
        self.vector_store = WeaviateVectorStore(weaviate_client=self.client, index_name=collection_name)

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        filters: Optional[Dict[str, Any]] = None,
        candidate_pool_size: int = 20,
        include_context: bool = True,
    ) -> List[PolicyChunk]:
        """
        Retrieve relevant policy chunks using hybrid search.

        Args:
            query: Search query
            top_k: Number of chunks to return
            filters: Metadata filters (e.g., {"active": True, "country": "CH"})
            candidate_pool_size: Size of candidate pool before reranking
            include_context: Whether to include previous/next chunks

        Returns:
            List of PolicyChunk objects with metadata and optional context
        """
        logger.info(f"Retrieving chunks for query: '{query}'")
        logger.info(f"Filters: {filters}, top_k: {top_k}")

        # Ensure active filter is always applied
        if filters is None:
            filters = {}
        if "active" not in filters:
            filters["active"] = True
            logger.info("Adding default filter: active=True")

        try:
            # Get collection
            collection = self.client.collections.get(self.collection_name)

            # Build Weaviate filters
            weaviate_filters = self._build_weaviate_filters(filters)

            # Generate query embedding
            query_embedding = self.embed_model.get_query_embedding(query)

            # Perform hybrid search
            logger.info(f"Performing hybrid search with alpha={self.alpha}")
            response = collection.query.hybrid(
                query=query,
                vector=query_embedding,
                alpha=self.alpha,
                limit=candidate_pool_size,
                filters=weaviate_filters,
                return_metadata=MetadataQuery(score=True),
            )

            logger.info(f"Found {len(response.objects)} candidates")

            if not response.objects:
                logger.warning("No results found. Consider relaxing filters or checking data.")
                return []

            # Convert to PolicyChunk objects
            chunks = []
            # Fetch `candidate_pool_size` candidates initially if reranking is enabled,
            # otherwise just fetch `top_k` (though we already limited to candidate_pool_size above).
            # The query above uses limit=candidate_pool_size.
            # So response.objects contains up to candidate_pool_size items.

            for obj in response.objects:
                chunk = self._convert_to_policy_chunk(obj, include_context)
                chunks.append(chunk)

            if self.reranker:
                logger.info("Reranking candidates...")
                chunks = self.reranker.rerank(query, chunks, top_k=top_k)
            else:
                chunks = chunks[:top_k]

            logger.info(f"Returning {len(chunks)} chunks")
            return chunks

        except Exception as e:
            logger.error(f"Error during retrieval: {e}")
            raise

    def _build_weaviate_filters(self, filters: Dict[str, Any]) -> Optional[Any]:
        """Build Weaviate filter from dict."""
        if not filters:
            return None

        logger.info(f"Building Weaviate filters from: {filters}")

        from weaviate.classes.query import Filter

        filter_conditions = []
        for key, value in filters.items():
            filter_conditions.append(Filter.by_property(key).equal(value))

        # Combine filters with AND
        if len(filter_conditions) == 1:
            return filter_conditions[0]
        else:
            result = filter_conditions[0]
            for condition in filter_conditions[1:]:
                result = result & condition
            return result

    def _convert_to_policy_chunk(self, obj: Any, include_context: bool) -> PolicyChunk:
        """Convert Weaviate object to PolicyChunk."""
        props = obj.properties

        chunk = PolicyChunk(
            text=props.get("text", ""),
            document_id=props.get("document_id", ""),
            document_name=props.get("document_name", ""),
            section_path_str=props.get("section_path_str", ""),
            chunk_id=props.get("chunk_id", ""),
            chunk_index=props.get("chunk_index", 0),
            topic=props.get("topic", ""),
            country=props.get("country", ""),
            active=props.get("active", True),
            last_modified=props.get("last_modified", ""),
            score=obj.metadata.score if hasattr(obj.metadata, "score") else 0.0,
        )

        # Get context chunks if requested
        if include_context:
            chunk.previous_chunk = self._get_adjacent_chunk(props.get("document_id"), props.get("chunk_index", 0) - 1)
            chunk.next_chunk = self._get_adjacent_chunk(props.get("document_id"), props.get("chunk_index", 0) + 1)

        return chunk

    def _get_adjacent_chunk(self, document_id: str, chunk_index: int) -> Optional[str]:
        """Get adjacent chunk text by document_id and chunk_index."""
        if chunk_index < 0:
            return None

        try:
            from weaviate.classes.query import Filter

            collection = self.client.collections.get(self.collection_name)
            response = collection.query.fetch_objects(
                filters=(
                    Filter.by_property("document_id").equal(document_id)
                    & Filter.by_property("chunk_index").equal(chunk_index)
                ),
                limit=1,
            )

            if response.objects:
                return response.objects[0].properties.get("text", "")
            return None
        except Exception as e:
            logger.warning(f"Could not fetch adjacent chunk: {e}")
            return None

    def close(self):
        """Close Weaviate client connection."""
        if self.client:
            self.client.close()
            logger.info("Weaviate client closed")
