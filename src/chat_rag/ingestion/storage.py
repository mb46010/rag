import logging
import weaviate
from weaviate.classes.config import Configure, Property, DataType, Tokenization
from dataclasses import asdict
from typing import List
from .models import PolicyChunk
from llama_index.embeddings.openai import OpenAIEmbedding

logger = logging.getLogger(__name__)


class WeaviateStorage:
    """Handles interactions with Weaviate."""

    def __init__(self, weaviate_url: str, collection_name: str, embedding_model: str):
        self.collection_name = collection_name
        self.client = weaviate.connect_to_local(host=weaviate_url.replace("http://", "").split(":")[0])
        self.embed_model = OpenAIEmbedding(model=embedding_model)
        logger.info(f"Using embedding model: {embedding_model}")

    def setup_collection(self):
        """Create or recreate Weaviate collection."""
        logger.info(f"Setting up collection: {self.collection_name}")

        # Delete existing collection if it exists
        if self.client.collections.exists(self.collection_name):
            logger.info(f"Deleting existing collection: {self.collection_name}")
            self.client.collections.delete(self.collection_name)

        # Create collection with schema
        # NOTE: Weaviate does not have a dedicated "categorical" type (like ENUM).
        # Instead, we use `Tokenization.FIELD` for text properties that act as categories (e.g., country codes, IDs).
        # This treats the entire string as a single token, preventing stopword removal (fixing issues like country="IT" -> "it" -> removed)
        # and ensuring exact matching for filters.
        logger.info("Creating new collection with schema")
        self.client.collections.create(
            name=self.collection_name,
            properties=[
                Property(name="document_id", data_type=DataType.TEXT, tokenization=Tokenization.FIELD),
                Property(name="document_name", data_type=DataType.TEXT, tokenization=Tokenization.FIELD),
                Property(name="section_path_str", data_type=DataType.TEXT, tokenization=Tokenization.FIELD),
                Property(name="chunk_id", data_type=DataType.TEXT, tokenization=Tokenization.FIELD),
                Property(name="chunk_index", data_type=DataType.INT),
                Property(name="text", data_type=DataType.TEXT),
                Property(name="text_indexed", data_type=DataType.TEXT),
                Property(name="topic", data_type=DataType.TEXT, tokenization=Tokenization.FIELD),
                Property(name="country", data_type=DataType.TEXT, tokenization=Tokenization.FIELD),
                Property(name="active", data_type=DataType.BOOL),
                Property(name="last_modified", data_type=DataType.TEXT),
                Property(name="qa_text", data_type=DataType.TEXT),
            ],
            vectorizer_config=Configure.Vectorizer.none(),  # BYOV mode
        )
        logger.info("Collection created successfully")

    def ingest_chunks(self, chunks: List[PolicyChunk]):
        """
        Ingest chunks into Weaviate with embeddings.
        """
        logger.info(f"Ingesting {len(chunks)} chunks into Weaviate")

        collection = self.client.collections.get(self.collection_name)

        # Process in batches
        batch_size = 10
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i : i + batch_size]
            logger.info(f"Processing batch {i // batch_size + 1}/{(len(chunks) - 1) // batch_size + 1}")

            # Generate embeddings for batch
            texts = [chunk.text_indexed for chunk in batch]
            embeddings = self.embed_model.get_text_embedding_batch(texts)

            # Insert into Weaviate
            with collection.batch.dynamic() as batch_insert:
                for chunk, embedding in zip(batch, embeddings):
                    # Convert chunk to dict, excluding section_path (list not supported)
                    chunk_dict = asdict(chunk)
                    del chunk_dict["section_path"]

                    batch_insert.add_object(properties=chunk_dict, vector=embedding)

        logger.info(f"Successfully ingested {len(chunks)} chunks")

    def close(self):
        """Close Weaviate client."""
        if self.client:
            self.client.close()
            logger.info("Weaviate client closed")
