import logging
import json
from pathlib import Path
from typing import List, Dict, Any
from llama_index.core.node_parser import SentenceSplitter

from .models import PolicyChunk
from .text_processing import (
    split_by_headers,
    generate_document_id,
    count_tokens,
    create_policy_chunk,
)
from .storage import WeaviateStorage

logger = logging.getLogger(__name__)


class IngestionPipeline:
    """Ingestion pipeline for policy documents."""

    def __init__(
        self,
        docs_dir: str = "./documents",
        weaviate_url: str = "http://localhost:8080",
        collection_name: str = "PolicyChunk",
        embedding_model: str = "text-embedding-3-large",
        max_chunk_tokens: int = 400,
    ):
        self.docs_dir = Path(docs_dir)
        self.max_chunk_tokens = max_chunk_tokens

        logger.info("Initializing IngestionPipeline")
        logger.info(f"Docs directory: {self.docs_dir}")

        self.storage = WeaviateStorage(weaviate_url, collection_name, embedding_model)
        self.splitter = SentenceSplitter(chunk_size=max_chunk_tokens, chunk_overlap=50)

    def load_document(self, file_path: Path) -> Dict[str, Any]:
        """Load policy document from JSON file."""
        logger.info(f"Loading document: {file_path.name}")
        with open(file_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def create_chunks(self, document: Dict[str, Any]) -> List[PolicyChunk]:
        """
        Create chunks from document content.

        Splits content by markdown headers and creates chunks with section paths.
        """
        metadata = document["metadata"]
        content = document["content"]

        logger.info(f"Processing: {metadata['document_name']}")

        # Generate document_id from content hash
        document_id = generate_document_id(content)
        logger.info(f"Generated document_id: {document_id}")

        # Split content by sections
        sections = split_by_headers(content)
        logger.info(f"Found {len(sections)} sections")

        # Create chunks
        chunks = []
        for section_path, section_text in sections:
            section_path_str = " > ".join(section_path)

            # Check if section is too long
            if count_tokens(section_text) > self.max_chunk_tokens:
                # Split into smaller chunks
                logger.info(f"Section '{section_path_str}' is too long, splitting...")
                sub_chunks = self.splitter.split_text(section_text)

                for i, sub_chunk in enumerate(sub_chunks):
                    chunk = create_policy_chunk(
                        document_id=document_id,
                        metadata=metadata,
                        section_path=section_path,
                        section_path_str=section_path_str,
                        text=sub_chunk,
                        chunk_index=len(chunks),
                    )
                    chunks.append(chunk)
            else:
                # Use section as-is
                chunk = create_policy_chunk(
                    document_id=document_id,
                    metadata=metadata,
                    section_path=section_path,
                    section_path_str=section_path_str,
                    text=section_text,
                    chunk_index=len(chunks),
                )
                chunks.append(chunk)

        logger.info(f"Created {len(chunks)} chunks")
        return chunks

    def run(self):
        """Run the full ingestion pipeline."""
        logger.info("=" * 80)
        logger.info("Starting ingestion pipeline")
        logger.info("=" * 80)

        # Setup collection
        self.storage.setup_collection()

        # Find all JSON files
        json_files = list(self.docs_dir.glob("*.json"))
        logger.info(f"Found {len(json_files)} JSON files")

        # Process each document
        total_chunks = 0
        for file_path in json_files:
            try:
                # Load document
                document = self.load_document(file_path)

                # Skip deprecated policies (not active)
                if not document["metadata"].get("active", True):
                    logger.info(f"Skipping inactive document: {file_path.name}")
                    continue

                # Create chunks
                chunks = self.create_chunks(document)

                # Ingest chunks
                self.storage.ingest_chunks(chunks)

                total_chunks += len(chunks)

            except Exception as e:
                logger.error(f"Error processing {file_path.name}: {e}")
                # We rename 'raise' to allow continuing with other files or fail completely?
                # Original code raised, so we stick to that.
                raise

        logger.info("=" * 80)
        logger.info(f"Ingestion complete! Total chunks: {total_chunks}")
        logger.info("=" * 80)

    def close(self):
        self.storage.close()
