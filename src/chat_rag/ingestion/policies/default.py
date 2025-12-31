import logging
from typing import List, Dict, Any
from llama_index.core.node_parser import SentenceSplitter

from ..models import PolicyChunk
from ..text_processing import (
    split_by_headers,
    generate_document_id,
    count_tokens,
    create_policy_chunk,
)
from .base import IngestionPolicy
from ..config import IngestionConfig

logger = logging.getLogger(__name__)


class DefaultIngestionPolicy(IngestionPolicy):
    """
    Default policy: splits by markdown headers and chunk size.
    Ported from original pipeline.py.
    """

    def __init__(self, config: IngestionConfig):
        super().__init__(config)
        self.splitter = SentenceSplitter(chunk_size=config.max_chunk_tokens, chunk_overlap=config.chunk_overlap)

    def get_name(self) -> str:
        return "default"

    def get_version(self) -> str:
        return "1.0.0"

    def process_document(self, document: Dict[str, Any]) -> List[PolicyChunk]:
        metadata = document["metadata"]
        content = document["content"]

        logger.info(f"Processing with Default Policy: {metadata.get('document_name', 'Unknown')}")

        document_id = generate_document_id(content)

        # Split content by sections
        sections = split_by_headers(content)

        chunks = []
        for section_path, section_text in sections:
            section_path_str = " > ".join(section_path)

            if count_tokens(section_text) > self.config.max_chunk_tokens:
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
                chunk = create_policy_chunk(
                    document_id=document_id,
                    metadata=metadata,
                    section_path=section_path,
                    section_path_str=section_path_str,
                    text=section_text,
                    chunk_index=len(chunks),
                )
                chunks.append(chunk)

        # Set policy info in metadata (although PolicyChunk struct doesn't strictly have a policy field in the original model,
        # the user asked for local JSON file with metadata including pipeline name and version.
        # We'll attach it to the chunks or handle it in the pipeline when saving the JSON.)

        return chunks
