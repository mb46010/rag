import logging
from typing import List, Dict, Any
from llama_index.core import Document
from llama_index.core.ingestion import IngestionPipeline as LlamaIngestionPipeline
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.extractors import (
    TitleExtractor,
    QuestionsAnsweredExtractor,
)
from llama_index.llms.openai import OpenAI

from ..models import PolicyChunk
from ..text_processing import (
    generate_document_id,
    create_policy_chunk,
)
from .base import IngestionPolicy
from ..config import IngestionConfig

logger = logging.getLogger(__name__)


class QaExtractorPolicy(IngestionPolicy):
    """
    Policy that uses LlamaIndex extractors (Title, QA) to enrich chunks.
    """

    def __init__(self, config: IngestionConfig):
        super().__init__(config)
        self.llm = OpenAI(model=config.llm_model)

        # Transformations
        self.transformations = [
            SentenceSplitter(chunk_size=config.max_chunk_tokens, chunk_overlap=config.chunk_overlap),
            TitleExtractor(llm=self.llm, nodes=5),
            QuestionsAnsweredExtractor(llm=self.llm, questions=config.questions_to_generate),
        ]

    def get_name(self) -> str:
        return "qa_extractor"

    def get_version(self) -> str:
        return "1.0.0"

    def process_document(self, document: Dict[str, Any]) -> List[PolicyChunk]:
        metadata = document["metadata"]
        content = document["content"]

        logger.info(f"Processing with QA Extractor Policy: {metadata.get('document_name', 'Unknown')}")

        # Create LlamaIndex Document
        doc = Document(text=content, metadata=metadata)

        # Run LlamaIndex Pipeline
        pipeline = LlamaIngestionPipeline(transformations=self.transformations)
        nodes = pipeline.run(documents=[doc])

        document_id = generate_document_id(content)

        chunks = []
        for i, node in enumerate(nodes):
            # Extract generated metadata
            node_metadata = node.metadata

            # Construct section path string if available, or use title
            # TitleExtractor puts title in 'document_title' or 'section_summary' depending on config
            # We'll try to map what we can.

            # Note: The original 'section_path' logic relied on markdown headers.
            # LlamaIndex SentenceSplitter doesn't inherently preserve that hierarchy unless we use MarkdownNodeParser.
            # But the user asked to leverage IngestionPipeline with transformations.
            # We will map the node content to our PolicyChunk.

            # If TitleExtractor added a title, we might want to prepend it to text or use it.
            # For now, we'll store the raw node text.

            section_path_str = node_metadata.get("section_summary", "") or node_metadata.get("document_title", "")

            # We might want to capture the Q&A in the text or metadata.
            # QuestionsAnsweredExtractor adds 'questions_this_excerpt_can_answer' to metadata.
            qa_text = node_metadata.get("questions_this_excerpt_can_answer", "")

            chunk_text = node.get_content()
            if qa_text:
                # Append QA to text or just let it exist in metadata?
                # The user asked for "leverage ... transformations", likely to enrich the embedding.
                # Standard LlamaIndex behavior embeds the metadata if configured.
                # For our PolicyChunk, we store explicit text. Let's append questions to text for visibility/indexing
                # similar to how we might want them retrieved.
                chunk_text += f"\n\nQuestions this section can answer:\n{qa_text}"

            chunk = create_policy_chunk(
                document_id=document_id,
                metadata=metadata,  # original metadata
                section_path=[section_path_str] if section_path_str else [],  # simplified path
                section_path_str=section_path_str,
                text=chunk_text,
                chunk_index=i,
            )
            chunks.append(chunk)

        return chunks
