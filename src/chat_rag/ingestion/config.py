from dataclasses import dataclass
from pathlib import Path


@dataclass
class IngestionConfig:
    """Configuration for ingestion pipeline."""

    docs_dir: Path = Path("./documents")
    weaviate_url: str = "http://localhost:8080"
    collection_name: str = "PolicyChunk"
    embedding_model: str = "text-embedding-3-large"
    max_chunk_tokens: int = 400
    chunk_overlap: int = 50
    output_dir: Path = Path("./output")
    llm_model: str = "gpt-3.5-turbo"
    questions_to_generate: int = 3
    minimum_split_level: str = "Header6"  # Default to splitting on all headers (up to h6)
