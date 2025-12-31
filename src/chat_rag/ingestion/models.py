from dataclasses import dataclass
from typing import List


@dataclass
class PolicyChunk:
    """Policy chunk with metadata."""

    document_id: str
    document_name: str
    section_path: List[str]
    section_path_str: str
    chunk_id: str
    chunk_index: int
    text: str
    text_indexed: str  # Text with prepended section path for BM25
    topic: str
    country: str
    active: bool
    last_modified: str
    qa_text: str = ""  # Optional generated questions
