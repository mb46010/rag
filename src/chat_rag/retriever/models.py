from dataclasses import dataclass
from typing import Optional


@dataclass
class PolicyChunk:
    """Retrieved policy chunk with metadata and context."""

    text: str
    document_id: str
    document_name: str
    section_path_str: str
    chunk_id: str
    chunk_index: int
    topic: str
    country: str
    active: bool
    last_modified: str
    score: float
    previous_chunk: Optional[str] = None
    next_chunk: Optional[str] = None
