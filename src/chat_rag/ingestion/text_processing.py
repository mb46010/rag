import re
import hashlib
from typing import List, Tuple, Dict, Any
from .models import PolicyChunk


def split_by_headers(content: str, min_level: int = 6) -> List[Tuple[List[str], str]]:
    """
    Split markdown content by headers and create section paths.
    Only splits if the header level is <= min_level.
    """
    sections = []
    current_path = []
    current_text = []
    lines = content.split("\n")

    for line in lines:
        # Check if line is a header
        header_match = re.match(r"^(#{1,6})\s+(.+)$", line)

        if header_match:
            level = len(header_match.group(1))

            if level <= min_level:
                # meaningful split point
                # Save previous section
                if current_text and current_path:
                    text = "\n".join(current_text).strip()
                    if text:
                        sections.append((current_path.copy(), text))
                    current_text = []

                header_text = header_match.group(2).strip()

                # Adjust path based on header level
                # If we skipped some levels (e.g. current path is H1, and now we see H2),
                # We just append.
                # But if we go back up (e.g. at H3, see H2), we trunk.
                # Standard markdown logic:
                current_path = current_path[: level - 1] + [header_text]
            else:
                # Treat as normal text because it's too deep
                current_text.append(line)
        else:
            current_text.append(line)

    # Save last section
    if current_text and current_path:
        text = "\n".join(current_text).strip()
        if text:
            sections.append((current_path.copy(), text))

    return sections


def generate_document_id(content: str) -> str:
    """Generate idempotent document_id from content."""
    return hashlib.sha256(content.encode()).hexdigest()[:16]


def generate_chunk_id(document_id: str, section_path: str, text: str) -> str:
    """Generate idempotent chunk_id from content and path."""
    combined = f"{document_id}:{section_path}:{text}"
    return hashlib.sha256(combined.encode()).hexdigest()[:16]


def count_tokens(text: str) -> int:
    """Rough token count estimate (4 chars per token)."""
    return len(text) // 4


def create_policy_chunk(
    document_id: str,
    metadata: Dict[str, Any],
    section_path: List[str],
    section_path_str: str,
    text: str,
    chunk_index: int,
    qa_text: str = "",
) -> PolicyChunk:
    """Create a PolicyChunk object."""
    # Generate chunk_id from content and path
    chunk_id = generate_chunk_id(document_id, section_path_str, text)

    # Create indexed text with prepended section path
    # If qa_text is present, we might want to include it in the indexed text too, or strictly keep it separate.
    # User's previous request ("enriching the text for retrieval") suggests we should index it somehow on the text side too?
    # Or rely on Weaviate property.
    # Let's keep `text_indexed` as: section_path + text + qa_text (if any).

    text_indexed_parts = [section_path_str, text]
    if qa_text:
        text_indexed_parts.append(f"Questions this answer:\n{qa_text}")

    text_indexed = "\n\n".join(text_indexed_parts)

    return PolicyChunk(
        document_id=document_id,
        document_name=metadata["document_name"],
        section_path=section_path,
        section_path_str=section_path_str,
        chunk_id=chunk_id,
        chunk_index=chunk_index,
        text=text,
        text_indexed=text_indexed,
        topic=metadata["topic"],
        country=metadata["country"],
        active=metadata["active"],
        last_modified=metadata["last_modified"],
        qa_text=qa_text,
    )
