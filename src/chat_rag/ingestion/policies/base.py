from abc import ABC, abstractmethod
from typing import List, Dict, Any
from ..models import PolicyChunk
from ..config import IngestionConfig


class IngestionPolicy(ABC):
    """Abstract base class for ingestion policies."""

    def __init__(self, config: IngestionConfig):
        self.config = config

    @abstractmethod
    def process_document(self, document: Dict[str, Any]) -> List[PolicyChunk]:
        """
        Process a document and return a list of PolicyChunks.

        Args:
            document: Dictionary containing document content and metadata.

        Returns:
            List[PolicyChunk]: List of chunks to ingest.
        """
        pass

    @abstractmethod
    def get_name(self) -> str:
        """Return the name of the policy."""
        pass

    @abstractmethod
    def get_version(self) -> str:
        """Return the version of the policy."""
        pass
