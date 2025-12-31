import logging
import json
import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import asdict

from .models import PolicyChunk
from .storage import WeaviateStorage
from .config import IngestionConfig
from .policies.base import IngestionPolicy
from .policies.default import DefaultIngestionPolicy
from .policies.qa_extractor import QaExtractorPolicy

logger = logging.getLogger(__name__)


class IngestionPipeline:
    """Ingestion pipeline for policy documents."""

    def __init__(
        self,
        config: IngestionConfig,
        policy_name: str = "default",
    ):
        self.config = config
        self.policy_name = policy_name

        # Initialize Policy
        if policy_name == "default":
            self.policy: IngestionPolicy = DefaultIngestionPolicy(config)
        elif policy_name == "qa_extractor":
            self.policy = QaExtractorPolicy(config)
        else:
            raise ValueError(f"Unknown policy name: {policy_name}")

        logger.info(f"Initializing IngestionPipeline with policy: {policy_name}")
        logger.info(f"Docs directory: {self.config.docs_dir}")

        self.storage = WeaviateStorage(
            self.config.weaviate_url, self.config.collection_name, self.config.embedding_model
        )

        # Ensure output directory exists for contracts
        self.config.output_dir.mkdir(parents=True, exist_ok=True)

    def load_document(self, file_path: Path) -> Dict[str, Any]:
        """Load policy document from JSON file."""
        logger.info(f"Loading document: {file_path.name}")
        with open(file_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def save_ingestion_contract(self, document: Dict[str, Any], chunks: List[PolicyChunk], file_path: Path):
        """Save ingestion contract to JSON file."""
        metadata = document["metadata"]

        contract = {
            "ingestion_metadata": {
                "pipeline_name": self.policy.get_name(),
                "pipeline_version": self.policy.get_version(),
                "ingestion_date": datetime.datetime.now().isoformat(),
                "source_file": file_path.name,
            },
            "document_metadata": metadata,
            "chunks": [asdict(chunk) for chunk in chunks],
        }

        # Construct output filename: {original_name}_{policy}_{timestamp or just policy}.json
        # User example: policy_wfh_it_chunks.json
        # We'll use: {stem}_chunks_{policy}.json
        output_filename = f"{file_path.stem}_chunks_{self.policy.get_name()}.json"
        output_path = self.config.output_dir / output_filename

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(contract, f, indent=2, ensure_ascii=False)

        logger.info(f"Saved ingestion contract to: {output_path}")

    def run(self):
        """Run the full ingestion pipeline."""
        logger.info("=" * 80)
        logger.info(f"Starting ingestion pipeline [{self.policy.get_name()}]")
        logger.info("=" * 80)

        # Setup collection
        self.storage.setup_collection()

        # Find all JSON files
        json_files = list(self.config.docs_dir.glob("*.json"))
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

                # Create chunks using policy
                chunks = self.policy.process_document(document)

                # Save Contract JSON
                self.save_ingestion_contract(document, chunks, file_path)

                # Ingest chunks
                self.storage.ingest_chunks(chunks)

                total_chunks += len(chunks)

            except Exception as e:
                logger.error(f"Error processing {file_path.name}: {e}", exc_info=True)
                raise

        logger.info("=" * 80)
        logger.info(f"Ingestion complete! Total chunks: {total_chunks}")
        logger.info("=" * 80)

    def close(self):
        self.storage.close()
