"""Ingestion pipeline module."""

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
        """Initialize the ingestion pipeline.

        Args:
            config: The ingestion configuration.
            policy_name: The name of the policy to use (default: "default").
        """
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
        """Load policy document from JSON file, optionally reading external content."""
        logger.info(f"Loading document: {file_path.name}")
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        # check for external content file
        external_file = data.get("markdown_file") or data.get("content_file")
        if external_file:
            content_path = file_path.parent / external_file
            if content_path.exists():
                logger.info(f"Loading external content from: {external_file}")
                with open(content_path, "r", encoding="utf-8") as cf:
                    data["content"] = cf.read()
            else:
                logger.warning(f"External content file not found: {content_path}")

        return data

    def _calculate_statistics(self, chunks: List[PolicyChunk]) -> Dict[str, Any]:
        """Calculate detailed statistics for the ingested chunks."""
        if not chunks:
            return {}

        # 1. Basic Counts
        total_chunks = len(chunks)

        # 2. Length Statistics (Word count approximation)
        # Using simple split() is a good enough proxy for words
        word_counts = [len(c.text.split()) for c in chunks]
        min_len = min(word_counts)
        max_len = max(word_counts)
        mean_len = sum(word_counts) / total_chunks
        sorted_lens = sorted(word_counts)
        median_len = sorted_lens[total_chunks // 2]

        # 3. Section Analysis
        # Count unique sections and fragmentation
        # Convert list to tuple to make it hashable for set
        all_section_paths = [tuple(c.section_path) for c in chunks]
        unique_sections = set(all_section_paths)

        # Max depth
        max_depth = max(len(path) for path in all_section_paths) if all_section_paths else 0

        # Fragmentation: Sections broken into > 1 chunk
        section_counts = {}
        for path in all_section_paths:
            section_counts[path] = section_counts.get(path, 0) + 1

        fragmented_sections = sum(1 for count in section_counts.values() if count > 1)
        chunks_per_section = total_chunks / len(unique_sections) if unique_sections else 0

        # Leaf sections analysis (heuristic: sections that appear as full paths)
        # In this context, every chunk's section path is effectively a "leaf" of that chunk's context

        # 4. List Heuristics
        # Check for lines starting with "- ", "* ", or "1. "
        list_markers = ["- ", "* "]

        chunks_with_lists = 0
        total_list_items = 0

        for c in chunks:
            lines = c.text.split("\n")
            has_list = False
            for line in lines:
                stripped = line.strip()
                # Check unordered list or numbered list (simple heuristic)
                if any(stripped.startswith(m) for m in list_markers) or (
                    stripped and stripped[0].isdigit() and ". " in stripped[:4]
                ):
                    total_list_items += 1
                    has_list = True

            if has_list:
                chunks_with_lists += 1

        return {
            "total_chunks": total_chunks,
            "lengths_words": {"min": min_len, "max": max_len, "mean": round(mean_len, 1), "median": median_len},
            "section_stats": {
                "unique_sections_count": len(unique_sections),
                "max_depth": max_depth,
                "fragmented_sections_count": fragmented_sections,
                "avg_chunks_per_section": round(chunks_per_section, 2),
            },
            "list_stats": {
                "chunks_with_lists": chunks_with_lists,
                "total_list_items_found": total_list_items,
                "percent_chunks_with_lists": round((chunks_with_lists / total_chunks) * 100, 1),
            },
        }

    def save_ingestion_contract(self, document: Dict[str, Any], chunks: List[PolicyChunk], file_path: Path):
        """Save ingestion contract to JSON file."""
        metadata = document["metadata"]

        # Convert config to dict and handle Path objects
        config_dict = asdict(self.config)
        for k, v in config_dict.items():
            if isinstance(v, Path):
                config_dict[k] = str(v)

        contract = {
            "ingestion_metadata": {
                "pipeline_name": self.policy.get_name(),
                "pipeline_version": self.policy.get_version(),
                "ingestion_date": datetime.datetime.now().isoformat(),
                "source_file": file_path.name,
            },
            "ingestion_config": config_dict,
            "document_metadata": metadata,
            "chunk_statistics": self._calculate_statistics(chunks),
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
        """Close the storage client."""
        self.storage.close()
