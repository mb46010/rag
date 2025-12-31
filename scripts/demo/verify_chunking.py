"""
Verify chunking of policy documents.

This script is used to verify that the chunking logic in ingestion.py is working as expected.
It performs the following steps:
1. Loads a policy document from a JSON file.
2. Saves the original markdown content to a file for easier inspection.
3. Uses the IngestionPipeline to create chunks from the document.
4. Saves the resulting chunks to a JSON file.

Usage:
    python verify_chunking.py
"""

import os
import sys
import json
import logging
import argparse
from pathlib import Path
from dataclasses import asdict

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Add project root to sys.path to allow imports from demos.chat_rag
# This script is at: <REPO_ROOT>/demos/chat_rag/scripts/demo/verify_chunking.py
# We need to add <REPO_ROOT> to sys.path
# Example: /home/marco/Code/Templates/demos/chat_rag/scripts/demo/verify_chunking.py
# Root should be /home/marco/Code/Templates
# current_file = Path(__file__).resolve()
# templates_root = current_file.parents[4]  # Go up 5 levels to reach Templates
# if str(templates_root) not in sys.path:
#     sys.path.insert(0, str(templates_root))


def run_verification(input_path: Path, output_dir: Path):
    """Run chunking verification for a given policy document."""
    logger.info(f"Verifying chunking for: {input_path}")

    if not input_path.exists():
        logger.error(f"Input file not found: {input_path}")
        return

    # Ensure output directory exists
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        # Import after adding to sys.path
        from chat_rag.ingestion import IngestionPipeline

        # NOTE: Weaviate and OpenAI are initialized in IngestionPipeline.__init__.
        # If you don't have them running/configured, this might fail.
        # We previously used mocks here to allow chunking verification without external dependencies.
        try:
            pipeline = IngestionPipeline()
        except Exception as e:
            logger.warning(f"Could not initialize full IngestionPipeline (probably Weaviate/OpenAI missing): {e}")
            logger.info("Attempting to run with a partially initialized pipeline for chunking only...")

            # Fallback: Mock enough to get create_chunks working
            from unittest.mock import MagicMock
            from llama_index.core.node_parser import SentenceSplitter

            pipeline = IngestionPipeline.__new__(IngestionPipeline)
            pipeline.max_chunk_tokens = 400
            pipeline.splitter = SentenceSplitter(chunk_size=400, chunk_overlap=50)
            pipeline._generate_document_id = lambda content: "mock_doc_id"
            pipeline._generate_chunk_id = lambda doc_id, path, text: "mock_chunk_id"

        # Load document
        document = pipeline.load_document(input_path)

        # 1. Save original markdown content for readability
        content = document.get("content", "")
        md_file = output_dir / f"{input_path.stem}.md"
        with open(md_file, "w", encoding="utf-8") as f:
            f.write(content)
        logger.info(f"Saved original markdown content to: {md_file}")

        # 2. Create chunks using the pipeline's logic
        chunks = pipeline.create_chunks(document)

        # 3. Save resulting chunks to JSON
        chunks_data = [asdict(c) for c in chunks]
        json_file = output_dir / f"{input_path.stem}_chunks.json"
        with open(json_file, "w", encoding="utf-8") as f:
            json.dump(chunks_data, f, indent=2, ensure_ascii=False)
        logger.info(f"Saved {len(chunks)} chunks to: {json_file}")

        # Print a visual summary to the console
        print("\n" + "=" * 80)
        print(f"CHUNKING SUMMARY: {input_path.name}")
        print("=" * 80)
        for i, chunk in enumerate(chunks):
            path_str = " > ".join(chunk.section_path)
            # Preview cleaning: remove multiple newlines and truncate
            preview = chunk.text.replace("\n", " ").strip()
            preview = (preview[:75] + "...") if len(preview) > 75 else preview

            print(f"CHUNK {i:02d} | Path: {path_str}")
            print(f"         | Preview: {preview}")
            print("-" * 80)
        print(f"Total Chunks Created: {len(chunks)}\n")

    except Exception as e:
        logger.exception(f"An error occurred during verification: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Verify chunking of policy documents.")
    parser.add_argument("--input", type=str, help="Path to the policy JSON file.")
    parser.add_argument("--output_dir", type=str, help="Directory to save the results.")

    args = parser.parse_args()

    # Define default paths relative to this script's location
    script_dir = Path(__file__).parent.resolve()

    # Resolve input file
    if args.input:
        target_file = Path(args.input).resolve()
    else:
        docs_dir = (script_dir / "../../documents").resolve()
        target_file = docs_dir / "policy_wfh_it.json"
        if not target_file.exists():
            json_files = sorted(list(docs_dir.glob("policy_*.json")))
            if json_files:
                target_file = json_files[0]
            else:
                logger.error(f"No default policy file found in {docs_dir}. Please specify one using --input.")
                sys.exit(1)

    # Resolve output directory
    if args.output_dir:
        output_dir = Path(args.output_dir).resolve()
    else:
        output_dir = script_dir / "output"

    # Run the verification
    run_verification(target_file, output_dir)
