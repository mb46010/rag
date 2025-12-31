# Usage:
# python scripts/demo/verify_retriever.py "query" output.json
# python scripts/demo/verify_retriever.py "query" output.json --filters '{"country": "CH"}'

import sys
import os
import argparse
import json
import logging
from dataclasses import asdict
from dotenv import load_dotenv

load_dotenv()

# Add src to sys.path
sys.path.append(os.path.join(os.path.dirname(__file__), "../../src"))

from chat_rag.retriever.hybrid import HybridRetriever

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Verify retriever implementation.")
    parser.add_argument("query", type=str, help="Search query")
    parser.add_argument("output_file", type=str, help="Output JSON file path")
    parser.add_argument("--filters", type=str, help='JSON string of filters (e.g. \'{"country": "CH"}\')')

    args = parser.parse_args()

    try:
        logger.info("Initializing HybridRetriever...")
        retriever = HybridRetriever()

        filters = {}
        if args.filters:
            try:
                filters = json.loads(args.filters)
                logger.info(f"Using filters: {filters}")
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse filters JSON: {e}")
                sys.exit(1)

        logger.info(f"Retrieving for query: '{args.query}'")
        chunks = retriever.retrieve(query=args.query, top_k=5, filters=filters)

        logger.info(f"Retrieved {len(chunks)} chunks.")

        # Serialize chunks
        results = [asdict(chunk) for chunk in chunks]

        # Save to file
        logger.info(f"Saving results to {args.output_file}")
        with open(args.output_file, "w") as f:
            json.dump(results, f, indent=2)

        logger.info("Done.")

    except Exception as e:
        logger.error(f"An error occurred: {e}")
        sys.exit(1)
    finally:
        # Ensure resources are closed if needed (HybridRetriever has a close method)
        if "retriever" in locals():
            retriever.close()


if __name__ == "__main__":
    main()
