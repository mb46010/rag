import os
import logging
import argparse
from dotenv import load_dotenv
from .pipeline import IngestionPipeline
from .config import IngestionConfig

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")

if __name__ == "__main__":
    load_dotenv()

    parser = argparse.ArgumentParser(description="Ingest policy documents.")
    parser.add_argument(
        "--policy", type=str, default="default", help="Ingestion policy to use (default, qa_extractor)"
    )
    args = parser.parse_args()

    # Ensure OPENAI_API_KEY is set
    if not os.getenv("OPENAI_API_KEY"):
        raise ValueError("OPENAI_API_KEY environment variable not set")

    config = IngestionConfig()

    # Run ingestion
    pipeline = IngestionPipeline(config=config, policy_name=args.policy)
    try:
        pipeline.run()
    finally:
        pipeline.close()
