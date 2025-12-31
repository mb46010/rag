import os
import logging
from dotenv import load_dotenv
from .pipeline import IngestionPipeline

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")

if __name__ == "__main__":
    load_dotenv()

    # Ensure OPENAI_API_KEY is set
    if not os.getenv("OPENAI_API_KEY"):
        raise ValueError("OPENAI_API_KEY environment variable not set")

    # Run ingestion
    pipeline = IngestionPipeline()
    try:
        pipeline.run()
    finally:
        pipeline.close()
