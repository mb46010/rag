from chat_rag.retriever.hybrid import EnhancedHybridRetriever
from chat_rag.retriever.config import RetrievalConfig
import logging
import sys
import json
import glob
import os
from dotenv import load_dotenv

load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)


def main():
    config = RetrievalConfig()
    config.debug_to_file = True
    config.retrieval_output_dir = "output/retrieval_verification"

    # Ensure clean state
    os.makedirs(config.retrieval_output_dir, exist_ok=True)

    retriever = EnhancedHybridRetriever(config=config)

    query = "Can I book a business flight from ZRH to Tokyo?"
    print(f"Running retrieval for query: {query}")

    chunks = retriever.retrieve(query, top_k=3)

    print(f"Retrieved {len(chunks)} chunks.")

    # Check if files were generated
    files = glob.glob(f"{config.retrieval_output_dir}/*_1_retrieval.json")
    if not files:
        print("ERROR: No retrieval debug file found.")
        sys.exit(1)

    latest_file = max(files, key=os.path.getctime)
    print(f"Checking file: {latest_file}")

    with open(latest_file, "r") as f:
        data = json.load(f)

    chunks_data = data["chunks"]
    if not chunks_data:
        print("ERROR: No chunks in debug file.")
        sys.exit(1)

    first_chunk = chunks_data[0]

    # Check fields
    issues = []

    if first_chunk.get("query_type") is None:
        issues.append("query_type is None")

    if first_chunk.get("alpha_used") is None:
        issues.append("alpha_used is None")

    # Check previous/next chunk - note: might be null if no neighbors,
    # but at least one chunk in a large doc should have them.
    # We can check if the key exists and if text is populated for *some* chunk.
    has_neighbor = False
    for c in chunks_data:
        if c.get("previous_chunk") or c.get("next_chunk"):
            has_neighbor = True
            break

    if not has_neighbor:
        print("WARNING: No neighbors found in any chunk. This might be valid if chunks are isolated, but suspicious.")

    if issues:
        print(f"ISSUES FOUND: {issues}")
    else:
        print("SUCCESS: Debug data looks populated.")
        print(f"Query Type: {first_chunk.get('query_type')}")
        print(f"Alpha: {first_chunk.get('alpha_used')}")

    retriever.close()


if __name__ == "__main__":
    main()
