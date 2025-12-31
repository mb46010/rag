"""Integration tests for the ingestion pipeline using a local Weaviate instance."""

import pytest

import json
import os
import weaviate
from src.ingestion import IngestionPipeline

# start weaviate with:
# bash scripts/start_weaviate.sh


class TestIngestionIntegration:
    """Integration tests for IngestionPipeline with local Weaviate."""

    @pytest.fixture
    def sample_document(self):
        """Sample policy document for integration testing."""
        return {
            "metadata": {
                "document_id": "pol_int_test",
                "document_name": "Integration Test Policy",
                "topic": "Integration Testing",
                "country": "IT",
                "active": True,
                "last_modified": "2024-01-01",
            },
            "content": """# Integration Policy

## 1. Scope
This is a test policy for integration testing against Weaviate.

## 2. Details
It should be chunked and stored in the local vector DB.""",
        }

    @pytest.fixture
    def pipeline(self, tmp_path):
        """Create a real pipeline instance pointing to a test collection."""
        from dotenv import load_dotenv

        load_dotenv()

        # Ensure OPENAI_API_KEY is present for embeddings
        if not os.getenv("OPENAI_API_KEY"):
            pytest.skip("OPENAI_API_KEY not set")

        # Check if Weaviate is reachable (fail fast if not, or skip)
        try:
            # Simple check
            client = weaviate.connect_to_local()
            client.close()
        except Exception as e:
            pytest.skip(f"Local Weaviate not reachable: {e}")

        # Use a specific collection name for testing to avoid messing with real data
        return IngestionPipeline(docs_dir=str(tmp_path), collection_name="PolicyChunkIntegrationTest")

    def test_ingestion_flow(self, pipeline, sample_document, tmp_path):
        """Test full ingestion flow against real Weaviate."""
        try:
            # 1. Setup: Create temp JSON file
            test_file = tmp_path / "integration_policy.json"
            with open(test_file, "w") as f:
                json.dump(sample_document, f)

            # 2. Execution: Run pipeline
            # This calls setup_collection (wipes DB for this collection) and ingests
            pipeline.run()

            # 3. Verification: Query Weaviate
            collection = pipeline.storage.client.collections.get("PolicyChunkIntegrationTest")
            response = collection.query.fetch_objects(limit=10)

            assert len(response.objects) > 0, "No objects found in Weaviate after ingestion"

            # Check properties of the first object
            found_doc = False
            for obj in response.objects:
                if obj.properties["document_name"] == "Integration Test Policy":
                    found_doc = True
                    assert obj.properties["country"] == "IT"
                    assert "Integration Policy" in obj.properties["text_indexed"]
                    break

            assert found_doc, "Ingested document not found in query results"

        finally:
            # Cleanup
            if pipeline:
                pipeline.close()
