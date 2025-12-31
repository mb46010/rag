#!/bin/bash
# Start Weaviate vector database in BYOV (Bring Your Own Vectors) mode
# Weaviate will run on port 8080 and perform hybrid search (BM25 + vector similarity)
# Data is persisted in a Docker volume named 'weaviate_data'.
# To wipe the data completely, run: docker volume rm weaviate_data

set -e

echo "Starting Weaviate vector database..."

# Stop and remove existing container if it exists
if docker ps -a --format '{{.Names}}' | grep -q '^weaviate$'; then
    echo "Stopping existing Weaviate container..."
    docker stop weaviate || true
    docker rm weaviate || true
fi

# Start Weaviate with Docker
docker run -d \
  --name weaviate \
  -p 8080:8080 \
  -p 50051:50051 \
  -e AUTHENTICATION_ANONYMOUS_ACCESS_ENABLED=true \
  -e PERSISTENCE_DATA_PATH=/var/lib/weaviate \
  -e DEFAULT_VECTORIZER_MODULE=none \
  -e ENABLE_MODULES="" \
  -e CLUSTER_HOSTNAME=node1 \
  -e QUERY_DEFAULTS_LIMIT=20 \
  -e DISK_USE_WARNING_PERCENTAGE=80 \
  -e DISK_USE_READONLY_PERCENTAGE=90 \
  -v weaviate_data:/var/lib/weaviate \
  cr.weaviate.io/semitechnologies/weaviate:1.27.5

echo "Waiting for Weaviate to be ready..."
for i in {1..30}; do
    if curl -sf http://localhost:8080/v1/.well-known/ready > /dev/null 2>&1; then
        echo "Weaviate is ready!"
        echo "Weaviate UI: http://localhost:8080"
        echo "Weaviate API: http://localhost:8080/v1"
        exit 0
    fi
    echo "Waiting... ($i/30)"
    sleep 2
done

echo "ERROR: Weaviate failed to start within 60 seconds"
docker logs weaviate
exit 1
