# Ingestion Pipeline Documentation

## 1. Overview
The ingestion pipeline is responsible for loading HR policy documents (in JSON format), splitting them into semantically meaningful chunks, generating embeddings, and storing them in a Weaviate vector database.

The pipeline ensures that chunks preserve their structural context (i.e., which section/header they belong to) to improve retrieval quality.

## 2. Architecture
The code is organized as a modular package in `src/ingestion/`:

| File | Description |
|------|-------------|
| `models.py` | Defines data structures, primarily the `PolicyChunk` dataclass. |
| `text_processing.py` | Contains pure functions for text analysis, splitting by headers, and token counting. |
| `storage.py` | Encapsulates all interactions with the Weaviate vector database. |
| `pipeline.py` | The main controller class `IngestionPipeline` that binds everything together. |
| `__main__.py` | Entry point for running the pipeline as a script (`python -m src.ingestion`). |

## 3. detailed Components

### Data Model (`models.py`)
**`PolicyChunk`** is the core data unit.
- **Metadata**: `document_id`, `document_name`, `topic`, `country`, `active`, `last_modified`.
- **Content**:
    - `text`: The raw text of the chunk.
    - `text_indexed`: The text prepended with the `section_path_str`. This is what is actually embedded, giving the vector model context about where the text appeared.
- **Structure**:
    - `section_path`: A list of strings representing the markdown header hierarchy (e.g., `["Work From Home Policy", "2. Eligibility"]`).
    - `section_path_str`: A string representation of the path (e.g., `"Work From Home Policy > 2. Eligibility"`).

### Text Processing (`text_processing.py`)
Helper functions that operate on text:
- **`split_by_headers(content)`**: splits a markdown document into sections based on headers (`#`). It tracks the nesting level to build accurate section paths.
- **`generate_document_id` / `generate_chunk_id`**: Creates deterministic hashes (SHA256) so that re-running ingestion yields the same IDs for the same content.
- **`count_tokens`**: A heuristic token counter.

### Storage (`storage.py`)
**`WeaviateStorage`** handles the persistence layer.
- **`setup_collection`**: Defines the Weaviate schema.
- **`ingest_chunks`**: Takes a list of `PolicyChunk` objects, generates embeddings using OpenAI (`text-embedding-3-large`), and uploads them in batches.

### Pipeline Logic (`pipeline.py`)
**`IngestionPipeline`** orchestrates the flow:
1. **Load**: Reads JSON files from the `documents/` directory.
2. **Pre-processing**: Skips inactive policies.
3. **Chunking**:
    - First, splits by markdown headers using `split_by_headers`.
    - If a section is larger than `max_chunk_tokens` (default 400), it uses `llama_index.core.node_parser.SentenceSplitter` to further subdivide it while keeping the section context.
4. **Ingest**: Delegates to `WeaviateStorage` to save the chunks.

## 4. Usage

### Running Ingestion
To run the full ingestion process:

```bash
python -m src.ingestion
```

**Prerequisites**:
- `OPENAI_API_KEY` must be set in your `.env` file or environment.
- Weaviate must be running (default `http://localhost:8080`).

### Running Tests
Unit tests interface with `src.ingestion` but mock the database layer to be fast and independent.

```bash
uv run pytest tests/ingestion/test_ingestion.py
```
