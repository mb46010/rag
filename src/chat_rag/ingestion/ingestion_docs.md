# Ingestion Pipeline Documentation

## 1. Overview
The ingestion pipeline is responsible for loading HR policy documents (in JSON format), processing them according to a specific **Policy**, generating metadata and chunks, saving a local **Ingestion Contract**, generating embeddings, and storing them in a Weaviate vector database.

The pipeline ensures that chunks preserve their structural context (i.e., which section/header they belong to) to improve retrieval quality.

## 2. Architecture
The code is organized as a modular package in `src/chat_rag/ingestion/`:

| File/Directory | Description |
|----------------|-------------|
| `config.py` | Central configuration (`IngestionConfig`) for paths, URLs, models, etc. |
| `models.py` | Defines data structures, primarily the `PolicyChunk` dataclass. |
| `policies/` | Directory containing strategy implementations for processing documents. |
| `text_processing.py` | Helper functions for text analysis, splitting by headers, and token counting. |
| `storage.py` | Encapsulates all interactions with the Weaviate vector database. |
| `pipeline.py` | The main controller class `IngestionPipeline` that orchestrates the flow. |
| `__main__.py` | Entry point for running the pipeline as a script (`python -m src.chat_rag.ingestion`). |

## 3. Detailed Components

### Configuration (`config.py`)
- **`IngestionConfig`**: Dataclass holding settings like `weaviate_url`, `embedding_model`, `max_chunk_tokens`, `questions_to_generate`, and directory paths.

### Policies (`policies/`)
The pipeline uses the **Strategy Pattern** to define how documents are processed.
- **`IngestionPolicy`** (Base): Abstract base class defining the interface.
- **`DefaultIngestionPolicy`**: Splits content by markdown headers and then by token count. fast and effective for structured markdown.
- **`QaExtractorPolicy`**: Leverage `llama_index` transformation pipeline. It uses:
    - `SentenceSplitter` for chunking.
    - `TitleExtractor` (LLM-based) to infer sections/titles.
    - `QuestionsAnsweredExtractor` (LLM-based) to generate questions that the chunk answers, enriching the text for retrieval.

### Data Model (`models.py`)
**`PolicyChunk`** is the core data unit.
- **Metadata**: `document_id`, `document_name`, `topic`, `country`, `active`, `last_modified`.
- **Content**:
    - `text`: The raw text of the chunk (enriched with QA pairs if using `QaExtractorPolicy`).
    - `text_indexed`: The text prepended with the `section_path_str`. This is what is actually embedded.
- **Structure**:
    - `section_path`: A list of strings representing the markdown header hierarchy.
    - `section_path_str`: A string representation of the path.

### Pipeline Logic (`pipeline.py`)
**`IngestionPipeline`** orchestrates the flow:
1. **Load**: Reads JSON metadata from `documents/`. Can optionally read content from an external markdown file if `markdown_file` matches.
2. **Policy Selection**: Instantiates the requested policy (default or qa_extractor).
3. **Processing**: Delegates content processing to the policy to produce `PolicyChunks`.
4. **Contract Generation**: Saves a local JSON file (`output/{filename}_chunks_{policy}.json`). This includes:
    - **Ingestion Metadata**: Pipeline version, date.
    - **Document Metadata**: Original file attributes.
    - **Chunk Statistics**: Detailed stats on counts, lengths, and section structure.
    - **Chunks**: The processed PolicyChunk objects.
5. **Ingest**: Delegates to `WeaviateStorage` to save the chunks.

## 4. Usage

### Running Ingestion
To run the ingestion process, use the CLI. You can specify which policy to use.

**Default Policy**:
```bash
python -m src.chat_rag.ingestion --policy default
```

**QA Extractor Policy** (requires `OPENAI_API_KEY`):
```bash
python -m src.chat_rag.ingestion --policy qa_extractor
```

**Prerequisites**:
- `OPENAI_API_KEY` must be set in your `.env` file or environment.
- Weaviate must be running (default `http://localhost:8080`).

### Generated Output
The pipeline produces JSON files in the `output/` directory (configurable in `config.py`).
Example naming: `policy_wfh_it_chunks_default.json` or `policy_wfh_it_chunks_qa_extractor.json`.

### Running Tests
Unit tests interface with `src.chat_rag.ingestion` but mock the database layer and LLM calls to be fast and independent.

```bash
uv run pytest tests/ingestion/test_ingestion.py
```
