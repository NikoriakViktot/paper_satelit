# Flood-Paper RAG Pipeline

A **fully local** Retrieval-Augmented Generation (RAG) system for extracting structured accuracy metrics from flood-mapping research papers.

```
PDF  →  text  →  chunks  →  embeddings  →  ChromaDB  →  retrieval  →  extraction  →  CSV
```

No cloud APIs. No data leaves your machine.

---

## Features

| Capability | Detail |
|---|---|
| PDF ingestion | PyMuPDF — handles multi-column, ligatures, soft hyphens |
| Chunking | Sliding-window (2000 chars / ~500 tok, 300-char overlap) |
| Embeddings | `sentence-transformers/all-MiniLM-L6-v2` (local, 384-D) |
| Vector DB | ChromaDB (persistent on disk) |
| Retrieval | Multi-query semantic search, deduplication by chunk_id |
| Extraction | Regex rules **or** local LLM via Ollama |
| Output | Pandas DataFrame + CSV |

---

## Project Structure

```
paper_satelit/
├── data/
│   └── literature/          ← drop your PDF files here
│
├── src/
│   ├── config.py            ← all tuneable settings
│   ├── ingestion/
│   │   └── pdf_reader.py    ← PyMuPDF extraction
│   ├── processing/
│   │   └── chunker.py       ← sliding-window chunker
│   ├── embedding/
│   │   └── embedder.py      ← sentence-transformers wrapper
│   ├── vectorstore/
│   │   └── chroma_store.py  ← ChromaDB CRUD
│   ├── retrieval/
│   │   └── retriever.py     ← multi-query semantic search
│   ├── extraction/
│   │   ├── base.py          ← ExtractionResult dataclass + ABC
│   │   ├── regex_extractor.py   ← rule-based (offline)
│   │   └── ollama_extractor.py  ← LLM-backed (optional)
│   ├── pipeline/
│   │   └── rag_pipeline.py  ← orchestrator
│   └── utils/
│       └── logging_config.py
│
├── notebooks/
│   └── query_example.ipynb  ← interactive exploration
├── outputs/                 ← extracted CSV lands here
├── main.py                  ← CLI entry point
└── requirements.txt
```

---

## Installation

```bash
# 1. Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. (Optional) Install Ollama for LLM extraction
curl -fsSL https://ollama.ai/install.sh | sh
ollama pull llama3.2   # ~2 GB
```

---

## Usage

### 1. Add PDFs

```bash
cp /path/to/your/papers/*.pdf data/literature/
```

### 2. Ingest (index all PDFs)

```bash
python main.py --ingest
```

To re-index from scratch:

```bash
python main.py --ingest --force
```

### 3. Query (extract structured data)

```bash
# Default: rule-based extractor (always works)
python main.py --query

# With local LLM (requires Ollama running)
python main.py --query --extractor ollama
```

### 4. Full pipeline in one command

```bash
python main.py --ingest --query
```

### 5. Status check

```bash
python main.py --status
```

Results are saved to `outputs/flood_papers_extracted.csv`.

---

## Output Schema

| Column | Type | Description |
|---|---|---|
| Author | str | First author (extracted from text or filename) |
| Method | str | Mapping method (U-Net, Random Forest, …) |
| Sensor | str | Sensor(s) used (Sentinel-1, Landsat, …) |
| Region | str | Study area |
| OA | float \| NaN | Overall Accuracy (0–1 scale) |
| F1 | float \| NaN | F1-score (0–1 scale) |
| IoU | float \| NaN | Intersection over Union (0–1 scale) |
| Kappa | float \| NaN | Cohen's Kappa (0–1 scale) |
| Accuracy_Level | str | Quantitative / Semi-quantitative / Qualitative |
| Accuracy_Desc | str | Human-readable accuracy summary |
| Source_File | str | Original PDF filename |
| Confidence | float | Extractor confidence score (0–1) |

**Accuracy_Level** logic:
- **Quantitative** — ≥ 2 numeric metrics extracted
- **Semi-quantitative** — exactly 1 numeric metric
- **Qualitative** — no numeric metrics found

---

## Configuration

Override any setting via environment variable or edit `src/config.py`:

```bash
CHUNK_SIZE=1800          # characters per chunk
CHUNK_OVERLAP=250        # overlap between chunks
EMBEDDING_DEVICE=cuda    # use GPU for embeddings
RETRIEVAL_TOP_K=20       # chunks retrieved per query
OLLAMA_MODEL=llama3.1    # Ollama model name
LOG_LEVEL=DEBUG          # verbosity
```

---

## Pipeline Diagram

```
┌─────────────┐     ┌───────────┐     ┌──────────────┐     ┌──────────────┐
│  PDF files  │────▶│ pdf_reader│────▶│   chunker    │────▶│   embedder   │
│ (data/lit/) │     │ (PyMuPDF) │     │ (2000c win.) │     │ (MiniLM-L6) │
└─────────────┘     └───────────┘     └──────────────┘     └──────┬───────┘
                                                                   │
                                                                   ▼
                                                          ┌──────────────────┐
                                                          │    ChromaDB      │
                                                          │  (.chromadb/)    │
                                                          └────────┬─────────┘
                                                                   │
                         ┌─────────────────────────────────────────┘
                         ▼
              ┌────────────────────┐     ┌───────────────────────┐
              │  multi-query       │────▶│  RegexExtractor  OR   │
              │  retriever         │     │  OllamaExtractor      │
              │  (6 queries)       │     │  (+ regex fallback)   │
              └────────────────────┘     └───────────┬───────────┘
                                                     │
                                                     ▼
                                          ┌──────────────────────┐
                                          │  ExtractionResult    │
                                          │  → DataFrame → CSV   │
                                          │  (outputs/)          │
                                          └──────────────────────┘
```

---

## Extending the Pipeline

### Add a new extractor

```python
# src/extraction/my_extractor.py
from src.extraction.base import BaseExtractor, ExtractionResult

class MyExtractor(BaseExtractor):
    def extract(self, chunks, source_file):
        result = ExtractionResult(source_file=source_file)
        # ... your logic ...
        return result.finalize()
```

Then pass `extractor_type="my"` in `RAGPipeline.__init__` after registering it.

### Change the embedding model

```bash
EMBEDDING_MODEL=BAAI/bge-small-en-v1.5 python main.py --ingest --force
```

---

## License

MIT
