"""
Central configuration for the Flood-Paper RAG system.
All values can be overridden via environment variables.
"""
import os
from pathlib import Path

# ── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR    = Path(__file__).parent.parent
DATA_DIR    = BASE_DIR / "data" / "literature"
OUTPUTS_DIR = BASE_DIR / "outputs"
CHROMA_DIR  = BASE_DIR / ".chromadb"

# ── Chunking ─────────────────────────────────────────────────────────────────
CHUNK_SIZE    = int(os.getenv("CHUNK_SIZE",    "2000"))   # characters ≈ 500 tok
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "300"))
CHUNK_MIN     = int(os.getenv("CHUNK_MIN",     "120"))    # discard tiny shards

# ── Embeddings ────────────────────────────────────────────────────────────────
EMBEDDING_MODEL      = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
EMBEDDING_BATCH_SIZE = int(os.getenv("EMBEDDING_BATCH_SIZE", "64"))
EMBEDDING_DEVICE     = os.getenv("EMBEDDING_DEVICE", "cpu")   # or "cuda"

# ── ChromaDB ─────────────────────────────────────────────────────────────────
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "flood_papers")

# ── Retrieval ────────────────────────────────────────────────────────────────
RETRIEVAL_TOP_K = int(os.getenv("RETRIEVAL_TOP_K", "12"))

RETRIEVAL_QUERIES = [
    "flood mapping satellite remote sensing sensor study area method",
    "SAR Sentinel-1 flood detection thresholding change detection backscatter",
    "Sentinel-2 Landsat optical flood mapping NDWI MNDWI water index",
    "near-real-time operational flood monitoring latency revisit time",
    "U-Net CNN deep learning flood segmentation classification",
    "hydrological hydrodynamic hydraulic model flood simulation HEC-RAS LISFLOOD",
    "flood extent mapping accuracy validation OA F1 IoU Kappa",
    "flood event study area country region river basin city",
]

# ── LLM (Ollama) ──────────────────────────────────────────────────────────────
OLLAMA_BASE_URL   = os.getenv("OLLAMA_BASE_URL",   "http://localhost:11434")
OLLAMA_MODEL      = os.getenv("OLLAMA_MODEL",      "llama3.2")
OLLAMA_TIMEOUT    = int(os.getenv("OLLAMA_TIMEOUT", "60"))

# ── Output ────────────────────────────────────────────────────────────────────
OUTPUT_CSV = OUTPUTS_DIR / "flood_papers_extracted.csv"

# ── Logging ───────────────────────────────────────────────────────────────────
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
LOG_FILE  = BASE_DIR / "rag_pipeline.log"
