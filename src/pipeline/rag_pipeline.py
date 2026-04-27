"""
Top-level pipeline orchestrator.

Two public methods:
  • ingest(folder)  — PDF → chunks → embeddings → ChromaDB
  • query()         — ChromaDB → retrieval → extraction → DataFrame / CSV
"""
from __future__ import annotations

import logging
import time
from pathlib import Path

import pandas as pd

from src.config import DATA_DIR, OUTPUT_CSV, RETRIEVAL_QUERIES, RETRIEVAL_TOP_K
from src.embedding.embedder import Embedder
from src.extraction.base import BaseExtractor, ExtractionResult
from src.extraction.ollama_extractor import OllamaExtractor
from src.extraction.regex_extractor import RegexExtractor
from src.ingestion.pdf_reader import read_all_pdfs
from src.processing.chunker import chunk_documents
from src.processing.metadata_cleaner import (
    apply_consistency_filter,
    clean_metadata_df,
    normalize_fields_df,
    print_report,
)
from src.retrieval.retriever import Retriever
from src.vectorstore.chroma_store import VectorStore

logger = logging.getLogger(__name__)


class RAGPipeline:
    """Coordinates all pipeline components."""

    def __init__(
        self,
        extractor_type: str = "regex",   # "regex" | "ollama"
        data_dir: Path = DATA_DIR,
        output_csv: Path = OUTPUT_CSV,
    ) -> None:
        self._data_dir   = Path(data_dir)
        self._output_csv = Path(output_csv)

        logger.info("Initialising components …")
        self._embedder = Embedder()
        self._store    = VectorStore()
        self._retriever = Retriever(self._store, self._embedder)
        self._extractor: BaseExtractor = (
            OllamaExtractor() if extractor_type == "ollama" else RegexExtractor()
        )
        logger.info("Extractor: %s", type(self._extractor).__name__)

    # ── Ingest ────────────────────────────────────────────────────────────────

    def ingest(self, folder: Path | None = None, force: bool = False) -> int:
        """
        Read all PDFs from *folder*, embed, and upsert into ChromaDB.

        Parameters
        ----------
        folder: override default DATA_DIR
        force:  if True, clear existing collection before ingesting

        Returns
        -------
        Number of chunks stored.
        """
        folder = Path(folder) if folder else self._data_dir
        t0 = time.perf_counter()

        if force:
            logger.warning("Force flag set — clearing existing collection.")
            self._store.clear()

        pages  = read_all_pdfs(folder)
        if not pages:
            logger.error("No pages extracted — aborting ingest.")
            return 0

        chunks = chunk_documents(pages)
        if not chunks:
            logger.error("No chunks produced — aborting ingest.")
            return 0

        embeddings = self._embedder.embed_chunks(chunks)
        self._store.upsert(chunks, embeddings)

        elapsed = time.perf_counter() - t0
        logger.info("Ingest complete in %.1f s — %d chunks stored.", elapsed, len(chunks))
        return len(chunks)

    # ── Query / Extract ───────────────────────────────────────────────────────

    def query(
        self,
        queries: list[str] | None = None,
        top_k: int = RETRIEVAL_TOP_K,
        save_csv: bool = True,
    ) -> pd.DataFrame:
        """
        Retrieve relevant chunks for every PDF and extract structured data.

        Returns
        -------
        DataFrame with columns matching ExtractionResult.to_dict()
        """
        if self._store.count() == 0:
            raise RuntimeError("Vector store is empty — run --ingest first.")

        queries = queries or RETRIEVAL_QUERIES
        t0 = time.perf_counter()

        by_file = self._retriever.retrieve_for_queries(queries=queries, top_k=top_k)

        results: list[ExtractionResult] = []
        for filename, chunks in by_file.items():
            logger.info("Extracting from: %s (%d chunks)", filename, len(chunks))
            result = self._extractor.extract(chunks, source_file=filename)
            results.append(result)
            logger.debug(
                "  → OA=%s  F1=%s  IoU=%s  Kappa=%s  Level=%s",
                result.oa, result.f1, result.iou, result.kappa, result.accuracy_level,
            )

        df = pd.DataFrame([r.to_dict() for r in results])
        df = _post_process(df)

        elapsed = time.perf_counter() - t0
        logger.info(
            "Extraction complete in %.1f s — %d papers processed.",
            elapsed, len(df),
        )

        if save_csv:
            self._output_csv.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(self._output_csv, index=False)
            logger.info("CSV saved → %s", self._output_csv)

        return df

    # ── Targeted single-file query ────────────────────────────────────────────

    def query_file(self, filename: str) -> ExtractionResult:
        """Extract from one specific PDF by filename."""
        chunks = self._retriever.retrieve_for_file(filename)
        return self._extractor.extract(chunks, source_file=filename)


# ── Post-processing ───────────────────────────────────────────────────────────

_OUTPUT_COLUMNS = [
    "Source_File", "Title", "Authors", "DOI", "DOI_Valid",
    "Abstract", "Abstract_Valid",
    "Method", "Sensor", "Region",
    "OA", "F1", "IoU", "Kappa",
    "Accuracy_Level", "Accuracy_Desc",
    "Missing_Data_Explanation",
    "Confidence", "Extraction_Score",
]


def _post_process(df: pd.DataFrame) -> pd.DataFrame:
    # 1. validate Abstract/DOI — needs Full_Text, adds _Valid columns
    df = clean_metadata_df(df)

    # 2. normalize Method/Sensor/Region + force region from Full_Text
    df = normalize_fields_df(df)

    # 3. drop Full_Text before column filtering (too large for CSV)
    df = df.drop(columns=["Full_Text"], errors="ignore")

    # 4. score rows and drop low-quality entries
    df = apply_consistency_filter(df, min_score=2)

    # 5. ensure all output columns exist, then select them
    for col in _OUTPUT_COLUMNS:
        if col not in df.columns:
            df[col] = None
    df = df[_OUTPUT_COLUMNS].copy()

    # 6. replace empty strings with None
    str_cols = [
        "Title", "Authors", "DOI", "Abstract",
        "Method", "Sensor", "Region", "Accuracy_Desc",
        "Missing_Data_Explanation",
    ]
    for col in str_cols:
        if col in df.columns:
            df[col] = df[col].replace("", None)

    # 7. sort: Quantitative → Semi → Qualitative, then OA desc
    level_order = {"Quantitative": 0, "Semi-quantitative": 1, "Qualitative": 2}
    df["_level_rank"] = df["Accuracy_Level"].map(level_order).fillna(3)
    df = df.sort_values(["_level_rank", "OA"], ascending=[True, False])
    df = df.drop(columns="_level_rank").reset_index(drop=True)

    print_report(df)
    return df
