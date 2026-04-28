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
from src.extraction.section_extractor import SectionExtractor
from src.ingestion.pdf_reader import read_all_pdfs
from src.processing.chunker import chunk_documents
from src.processing.analytics import classify_geo_relevance
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
        extractor_type: str = "regex",   # "regex" | "ollama" | "section"
        data_dir: Path = DATA_DIR,
        output_csv: Path = OUTPUT_CSV,
    ) -> None:
        self._data_dir   = Path(data_dir)
        self._output_csv = Path(output_csv)

        logger.info("Initialising components …")
        self._embedder  = Embedder()
        self._store     = VectorStore()
        self._retriever = Retriever(self._store, self._embedder)

        if extractor_type == "ollama":
            self._extractor: BaseExtractor = OllamaExtractor()
        elif extractor_type == "section":
            self._extractor = SectionExtractor()
        else:
            self._extractor = RegexExtractor()

        logger.info("Extractor: %s", type(self._extractor).__name__)

    # ── Ingest ────────────────────────────────────────────────────────────────

    def ingest(self, folder: Path | None = None, force: bool = False) -> int:
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

        if isinstance(self._extractor, SectionExtractor):
            logger.info("Section extractor active — fetching full document context.")
            by_file = self._retriever.fetch_full_context()
        else:
            by_file = self._retriever.retrieve_for_queries(queries=queries, top_k=top_k)

        results: list[ExtractionResult] = []
        for filename, chunks in by_file.items():
            logger.info("Extracting from: %s (%d chunks)", filename, len(chunks))
            result = self._extractor.extract(chunks, source_file=filename)
            results.append(result)
            if isinstance(self._extractor, SectionExtractor):
                print(f"    Sections used:    {result.sections_used or '—'}")
            logger.debug(
                "  → study_type=%s  satellite=%s  country=%s  methods=%s  "
                "NRT=%s  OA=%s  F1=%s",
                result.study_type, result.satellite_names, result.country,
                result.methods, result.near_real_time, result.oa, result.f1,
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
    "Source_File", "Title", "Authors", "DOI", "Year",
    "Abstract", "Abstract_Valid",
    "Study_Type",
    "Satellite_Names", "Sensor_Type", "Data_Product",
    "Geo_Relevance", "Ukraine_Relevance",
    "Country", "Region", "River_Basin", "River_Name", "City_Event",
    "Methods",
    "OA", "F1", "IoU", "Kappa", "Metrics_Reported",
    "Latency", "Revisit_Time", "Near_Real_Time",
    "DOI_Valid",
    "Missing_Data_Explanation",
    "Sections_Used",
    "Confidence", "Extraction_Score",
]


def _post_process(df: pd.DataFrame) -> pd.DataFrame:
    # 1. validate Abstract/DOI
    df = clean_metadata_df(df)

    # 2. normalize Sensor_Type / Methods / Study_Type + force-fill Country
    df = normalize_fields_df(df)

    # 2b. Compute Geo_Relevance from all geographic fields
    def _geo(row: pd.Series) -> str:
        return classify_geo_relevance(
            row.get("Country",           "") or "",
            row.get("Region",            "") or "",
            row.get("River_Basin",       "") or "",
            row.get("River_Name",        "") or "",
            bool(row.get("Ukraine_Relevance", False)),
        )
    df["Geo_Relevance"] = df.apply(_geo, axis=1)

    # 3. drop Full_Text before column filtering (too large for CSV)
    df = df.drop(columns=["Full_Text"], errors="ignore")

    # 4. score rows and drop low-quality entries
    df = apply_consistency_filter(df, min_score=2)

    # 5. ensure all output columns exist, then select
    for col in _OUTPUT_COLUMNS:
        if col not in df.columns:
            df[col] = None
    df = df[_OUTPUT_COLUMNS].copy()

    # 6. replace empty strings with None
    str_cols = [
        "Title", "Authors", "DOI", "Year", "Abstract",
        "Study_Type", "Satellite_Names", "Sensor_Type", "Data_Product",
        "Geo_Relevance", "Country", "Region", "River_Basin", "River_Name", "City_Event",
        "Methods", "Latency", "Revisit_Time", "Missing_Data_Explanation",
    ]
    for col in str_cols:
        if col in df.columns:
            df[col] = df[col].replace("", None)

    # 7. sort: Near_Real_Time first, then by number of metrics, then confidence
    df["_nrt_rank"]   = df["Near_Real_Time"].apply(lambda v: 0 if v is True else 1)
    df["_metric_cnt"] = df[["OA", "F1", "IoU", "Kappa"]].notna().sum(axis=1)
    df = df.sort_values(["_nrt_rank", "_metric_cnt", "Confidence"],
                        ascending=[True, False, False])
    df = df.drop(columns=["_nrt_rank", "_metric_cnt"]).reset_index(drop=True)

    print_report(df)
    return df
