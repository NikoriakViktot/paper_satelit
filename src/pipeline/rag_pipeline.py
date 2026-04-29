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
from src.extraction.models import FactExtractionResult
from src.extraction.ollama_extractor import OllamaExtractor
from src.extraction.regex_extractor import RegexExtractor
from src.extraction.scientific_extractor import ScientificExtractor
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
        elif extractor_type == "scientific":
            self._extractor = ScientificExtractor()
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
        DataFrame with columns matching ExtractionResult.to_dict() plus
        post-processing columns (Geo_Relevance, Extraction_Score, etc.)
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
        rejected_count = 0

        for filename, chunks in by_file.items():
            logger.info("Extracting from: %s (%d chunks)", filename, len(chunks))
            result = self._extractor.extract(chunks, source_file=filename)

            # ── Validation layer (Task 8) ─────────────────────────────────────
            is_valid, reason = _validate_extraction(result)
            if not is_valid:
                logger.warning("[VALIDATION REJECTED] %s: %s", filename, reason)
                print(f"  [REJECTED] {filename}: {reason}")
                result.missing_data_explanation = f"VALIDATION: {reason}"
                result.confidence = 0.0
                rejected_count += 1

            if isinstance(self._extractor, SectionExtractor):
                print(f"    Sections used:    {result.sections_used or '—'}")

            logger.debug(
                "  → study_type=%s  satellite=%s  country=%s  methods=%s  "
                "NRT=%s  OA=%s  F1=%s  mode=%s",
                result.study_type, result.satellite_names, result.country,
                result.methods, result.near_real_time, result.oa, result.f1,
                result.extractor_mode,
            )
            results.append(result)

        if rejected_count:
            logger.info("Validation rejected %d / %d papers.", rejected_count, len(results))

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

    # ── Fact-centric extraction ───────────────────────────────────────────────

    def query_facts(
        self,
        write_graph: bool = False,
        save_json: bool = True,
    ) -> list[FactExtractionResult]:
        """
        Run the fact-centric extraction pipeline.

        For each PDF in the vector store, extracts structured ScientificFact
        objects with full Evidence provenance, then optionally writes them to
        Neo4j and/or saves a JSON file.

        Returns
        -------
        list[FactExtractionResult]
        """
        if self._store.count() == 0:
            raise RuntimeError("Vector store is empty — run --ingest first.")

        if not isinstance(self._extractor, ScientificExtractor):
            logger.warning(
                "query_facts() requires ScientificExtractor.  "
                "Switching automatically."
            )
            self._extractor = ScientificExtractor()

        t0 = time.perf_counter()
        by_file = self._retriever.fetch_full_context()

        fact_results: list[FactExtractionResult] = []

        for filename, chunks in by_file.items():
            logger.info("Fact extraction from: %s (%d chunks)", filename, len(chunks))
            fact_result = self._extractor.extract_facts(chunks, source_file=filename)
            fact_results.append(fact_result)

        elapsed = time.perf_counter() - t0
        total_facts = sum(r.fact_count for r in fact_results)
        total_rejected = sum(len(r.rejected_facts) for r in fact_results)
        logger.info(
            "Fact extraction complete in %.1f s — %d papers, %d facts, %d rejected.",
            elapsed, len(fact_results), total_facts, total_rejected,
        )

        if write_graph:
            from src.graphstore.neo4j_writer import Neo4jWriter
            with Neo4jWriter() as gdb:
                gdb.write_fact_papers(fact_results)

        if save_json:
            import json
            out_path = self._output_csv.with_suffix(".facts.json")
            out_path.parent.mkdir(parents=True, exist_ok=True)
            with open(out_path, "w", encoding="utf-8") as fh:
                json.dump(
                    [r.to_dict() for r in fact_results],
                    fh,
                    ensure_ascii=False,
                    indent=2,
                )
            logger.info("Facts JSON saved → %s", out_path)

        return fact_results


# ── Validation layer (Task 8) ─────────────────────────────────────────────────

def _validate_extraction(result: ExtractionResult) -> tuple[bool, str]:
    """
    Hard validation rules (Task 8).

    Returns (is_valid, rejection_reason).
    A result is invalid if:
      - Metrics have explicit wrong-section provenance
      - Methods have explicit wrong-section provenance
      - Country is the generic string "study area"
    """
    provenance = result.provenance or {}

    # Rule 1: metrics must come from results section
    for field_name, attr in [("OA","oa"),("F1","f1"),("IoU","iou"),("Kappa","kappa")]:
        value = getattr(result, attr, None)
        prov  = provenance.get(field_name, {})
        # Only fail when provenance explicitly shows the WRONG section
        if value is not None and prov.get("section") and prov["section"] != "results":
            return False, (
                f"{field_name}={value} was extracted from '{prov['section']}' "
                f"but must come from 'results' section"
            )

    # Rule 2: methods must come from methods section
    methods_prov = provenance.get("Methods", {})
    if (result.methods
            and methods_prov.get("section")
            and methods_prov["section"] not in ("methods", "abstract+methods")):
        return False, (
            f"Methods were extracted from '{methods_prov['section']}' "
            f"but must come from 'methods' section"
        )

    # Rule 3: country == "study area" is not informative
    if result.country:
        norm = result.country.lower().strip().rstrip(".")
        if norm in ("study area", "the study area", "study region"):
            return False, f"Country field is generic '{result.country}' — not a real country"

    return True, ""


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
    # Task 7: extraction mode flags
    "Extractor_Mode", "LLM_Used", "Fallback_Used",
    # Task 6: quality scores
    "Quality_Score", "Evidence_Score",
    # Task 3: provenance summary
    "Provenance_JSON",
    "Confidence", "Extraction_Score",
]


def _post_process(df: pd.DataFrame) -> pd.DataFrame:
    # 1. Validate Abstract / DOI
    df = clean_metadata_df(df)

    # 2. Normalize Sensor_Type / Methods / Study_Type + force-fill Country
    df = normalize_fields_df(df)

    # 2b. Geo_Relevance from all geographic fields
    def _geo(row: pd.Series) -> str:
        return classify_geo_relevance(
            row.get("Country",           "") or "",
            row.get("Region",            "") or "",
            row.get("River_Basin",       "") or "",
            row.get("River_Name",        "") or "",
            bool(row.get("Ukraine_Relevance", False)),
        )
    df["Geo_Relevance"] = df.apply(_geo, axis=1)

    # 3. Drop Full_Text before column filtering (too large for CSV)
    df = df.drop(columns=["Full_Text"], errors="ignore")

    # 4. Score rows and drop low-quality entries
    df = apply_consistency_filter(df, min_score=2)

    # 5. Ensure all output columns exist, then select
    for col in _OUTPUT_COLUMNS:
        if col not in df.columns:
            df[col] = None
    df = df[_OUTPUT_COLUMNS].copy()

    # 6. Replace empty strings with None in string columns
    str_cols = [
        "Title", "Authors", "DOI", "Year", "Abstract",
        "Study_Type", "Satellite_Names", "Sensor_Type", "Data_Product",
        "Geo_Relevance", "Country", "Region", "River_Basin", "River_Name", "City_Event",
        "Methods", "Latency", "Revisit_Time", "Missing_Data_Explanation",
        "Extractor_Mode", "Provenance_JSON",
    ]
    for col in str_cols:
        if col in df.columns:
            df[col] = df[col].replace("", None)

    # 7. Sort: Near_Real_Time first, then metric count, then confidence
    df["_nrt_rank"]   = df["Near_Real_Time"].apply(lambda v: 0 if v is True else 1)
    df["_metric_cnt"] = df[["OA", "F1", "IoU", "Kappa"]].notna().sum(axis=1)
    df = df.sort_values(["_nrt_rank", "_metric_cnt", "Confidence"],
                        ascending=[True, False, False])
    df = df.drop(columns=["_nrt_rank", "_metric_cnt"]).reset_index(drop=True)

    print_report(df)
    return df
