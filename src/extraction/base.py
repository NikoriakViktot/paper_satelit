"""
Abstract base class for all extractors.
Defines the contract every extractor must satisfy.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field


@dataclass
class ExtractionResult:
    """Structured output for one flood-mapping paper."""
    source_file: str = ""

    # ── Bibliographic ─────────────────────────────────────────────────────────
    title:     str = ""
    authors:   str = ""
    doi:       str = ""
    year:      str = ""
    abstract:  str = ""
    full_text: str = ""

    # ── Study type ────────────────────────────────────────────────────────────
    study_type: str = ""

    # ── Satellite / sensor ────────────────────────────────────────────────────
    satellite_names: str = ""    # comma-separated
    sensor_type:     str = ""    # SAR | Optical | Multi-sensor
    data_product:    str = ""

    # ── Study area ────────────────────────────────────────────────────────────
    country:     str = ""
    region:      str = ""
    river_basin: str = ""
    river_name:  str = ""
    city_event:  str = ""

    # ── Geographic relevance ──────────────────────────────────────────────────
    geo_relevance:     str  = ""
    ukraine_relevance: bool = False

    # ── Method / processing ───────────────────────────────────────────────────
    methods: str = ""            # comma-separated list

    # ── Metrics ───────────────────────────────────────────────────────────────
    oa:    float | None = None
    f1:    float | None = None
    iou:   float | None = None
    kappa: float | None = None
    metrics_reported: bool = False

    # ── Timeliness ────────────────────────────────────────────────────────────
    latency:        str         = ""
    revisit_time:   str         = ""
    near_real_time: bool | None = None

    # ── Provenance (Task 3) ───────────────────────────────────────────────────
    # field_name → {section, snippet, source, extractor_mode, value}
    provenance: dict = field(default_factory=dict)

    # ── Extraction mode flags (Task 7) ────────────────────────────────────────
    extractor_mode: str  = ""     # "section" | "fallback" | "mixed"
    llm_used:       bool = False
    fallback_used:  bool = False

    # ── Quality scores (Task 6) ───────────────────────────────────────────────
    # quality_score: structural completeness (title, abstract, methods, results, satellite)
    # evidence_score: count of fields with valid section-based provenance
    quality_score:  float = 0.0
    evidence_score: int   = 0

    # ── QA ────────────────────────────────────────────────────────────────────
    confidence:               float     = 0.0
    evidence:                 list[str] = field(default_factory=list)
    sections_used:            list[str] = field(default_factory=list)
    missing_data_explanation: str       = ""

    def to_dict(self) -> dict:
        return {
            "Source_File":               self.source_file,
            "Title":                     self.title,
            "Authors":                   self.authors,
            "DOI":                       self.doi,
            "Year":                      self.year,
            "Abstract":                  self.abstract,
            "Full_Text":                 self.full_text,
            "Study_Type":                self.study_type,
            "Satellite_Names":           self.satellite_names,
            "Sensor_Type":               self.sensor_type,
            "Data_Product":              self.data_product,
            "Country":                   self.country,
            "Region":                    self.region,
            "River_Basin":               self.river_basin,
            "River_Name":                self.river_name,
            "City_Event":                self.city_event,
            "Geo_Relevance":             self.geo_relevance,
            "Ukraine_Relevance":         self.ukraine_relevance,
            "Methods":                   self.methods,
            "OA":                        self.oa,
            "F1":                        self.f1,
            "IoU":                       self.iou,
            "Kappa":                     self.kappa,
            "Metrics_Reported":          self.metrics_reported,
            "Latency":                   self.latency,
            "Revisit_Time":              self.revisit_time,
            "Near_Real_Time":            self.near_real_time,
            "Missing_Data_Explanation":  self.missing_data_explanation,
            "Sections_Used":             ", ".join(self.sections_used) if self.sections_used else "",
            # Task 7: extraction mode flags
            "Extractor_Mode":            self.extractor_mode,
            "LLM_Used":                  self.llm_used,
            "Fallback_Used":             self.fallback_used,
            # Task 6: quality scores
            "Quality_Score":             round(self.quality_score, 3),
            "Evidence_Score":            self.evidence_score,
            "Confidence":                round(self.confidence, 3),
            # Task 3: provenance (JSON string — nested dict not flat-CSV-safe)
            "Provenance_JSON":           _provenance_summary(self.provenance),
        }

    def _num_metrics(self) -> int:
        return sum(v is not None for v in [self.oa, self.f1, self.iou, self.kappa])

    def finalize(self) -> "ExtractionResult":
        """Compute derived fields before returning."""
        self.metrics_reported = self._num_metrics() > 0
        # evidence_score = fields that have section-level provenance with a snippet
        self.evidence_score = sum(
            1 for v in self.provenance.values()
            if isinstance(v, dict) and v.get("section") and v.get("snippet")
        )
        return self


def _provenance_summary(provenance: dict) -> str:
    """Compact single-line summary of field→section mapping for CSV storage."""
    if not provenance:
        return ""
    parts = []
    for field_name, prov in provenance.items():
        if isinstance(prov, dict):
            section = prov.get("section", "?")
            source  = prov.get("source", "?")
            parts.append(f"{field_name}@{section}[{source}]")
    return "; ".join(parts)


class BaseExtractor(ABC):
    """All extractors implement this interface."""

    @abstractmethod
    def extract(
        self,
        chunks: list[dict],
        source_file: str,
    ) -> ExtractionResult:
        """
        Extract structured information from a list of retrieved chunks.

        Parameters
        ----------
        chunks:
            Each dict has at least {"text": str, "filename": str}
        source_file:
            Original PDF filename

        Returns
        -------
        ExtractionResult (finalized)
        """
