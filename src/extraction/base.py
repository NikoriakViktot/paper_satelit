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
    # Satellite flood mapping | ML/DL classification | Hydrological forecasting
    # Hydraulic modeling | Operational mapping system | Review paper
    # Dataset/benchmark paper
    study_type: str = ""

    # ── Satellite / sensor ────────────────────────────────────────────────────
    satellite_names: str = ""    # comma-separated, e.g. "Sentinel-1, Sentinel-2"
    sensor_type:     str = ""    # SAR | Optical | Multi-sensor
    data_product:    str = ""    # GRD, MSI, OLI, …

    # ── Study area ────────────────────────────────────────────────────────────
    country:     str = ""
    region:      str = ""        # sub-national area, basin region, or continent
    river_basin: str = ""
    river_name:  str = ""        # specific river name(s), comma-separated
    city_event:  str = ""

    # ── Geographic relevance ──────────────────────────────────────────────────
    # Ukraine-specific | Eastern Europe | Europe | Global | Other region | Unspecified
    geo_relevance:     str  = ""
    ukraine_relevance: bool = False

    # ── Method / processing ───────────────────────────────────────────────────
    methods: str = ""            # comma-separated list

    # ── Metrics (optional — only when explicitly reported) ────────────────────
    oa:    float | None = None
    f1:    float | None = None
    iou:   float | None = None
    kappa: float | None = None
    metrics_reported: bool = False

    # ── Timeliness ────────────────────────────────────────────────────────────
    latency:        str         = ""    # e.g. "<6 h", "1–3 days"
    revisit_time:   str         = ""    # e.g. "6 days (Sentinel-1)"
    near_real_time: bool | None = None

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
            "Confidence":                round(self.confidence, 3),
        }

    def _num_metrics(self) -> int:
        return sum(v is not None for v in [self.oa, self.f1, self.iou, self.kappa])

    def finalize(self) -> "ExtractionResult":
        """Compute derived fields before returning."""
        self.metrics_reported = self._num_metrics() > 0
        return self


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
