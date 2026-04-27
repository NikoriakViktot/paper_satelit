"""
Abstract base class for all extractors.
Defines the contract every extractor must satisfy.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field


@dataclass
class ExtractionResult:
    """Structured output for one paper."""
    source_file: str = ""
    author: str      = ""
    method: str      = ""
    sensor: str      = ""
    region: str      = ""

    # bibliographic metadata
    title:     str = ""
    abstract:  str = ""
    doi:       str = ""
    full_text: str = ""

    # scientific explanation for any field that could not be populated
    missing_data_explanation: str = ""

    # numeric metrics (None = not found)
    oa:    float | None = None
    f1:    float | None = None
    iou:   float | None = None
    kappa: float | None = None

    # human-readable description of accuracy when numeric is absent
    accuracy_desc: str = ""

    # Quantitative / Semi-quantitative / Qualitative
    accuracy_level: str = "Qualitative"

    # 0–1: how confident the extractor is
    confidence: float = 0.0

    # raw text snippets used for this result (for auditing)
    evidence: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "Source_File":               self.source_file,
            "Title":                     self.title,
            "Authors":                   self.author,
            "DOI":                       self.doi,
            "Abstract":                  self.abstract,
            "Full_Text":                 self.full_text,
            "Method":                    self.method,
            "Sensor":                    self.sensor,
            "Region":                    self.region,
            "OA":                        self.oa,
            "F1":                        self.f1,
            "IoU":                       self.iou,
            "Kappa":                     self.kappa,
            "Accuracy_Level":            self.accuracy_level,
            "Accuracy_Desc":             self.accuracy_desc,
            "Missing_Data_Explanation":  self.missing_data_explanation,
            "Confidence":                round(self.confidence, 3),
        }

    def _num_metrics(self) -> int:
        return sum(v is not None for v in [self.oa, self.f1, self.iou, self.kappa])

    def _classify_level(self) -> str:
        n = self._num_metrics()
        if n >= 2:
            return "Quantitative"
        if n == 1:
            return "Semi-quantitative"
        return "Qualitative"

    def finalize(self) -> "ExtractionResult":
        """Compute derived fields before returning."""
        self.accuracy_level = self._classify_level()
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
            Original PDF filename (used to populate ExtractionResult.source_file)

        Returns
        -------
        ExtractionResult (finalized)
        """
