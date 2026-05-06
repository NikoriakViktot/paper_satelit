"""
method_matcher.py
-----------------
Ontology-driven method detection and enrichment.

Responsibilities:
  - Match methods via ontology_methods.json aliases and descriptions
  - Map detected methods to their inputs/outputs from the ontology
  - Attach full method metadata to matches
  - Identify method pipelines from the remote_sensing_water_resources domain structure
"""

from __future__ import annotations

import re
import logging
from typing import Optional

from .knowledge_loader import KnowledgeBase, MethodRecord

log = logging.getLogger(__name__)

_POSITIVE_CONTEXT = [
    "used", "applied", "we used", "this study", "using",
    "was applied", "is applied", "were used",
    "we employ", "we adopt", "we utilize",
    "method", "algorithm", "model",
]
_NEGATIVE_CONTEXT = [
    "previous studies", "other studies", "review", "e.g.", "et al",
    "for example", "such as", "proposed by", "introduced by",
]


def _snippet(text: str, start: int, end: int, window: int = 250) -> str:
    return re.sub(
        r"\s+", " ",
        text[max(0, start - window): min(len(text), end + window)]
    ).strip()


def _is_real_usage(ctx: str) -> bool:
    ctx_l = ctx.lower()
    return (
        any(k in ctx_l for k in _POSITIVE_CONTEXT) and
        not any(k in ctx_l for k in _NEGATIVE_CONTEXT)
    )


class MethodMatcher:
    """
    Detects analysis methods, models, and algorithms from text using
    the ontology knowledge base.

    Usage:
        matcher = MethodMatcher(kb)
        methods = matcher.match(text)
        # Each item: {name, type, inputs, outputs, domain, evidence, ...}
    """

    def __init__(self, kb: KnowledgeBase):
        self.kb = kb

    def match(self, text: str, strict: bool = True) -> list[dict]:
        """
        Detect and enrich methods from text.

        Returns list of dicts with full ontology metadata.
        """
        results: dict[str, dict] = {}

        for mrec in self.kb.methods:
            if not mrec.pattern:
                continue
            try:
                for m in re.finditer(mrec.pattern, text, re.IGNORECASE):
                    ctx = _snippet(text, m.start(), m.end())
                    if strict and not _is_real_usage(ctx):
                        continue

                    key = mrec.method_name.lower()
                    if key not in results:
                        results[key] = self._build_result(mrec, ctx)
            except re.error as exc:
                log.debug("Pattern error %s: %s", mrec.method_name, exc)

        return list(results.values())

    def get_method_inputs(self, method_name: str) -> list[str]:
        """Return input variables for a method."""
        rec = self.kb.resolve_method(method_name)
        return rec.inputs if rec else []

    def get_method_outputs(self, method_name: str) -> list[str]:
        """Return output variables for a method."""
        rec = self.kb.resolve_method(method_name)
        return rec.outputs if rec else []

    def get_related_methods(self, method_name: str) -> list[str]:
        """Return related methods from ontology."""
        rec = self.kb.resolve_method(method_name)
        return rec.related if rec else []

    def infer_pipeline(self, methods: list[dict]) -> list[str]:
        """
        Identify the processing pipeline implied by detected methods.
        Uses the rs_domains structure from the knowledge base.
        """
        detected_names = {m["name"].lower() for m in methods}
        pipelines: list[str] = []

        # check flood modeling pipeline
        flood_kw = {
            "runoff", "routing", "hec", "swat", "lisflood",
            "precipitation", "hydrograph",
        }
        if any(any(kw in name for kw in flood_kw) for name in detected_names):
            pipelines.append("flood_modeling_pipeline")

        # check drought/veg pipeline
        veg_kw = {"ndvi", "evi", "savi", "vci", "pdsi", "spi"}
        if any(any(kw in name for kw in veg_kw) for name in detected_names):
            pipelines.append("drought_vegetation_pipeline")

        # check terrain analysis pipeline
        terrain_kw = {"dem", "slope", "twi", "hand", "srtm", "dtm"}
        if any(any(kw in name for kw in terrain_kw) for name in detected_names):
            pipelines.append("terrain_analysis_pipeline")

        return pipelines

    @staticmethod
    def _build_result(mrec: MethodRecord, ctx: str) -> dict:
        return {
            "name":        mrec.method_name,
            "type":        "method",
            "source":      "methods+results",
            "confidence":  0.88,
            "evidence":    ctx,
            "scores": {
                "pattern":   0.88,
                "context":   0.0,
                "embedding": 0.0,
                "llm":       0.0,
            },
            "final_score": 0.0,
            "accepted":    None,
            "role":        None,
            "kb_metadata": {
                "full_name":  mrec.method_name.replace("_", " ").title(),
                "type":       mrec.type,
                "type_group": "method",
                "domain":     mrec.domain,
                "subdomain":  mrec.subdomain,
                "definition": mrec.description,
                "inputs":     mrec.inputs,
                "outputs":    mrec.outputs,
                "related":    mrec.related,
                "aliases":    mrec.aliases,
                "source_kb":  mrec.source_kb,
            },
        }
