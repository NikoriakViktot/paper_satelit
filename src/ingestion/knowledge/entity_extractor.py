"""
entity_extractor.py
-------------------
Generic, KB-driven entity extraction engine.

Replaces all hardcoded SATELLITE_PATTERNS, DEM_DATASETS, METHOD_PATTERNS.

Core idea:
  - All entities come from the KnowledgeBase
  - Patterns are generated from entity acronym + full_name (+ overrides)
  - Each match is enriched with KB metadata (domain, used_for, related, etc.)
  - Context scoring filters out mere mentions vs. actual usage
"""

from __future__ import annotations

import re
import logging
from typing import Optional

from .knowledge_loader import KnowledgeBase, EntityRecord
from .normalizer import Normalizer

log = logging.getLogger(__name__)

# ─── metric patterns (value-capture aware) ───────────────────────────────────
# These keep their numeric-capture groups and are NOT replaced by KB.
METRIC_PATTERNS: dict[str, str] = {
    "OA":      r"(overall accuracy|global accuracy|\boa\b)\s*(?:=|:|of|values?)?\s*(\d+(?:\.\d+)?)\s*%?",
    "F1":      r"(f1[\-\s]?score|\bf1\b|f[\-\s]?measure)\s*(?:=|:|of|values?)?\s*(\d+(?:\.\d+)?)\s*%?",
    "IoU":     r"(\biou\b|intersection over union|jaccard)\s*(?:=|:|of|values?)?\s*(\d+(?:\.\d+)?)\s*%?",
    "Kappa":   r"(kappa|kp)\s*(?:coefficient)?\s*(?:=|:|of|values?)?\s*(\d+(?:\.\d+)?)\s*%?",
    "RMSE":    r"\brmse\b\s*(?:=|:|of)?\s*(\d+(?:\.\d+)?)",
    "MAE":     r"\bmae\b\s*(?:=|:|of)?\s*(\d+(?:\.\d+)?)",
    "R":       r"\br\s*=\s*(\d+(?:\.\d+)?)",
    "p_value": r"\bp\s*[<=>]\s*(\d+(?:\.\d+)?)",
    "Percent": r"(up to|around|approximately|about)?\s*(\d+(?:\.\d+)?)\s*%",
    "NSE":     r"\bnse\b\s*(?:=|:|of)?\s*(\d+(?:\.\d+)?)",
    "KGE":     r"\bkge\b\s*(?:=|:|of)?\s*(\d+(?:\.\d+)?)",
    "R2":      r"\br[\^²]?2\b\s*(?:=|:|of)?\s*(\d+(?:\.\d+)?)",
    "PBIAS":   r"\bpbias\b\s*(?:=|:|of)?\s*(\d+(?:\.\d+)?)\s*%?",
}

INVALID_METRIC_CONTEXT = [
    "aep", "annual exceedance probability",
    "world settlement footprint", "wsf", "table",
]

# ─── real-usage context filter ────────────────────────────────────────────────
_USAGE_POSITIVE = [
    "used", "applied", "we used", "this study uses", "data used",
    "method used", "we applied", "this paper uses", "using",
    "was applied", "is applied", "are used", "were used",
    "we employ", "employed", "we adopt", "adopted",
    "we utilize", "utilized",
]
_USAGE_NEGATIVE = [
    "previous studies", "other studies", "review", "for example",
    "such as", "e.g.", "et al", "proposed by", "introduced by",
    "according to", "in contrast to",
]


def _is_real_usage(ctx: str) -> bool:
    ctx_l = ctx.lower()
    pos = any(k in ctx_l for k in _USAGE_POSITIVE)
    neg = any(k in ctx_l for k in _USAGE_NEGATIVE)
    return pos and not neg


def _snippet(text: str, start: int, end: int, window: int = 200) -> str:
    return re.sub(r"\s+", " ", text[max(0, start - window): min(len(text), end + window)]).strip()


# ─── main extractor class ─────────────────────────────────────────────────────

class EntityExtractor:
    """
    Generic KB-driven entity extractor.

    All domain knowledge (satellite names, DEM names, method names, etc.)
    comes from the KnowledgeBase. No hardcoded patterns.

    Usage:
        ex = EntityExtractor(kb)
        satellites = ex.extract_satellites(text)
        dems       = ex.extract_dems(text)
        methods    = ex.extract_methods(text)
        metrics    = ex.extract_metrics(text)
    """

    def __init__(self, kb: KnowledgeBase):
        self.kb   = kb
        self.norm = Normalizer(kb)

    # ── generic KB-driven extraction ──────────────────────────────────────────

    def extract_from_knowledge_base(
        self,
        text: str,
        entities: list[EntityRecord],
        entity_type: str,
        source: str,
        strict: bool = True,
    ) -> list[dict]:
        """
        Extract entities from text using KB patterns.

        Args:
            text:        Input text to search.
            entities:    List of EntityRecord instances to match.
            entity_type: Label for the extracted entity type (e.g. "satellite").
            source:      Source section label (e.g. "data_sources+methods").
            strict:      If True, apply usage-context filter.

        Returns:
            List of entity dicts with name, type, source, confidence,
            evidence, and kb_metadata.
        """
        results: dict[str, dict] = {}

        for rec in entities:
            if not rec.pattern:
                continue
            try:
                for m in re.finditer(rec.pattern, text, re.IGNORECASE):
                    ctx = _snippet(text, m.start(), m.end(), window=200)

                    if strict and not _is_real_usage(ctx):
                        continue

                    key = rec.acronym.lower()
                    if key not in results:
                        results[key] = self._make_entity(rec, entity_type, source, ctx, m)
            except re.error as exc:
                log.debug("Pattern error for %s: %s", rec.acronym, exc)

        return list(results.values())

    def _make_entity(
        self,
        rec: EntityRecord,
        entity_type: str,
        source: str,
        ctx: str,
        match: re.Match,
    ) -> dict:
        return {
            "name":       rec.acronym,
            "type":       entity_type,
            "source":     source,
            "confidence": 0.90,
            "evidence":   ctx,
            "scores": {
                "pattern":   0.9,
                "context":   0.0,
                "embedding": 0.0,
                "llm":       0.0,
            },
            "final_score": 0.0,
            "accepted":    None,
            "role":        None,
            "kb_metadata": {
                "full_name":  rec.full_name,
                "type":       rec.type,
                "type_group": rec.type_group,
                "domain":     rec.domain,
                "definition": rec.definition,
                "used_for":   rec.used_for,
                "related":    rec.related,
                "contexts":   rec.contexts,
                "source_kb":  rec.source_kb,
            },
        }

    # ── typed extractors (mirror old pipeline API) ────────────────────────────

    def extract_satellites(self, text: str, strict: bool = True) -> list[dict]:
        """Detect satellite/sensor names from text."""
        return self.extract_from_knowledge_base(
            text,
            self.kb.get_satellites(),
            entity_type="satellite",
            source="data_sources+methods",
            strict=strict,
        )

    def extract_dems(self, text: str, strict: bool = True) -> list[dict]:
        """Detect DEM/terrain dataset names from text."""
        return self.extract_from_knowledge_base(
            text,
            self.kb.get_dems(),
            entity_type="dem",
            source="data_sources+methods",
            strict=strict,
        )

    def extract_methods(self, text: str, strict: bool = True) -> list[dict]:
        """
        Detect analysis methods, algorithms, and models from text.
        Combines KB entity extraction + ontology method records.
        """
        kb_methods = self.extract_from_knowledge_base(
            text,
            self.kb.get_methods(),
            entity_type="method",
            source="methods+results",
            strict=strict,
        )

        # also scan ontology method records
        onto_methods = self._extract_ontology_methods(text, strict)

        # merge, deduplicate by name
        merged: dict[str, dict] = {}
        for ent in kb_methods + onto_methods:
            key = ent["name"].lower()
            if key not in merged:
                merged[key] = ent
            else:
                # merge used_for / related from both hits
                existing = merged[key]
                for field in ("used_for", "related"):
                    existing_meta = existing.get("kb_metadata", {})
                    new_meta = ent.get("kb_metadata", {})
                    combined = list(set(
                        existing_meta.get(field, []) + new_meta.get(field, [])
                    ))
                    existing_meta[field] = combined

        return list(merged.values())

    def _extract_ontology_methods(self, text: str, strict: bool) -> list[dict]:
        results: dict[str, dict] = {}
        for mrec in self.kb.methods:
            if not mrec.pattern:
                continue
            try:
                for m in re.finditer(mrec.pattern, text, re.IGNORECASE):
                    ctx = _snippet(text, m.start(), m.end(), window=200)
                    if strict and not _is_real_usage(ctx):
                        continue
                    key = mrec.method_name.lower()
                    if key not in results:
                        results[key] = {
                            "name":       mrec.method_name,
                            "type":       "method",
                            "source":     "methods+results",
                            "confidence": 0.88,
                            "evidence":   ctx,
                            "scores": {
                                "pattern": 0.88,
                                "context": 0.0,
                                "embedding": 0.0,
                                "llm": 0.0,
                            },
                            "final_score": 0.0,
                            "accepted":    None,
                            "role":        None,
                            "kb_metadata": {
                                "full_name":  mrec.method_name.replace("_", " ").title(),
                                "type":       mrec.type,
                                "type_group": "method",
                                "domain":     mrec.domain,
                                "definition": mrec.description,
                                "used_for":   mrec.inputs,
                                "related":    mrec.related,
                                "inputs":     mrec.inputs,
                                "outputs":    mrec.outputs,
                                "source_kb":  mrec.source_kb,
                            },
                        }
            except re.error:
                pass
        return list(results.values())

    # ── metric extraction (value-capture aware, not purely KB-driven) ─────────

    def extract_metrics(self, text: str) -> list[dict]:
        """
        Extract performance metrics with their numeric values.
        Uses METRIC_PATTERNS (value-capture aware) combined with KB enrichment.
        """
        metrics: list[dict] = []
        seen: set[tuple] = set()

        for metric_type, pattern in METRIC_PATTERNS.items():
            for m in re.finditer(pattern, text, re.IGNORECASE):
                ctx = _snippet(text, m.start(), m.end(), window=120)
                if not self._is_valid_metric_context(ctx):
                    continue

                # extract numeric value
                raw = self._extract_metric_value(m, metric_type)
                if raw is None:
                    continue

                key = (metric_type, raw)
                if key in seen:
                    continue
                seen.add(key)

                value = self._normalize_metric_value(raw, metric_type)

                # enrich with KB if we have an entity for it
                kb_meta = {}
                rec = self.kb.resolve(metric_type)
                if rec:
                    kb_meta = {
                        "full_name":  rec.full_name,
                        "definition": rec.definition,
                        "domain":     rec.domain,
                    }

                metrics.append({
                    "type":  metric_type,
                    "value": value,
                    "evidence": {
                        "field":   metric_type,
                        "section": "results",
                        "snippet": ctx,
                        "source":  "regex",
                    },
                    "kb_metadata": kb_meta,
                })

        return metrics

    @staticmethod
    def _extract_metric_value(m: re.Match, metric_type: str) -> Optional[str]:
        groups = m.groups()
        if not groups:
            return None
        if metric_type in {"RMSE", "MAE", "R", "p_value", "NSE", "KGE", "R2", "PBIAS"}:
            return groups[-1] if groups[-1] else None
        # For OA, F1, IoU, Kappa, Percent: value is in group[1]
        if len(groups) >= 2:
            return groups[1] if groups[1] else None
        return groups[0] if groups[0] else None

    @staticmethod
    def _normalize_metric_value(value: str, metric_type: str) -> float:
        v = float(value)
        if metric_type in {"OA", "F1", "IoU", "Kappa", "R", "Percent"} and v > 1:
            return round(v / 100, 4)
        return round(v, 4)

    @staticmethod
    def _is_valid_metric_context(snippet: str) -> bool:
        s = snippet.lower()
        return not any(k in s for k in INVALID_METRIC_CONTEXT)

    # ── sensor type inference ─────────────────────────────────────────────────

    def infer_sensor_types(
        self,
        satellites: list[dict],
        dems: list[dict],
    ) -> list[str]:
        """
        Classify detected sensors into broad categories: SAR, Optical, LiDAR, DEM.
        Uses KB type information rather than hardcoded sets.
        """
        found: set[str] = set()

        SAR_SIGNALS     = {"sar", "radar", "synthetic aperture"}
        LIDAR_SIGNALS   = {"lidar", "laser", "icesat"}
        OPTICAL_SIGNALS = {"optical", "multispectral", "hyperspectral",
                           "vis", "nir", "swir"}

        for sat in satellites:
            name = sat["name"].lower()
            meta = sat.get("kb_metadata", {})
            defn = (meta.get("definition") or "").lower()
            fn   = (meta.get("full_name") or "").lower()

            combined = f"{name} {fn} {defn}"

            if any(sig in combined for sig in SAR_SIGNALS):
                found.add("SAR")
            elif any(sig in combined for sig in LIDAR_SIGNALS):
                found.add("LiDAR")
            elif any(sig in combined for sig in OPTICAL_SIGNALS):
                found.add("Optical")
            else:
                # fallback: check domain
                if meta.get("type") == "sensor":
                    found.add("Optical")

        if dems:
            found.add("DEM")

        return sorted(found)
