"""
Section-aware scientific data extractor for flood mapping literature.

Public surface
──────────────
    extract_scientific_data(sections)  → structured dict (no class needed)
    ScientificExtractor                → BaseExtractor adapter for the pipeline

Source discipline (what gets read from where)
─────────────────────────────────────────────
    Satellite_Names  ← abstract + methods      (never results)
    Sensor_Type      ← derived from Satellite_Names
    Study_Type       ← abstract  (+title for review/dataset detection)
    Country          ← abstract + introduction  (Ukraine-first special rule)
    River_Basin      ← abstract + methods
    Methods          ← methods ONLY             (strict — never abstract)
    OA / F1 / IoU    ← results ONLY             (strict — validated in text)
    Near_Real_Time   ← abstract + methods

No LLM.  Numbers must appear literally in the source section.
"""
from __future__ import annotations

import logging
import re
from statistics import mean

from src.extraction.base import BaseExtractor, ExtractionResult
from src.extraction.regex_extractor import (
    # Metric patterns (reuse — no duplication)
    OA_PATTERNS, F1_PATTERNS, IOU_PATTERNS, KAPPA_PATTERNS,
    # Geography vocabulary (reuse)
    _UKRAINE_KW, _RIVER_BASINS, _KNOWN_COUNTRIES,
    # Study-type keyword banks (reuse)
    _REVIEW_KW, _DATASET_KW, _HYDRO_FORE_KW, _HYDRAULIC_KW,
    _OPERATIONAL_KW, _DL_KW, _ML_KW,
    # Method rules (reuse)
    _METHODS_RULES,
)
from src.extraction.section_parser import parse_document_sections, describe_parsed

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# 1. SATELLITE DETECTION
#    Compiled regex per satellite, ordered specific → generic.
#    Patterns handle: "Sentinel-1A", "S1", "S1B", "sentinel 1", etc.
# ─────────────────────────────────────────────────────────────────────────────

_SAT_DEFS: list[tuple[str, str]] = [
    # ── SAR ──────────────────────────────────────────────────────────────────
    ("Sentinel-1",   r"sentinel[\s\-]?1[abc]?|\bs1[ab]?\b"),
    ("TerraSAR-X",   r"terrasar[\s\-]?x?|\btsx\b"),
    ("TanDEM-X",     r"tandem[\s\-]?x|\btdx\b"),
    ("COSMO-SkyMed", r"cosmo[\s\-]?skymed|\bcsk\b"),
    ("ALOS-2",       r"alos[\s\-]?2(?:\s*palsar)?|\bpalsar[\s\-]?2\b"),
    ("ALOS PALSAR",  r"\balos\s+palsar\b|\bpalsar\b"),
    ("RADARSAT-2",   r"radarsat[\s\-]?2"),
    ("RADARSAT-1",   r"radarsat[\s\-]?1"),
    ("UAVSAR",       r"\buavsar\b"),
    ("ICEYE",        r"\biceye\b"),
    ("Capella",      r"\bcapella\s+(?:space|sar)\b"),
    ("ERS-2",        r"\bers[\s\-]?2\b"),
    ("ERS-1",        r"\bers[\s\-]?1\b"),
    ("Envisat",      r"\benvisat\b"),
    ("NovaSAR",      r"\bnovasar\b"),
    # ── Optical ──────────────────────────────────────────────────────────────
    ("Sentinel-2",   r"sentinel[\s\-]?2[abc]?|\bs2[ab]?\b"),
    ("Landsat-9",    r"landsat[\s\-]?9"),
    ("Landsat-8",    r"landsat[\s\-]?8"),
    ("Landsat-7",    r"landsat[\s\-]?7"),
    ("Landsat-5",    r"landsat[\s\-]?5"),
    ("Landsat",      r"\blandsat\b"),          # generic — checked after versioned
    ("MODIS",        r"\bmodis\b"),
    ("VIIRS",        r"\bviirs\b"),
    ("WorldView-3",  r"worldview[\s\-]?3|\bwv[\s\-]?3\b"),
    ("WorldView-2",  r"worldview[\s\-]?2|\bwv[\s\-]?2\b"),
    ("Pleiades",     r"pl[eé]iades[\s\-]?\d?|\bphr\b"),
    ("PlanetScope",  r"\bplanetscope\b|\bplanet\s+labs?\b|\brapideye\b"),
    ("SPOT-7",       r"\bspot[\s\-]?7\b"),
    ("SPOT-6",       r"\bspot[\s\-]?6\b"),
    ("SPOT-5",       r"\bspot[\s\-]?5\b"),
]

_SAT_PATTERNS: list[tuple[str, re.Pattern]] = [
    (name, re.compile(pat, re.IGNORECASE))
    for name, pat in _SAT_DEFS
]

# Canonical → sensor type lookup
_SAR_SATS: frozenset[str] = frozenset({
    "Sentinel-1", "TerraSAR-X", "TanDEM-X", "COSMO-SkyMed",
    "ALOS-2", "ALOS PALSAR", "RADARSAT-2", "RADARSAT-1",
    "UAVSAR", "ICEYE", "Capella", "ERS-2", "ERS-1", "Envisat", "NovaSAR",
})
_OPT_SATS: frozenset[str] = frozenset({
    "Sentinel-2", "Landsat-9", "Landsat-8", "Landsat-7", "Landsat-5", "Landsat",
    "MODIS", "VIIRS", "WorldView-3", "WorldView-2",
    "Pleiades", "PlanetScope", "SPOT-7", "SPOT-6", "SPOT-5",
})

# Fallback: generic SAR / optical mentions when no specific satellite named
_SAR_GENERIC_RE = re.compile(
    r"\bsar\b|\bsynthetic[\s\-]aperture\s+radar\b", re.IGNORECASE
)
_OPT_GENERIC_RE = re.compile(
    r"\boptical\s+(?:satellite|imagery|sensor|data|image)\b", re.IGNORECASE
)


# ─────────────────────────────────────────────────────────────────────────────
# 2. NEAR-REAL-TIME
# ─────────────────────────────────────────────────────────────────────────────

_NRT_RE = re.compile(
    r"near[\s\-]?real[\s\-]?time"
    r"|\bnrt\b"
    r"|real[\s\-]time\s+flood"
    r"|rapid\s+(?:flood\s+)?mapping"
    r"|operational\s+flood\s+monitoring"
    r"|within\s+\d+\s*h(?:ours?)?",
    re.IGNORECASE,
)


# ─────────────────────────────────────────────────────────────────────────────
# 3. METRIC VALIDATION
#    Value must be found literally in the results text — prevents hallucination.
# ─────────────────────────────────────────────────────────────────────────────

def _literal_in_text(value: float, text: str) -> bool:
    """Return True if *value* appears in *text* in at least one standard form."""
    representations = [
        f"{value:.4f}", f"{value:.3f}", f"{value:.2f}",
        f"{value * 100:.2f}", f"{value * 100:.1f}", f"{value * 100:.0f}",
    ]
    for raw in representations:
        cleaned = raw.rstrip("0").rstrip(".")
        if not cleaned:
            continue
        if re.search(r"(?<!\d)" + re.escape(cleaned) + r"(?!\d)", text):
            return True
    return False


def _parse_float(raw: str) -> float:
    """Normalise 0–100 scale to 0–1."""
    n = float(raw.replace(",", "."))
    return round(n / 100.0 if n > 1.0 else n, 4)


def _scan_metric(text: str, patterns: list[re.Pattern]) -> float | None:
    """
    Extract the first numeric metric that:
    - matches a pattern in *patterns*
    - can be validated as literally present in *text*

    Returns float 0–1 or None.
    """
    for pat in patterns:
        for m in pat.finditer(text):
            groups = [g for g in m.groups() if g is not None]
            if len(groups) == 2:                    # range match
                try:
                    lo = _parse_float(groups[0])
                    hi = _parse_float(groups[1])
                    if 0.0 <= lo <= hi <= 1.0:
                        val = round(mean([lo, hi]), 4)
                        if _literal_in_text(lo, text) and _literal_in_text(hi, text):
                            return val
                except ValueError:
                    continue
            elif len(groups) == 1:
                try:
                    val = _parse_float(groups[0])
                    if 0.0 <= val <= 1.0 and _literal_in_text(val, text):
                        return val
                except ValueError:
                    continue
    return None


# ─────────────────────────────────────────────────────────────────────────────
# 4. SECTION-SPECIFIC FIELD EXTRACTORS
# ─────────────────────────────────────────────────────────────────────────────

def _extract_satellites(abstract: str, methods: str) -> list[str]:
    """
    Detect canonical satellite names from abstract + methods.

    Uses compiled regex per satellite (handles all naming variants).
    Deduplicates; if a versioned name (Landsat-8) is found, the generic
    base (Landsat) is suppressed.
    """
    search = f"{abstract} {methods}"
    found: list[str] = []
    seen:  set[str]  = set()

    for canonical, pat in _SAT_PATTERNS:
        if pat.search(search) and canonical not in seen:
            found.append(canonical)
            seen.add(canonical)

    # Suppress generic base when a versioned variant was already found
    #   Landsat-8 found → drop generic "Landsat"
    versioned_bases = {s.split("-")[0] for s in found if "-" in s and not s.startswith("ERS")}
    found = [s for s in found if s not in versioned_bases]

    return found


def _derive_sensor_type(satellite_names: list[str], search_text: str = "") -> str:
    """
    Classify sensor type from satellite names.
    Falls back to generic text scan if no named satellites found.
    """
    has_sar = any(s in _SAR_SATS for s in satellite_names)
    has_opt = any(s in _OPT_SATS for s in satellite_names)

    if not satellite_names:
        # Fallback: look for generic "SAR" / "optical" mentions
        has_sar = bool(_SAR_GENERIC_RE.search(search_text))
        has_opt = bool(_OPT_GENERIC_RE.search(search_text))

    if has_sar and has_opt:
        return "Multi-sensor"
    if has_sar:
        return "SAR"
    if has_opt:
        return "Optical"
    return ""


def _classify_study_type(abstract: str, title: str) -> str:
    """
    Classify the study type from abstract text (primary) and title.

    Hierarchy: Review → Dataset → Hydraulic → Forecasting → Operational → DL → ML
    → Satellite flood mapping (default)
    """
    combined = (abstract + " " + title).lower()

    def _hits(kw_list: list[str]) -> int:
        return sum(1 for kw in kw_list if kw in combined)

    if _hits(_REVIEW_KW) >= 2 or "review" in (title or "").lower():
        return "Review paper"
    if _hits(_DATASET_KW) >= 2:
        return "Dataset/benchmark paper"
    if _hits(_HYDRAULIC_KW) >= 1:
        return "Hydraulic modeling"
    if _hits(_HYDRO_FORE_KW) >= 1:
        return "Hydrological forecasting"
    if _hits(_OPERATIONAL_KW) >= 1:
        return "Operational mapping system"
    if _hits(_DL_KW) >= 2:
        return "ML/DL classification"
    if _hits(_ML_KW) >= 2:
        return "ML/DL classification"
    return "Satellite flood mapping"


def _extract_country(abstract: str, introduction: str) -> str:
    """
    Detect country/region.

    Priority rule (from spec):
        1. Ukraine-specific keywords → return exactly "Ukraine"
        2. Other countries in priority order
        3. River basin / region as fallback

    Ukraine-specific keywords checked first across abstract + introduction.
    """
    combined = (abstract + " " + introduction).lower()

    # Priority 1: Ukraine-specific (spec mandates exact string "Ukraine")
    if any(kw in combined for kw in _UKRAINE_KW):
        return "Ukraine"

    # Priority 2: Country name scan
    for country in _KNOWN_COUNTRIES:
        if country in combined:
            return country.title()

    return ""


def _extract_river_basin(abstract: str, methods: str) -> str:
    """Extract the first matching basin/river name from abstract + methods."""
    combined = (abstract + " " + methods).lower()
    for basin in _RIVER_BASINS:
        if basin.lower() in combined:
            return basin.title()
    return ""


def _extract_methods(methods_text: str) -> list[str]:
    """
    Extract method names FROM THE METHODS SECTION ONLY.

    Applies the canonical _METHODS_RULES (same set as RegexExtractor)
    but scoped strictly to the methods text.
    Never reads the abstract — prevents false positives like
    "we compare Random Forest with U-Net" in the abstract.
    """
    tl = methods_text.lower()
    found: list[str] = []
    seen:  set[str]  = set()

    for canonical, triggers in _METHODS_RULES:
        for kw in triggers:
            if re.search(kw, tl):
                if canonical not in seen:
                    found.append(canonical)
                    seen.add(canonical)
                break

    return found


def _detect_nrt(abstract: str, methods: str) -> bool | None:
    """Detect near-real-time capability from abstract and methods sections."""
    combined = abstract + " " + methods
    if _NRT_RE.search(combined):
        return True
    return None


# ─────────────────────────────────────────────────────────────────────────────
# 5. MAIN EXTRACTION FUNCTION
# ─────────────────────────────────────────────────────────────────────────────

_OUTPUT_KEYS = (
    "Title", "Satellite_Names", "Sensor_Type", "Study_Type",
    "Country", "River_Basin", "Methods",
    "OA", "F1", "IoU", "Kappa",
    "Near_Real_Time",
)


def extract_scientific_data(sections: dict) -> dict:
    """
    Extract structured scientific data from parsed document sections.

    Parameters
    ----------
    sections : dict
        Output of parse_document_sections() — keys: title, abstract,
        introduction, methods, results, discussion.  Values may be None.

    Returns
    -------
    dict with keys defined in _OUTPUT_KEYS.
    Satellite_Names and Methods are list[str].
    All other fields are str, float | None, or bool | None.
    """
    # Safely retrieve each section (None → empty string for text operations)
    title        = sections.get("title")        or ""
    abstract     = sections.get("abstract")     or ""
    introduction = sections.get("introduction") or ""
    methods      = sections.get("methods")      or ""
    results      = sections.get("results")      or ""

    # ── Satellite names (abstract + methods) ──────────────────────────────────
    satellite_names = _extract_satellites(abstract, methods)

    # ── Sensor type ──────────────────────────────────────────────────────────
    sensor_type = _derive_sensor_type(satellite_names, abstract + " " + methods)

    # ── Study type (abstract primarily, title for review detection) ───────────
    study_type = _classify_study_type(abstract, title)

    # ── Country (abstract + introduction, Ukraine-first) ──────────────────────
    country = _extract_country(abstract, introduction)

    # ── River basin (abstract + methods) ─────────────────────────────────────
    river_basin = _extract_river_basin(abstract, methods)

    # ── Methods — STRICT: only from methods section ──────────────────────────
    extracted_methods = _extract_methods(methods) if methods else []

    # ── Metrics — STRICT: only from results section ───────────────────────────
    # Not extracted for review / forecasting / hydraulic papers.
    _metric_types = {"Satellite flood mapping", "ML/DL classification",
                     "Operational mapping system"}
    if results and study_type in _metric_types:
        oa    = _scan_metric(results, OA_PATTERNS)
        f1    = _scan_metric(results, F1_PATTERNS)
        iou   = _scan_metric(results, IOU_PATTERNS)
        kappa = _scan_metric(results, KAPPA_PATTERNS)
    else:
        oa = f1 = iou = kappa = None

    # ── Near-real-time (abstract + methods) ───────────────────────────────────
    nrt = _detect_nrt(abstract, methods)

    return {
        "Title":           title or None,
        "Satellite_Names": satellite_names,
        "Sensor_Type":     sensor_type or None,
        "Study_Type":      study_type or None,
        "Country":         country or None,
        "River_Basin":     river_basin or None,
        "Methods":         extracted_methods,
        "OA":              oa,
        "F1":              f1,
        "IoU":             iou,
        "Kappa":           kappa,
        "Near_Real_Time":  nrt,
    }


# ─────────────────────────────────────────────────────────────────────────────
# 6. PIPELINE ADAPTER
#    Wraps parse_document_sections + extract_scientific_data into the
#    BaseExtractor interface used by RAGPipeline.
# ─────────────────────────────────────────────────────────────────────────────

class ScientificExtractor(BaseExtractor):
    """
    Full section-aware extraction pipeline (no LLM).

    Pipeline per document:
        1. Reconstruct full text from ordered chunks.
        2. parse_document_sections()  → structured sections dict.
        3. extract_scientific_data()  → validated structured fields.
        4. Convert to ExtractionResult for the downstream pipeline.
    """

    def extract(
        self,
        chunks: list[dict],
        source_file: str,
    ) -> ExtractionResult:
        # 1. Reconstruct full document text in page order
        ordered = sorted(
            chunks,
            key=lambda c: (c.get("page_start", 0), c.get("chunk_id", "")),
        )
        full_text = "\n\n".join(c["text"] for c in ordered)

        # 2. Parse into named sections
        sections = parse_document_sections(full_text)
        found, missing = describe_parsed(sections)
        logger.info(
            "[%s]  sections found=%s  missing=%s",
            source_file, found, missing,
        )

        # 3. Extract structured data
        data = extract_scientific_data(sections)

        # 4. Build ExtractionResult
        result = self._to_result(data, source_file, full_text, sections)

        logger.debug(
            "  → satellite=%s  country=%s  methods=%s  OA=%s  F1=%s  NRT=%s",
            data["Satellite_Names"], data["Country"], data["Methods"],
            data["OA"], data["F1"], data["Near_Real_Time"],
        )
        return result

    @staticmethod
    def _to_result(
        data: dict,
        source_file: str,
        full_text: str,
        sections: dict,
    ) -> ExtractionResult:
        result = ExtractionResult(
            source_file     = source_file,
            title           = data.get("Title")          or "",
            satellite_names = ", ".join(data.get("Satellite_Names") or []),
            sensor_type     = data.get("Sensor_Type")    or "",
            study_type      = data.get("Study_Type")     or "",
            country         = data.get("Country")        or "",
            river_basin     = data.get("River_Basin")    or "",
            methods         = ", ".join(data.get("Methods") or []),
            oa              = data.get("OA"),
            f1              = data.get("F1"),
            iou             = data.get("IoU"),
            kappa           = data.get("Kappa"),
            near_real_time  = data.get("Near_Real_Time"),
            ukraine_relevance = (data.get("Country") == "Ukraine"),
            full_text       = full_text,
            abstract        = sections.get("abstract") or "",
            sections_used   = [k for k, v in sections.items() if v and k != "title"],
            confidence      = _score_confidence(data, sections),
        )
        return result.finalize()


def _score_confidence(data: dict, sections: dict) -> float:
    """Estimate extraction confidence 0–1 based on field completeness."""
    score = 0.0
    score += 0.20 if data.get("Title")          else 0.0
    score += 0.15 if data.get("Satellite_Names") else 0.0
    score += 0.15 if sections.get("abstract")    else 0.0
    score += 0.15 if sections.get("methods")     else 0.0
    score += 0.10 if data.get("Country")         else 0.0
    score += 0.10 if data.get("Methods")         else 0.0
    score += 0.10 if sections.get("results")     else 0.0
    score += 0.05 if data.get("Near_Real_Time") is not None else 0.0
    return min(round(score, 2), 1.0)
