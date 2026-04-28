"""
Rule-based extractor for satellite-based flood mapping literature.

Extracts: bibliographic metadata, study type, satellite/sensor information,
study area, processing methods, optional accuracy metrics, and timeliness.
Works entirely offline — no LLM required.
"""
from __future__ import annotations

import logging
import re
from statistics import mean

from src.extraction.base import BaseExtractor, ExtractionResult

logger = logging.getLogger(__name__)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _r(pattern: str) -> re.Pattern:
    return re.compile(pattern, re.IGNORECASE)


_NUM   = r"([\d]+\.?[\d]*)"
_PCT   = r"%?"
_SEP   = r"\s*[=:>]?\s*"
_RANGE = rf"{_NUM}\s*[-–]\s*{_NUM}"


# ── Accuracy metric patterns ──────────────────────────────────────────────────

OA_PATTERNS: list[re.Pattern] = [
    _r(rf"[Oo]verall\s+[Aa]ccuracy{_SEP}{_RANGE}{_PCT}"),
    _r(rf"\bOA{_SEP}{_RANGE}{_PCT}"),
    _r(rf"[Oo]verall\s+[Aa]ccuracy{_SEP}{_NUM}{_PCT}"),
    _r(rf"\bOA\b{_SEP}{_NUM}{_PCT}"),
    _r(rf"[Aa]ccuracy\s+of\s+{_NUM}{_PCT}"),
    _r(rf"[Aa]ccuracy\s+rate{_SEP}{_NUM}{_PCT}"),
    _r(rf"[Aa]ccuracy{_SEP}{_NUM}{_PCT}"),
]

F1_PATTERNS: list[re.Pattern] = [
    _r(rf"[Ff]1[-\s]*[Ss]core{_SEP}{_RANGE}"),
    _r(rf"\bF[-\s]?1{_SEP}{_RANGE}"),
    _r(rf"[Ff]1[-\s]*[Ss]core{_SEP}{_NUM}"),
    _r(rf"\bF[-\s]?1\b{_SEP}{_NUM}"),
    _r(rf"[Ff]-[Mm]easure{_SEP}{_NUM}"),
    _r(rf"[Ff][-\s]score{_SEP}{_NUM}"),
]

IOU_PATTERNS: list[re.Pattern] = [
    _r(rf"\bIoU{_SEP}{_RANGE}"),
    _r(rf"\bIoU\b{_SEP}{_NUM}"),
    _r(rf"[Ii]ntersection\s+[Oo]ver\s+[Uu]nion{_SEP}{_NUM}"),
    _r(rf"\bmIoU\b{_SEP}{_NUM}"),
    _r(rf"[Jj]accard\s+[Ii]ndex{_SEP}{_NUM}"),
    _r(rf"[Jj]accard{_SEP}{_NUM}"),
]

KAPPA_PATTERNS: list[re.Pattern] = [
    _r(rf"[Kk]appa\s+[Cc]oefficient{_SEP}{_RANGE}"),
    _r(rf"[Cc]ohen.?s\s+[Kk]appa{_SEP}{_RANGE}"),
    _r(rf"\b[Kk]appa{_SEP}{_RANGE}"),
    _r(rf"[Kk]appa\s+[Cc]oefficient{_SEP}{_NUM}"),
    _r(rf"[Cc]ohen.?s\s+[Kk]appa{_SEP}{_NUM}"),
    _r(rf"\b[Kk]appa\b{_SEP}{_NUM}"),
]


# ── Study type keyword banks ──────────────────────────────────────────────────

_REVIEW_KW      = ["systematic review", "literature review", "state of the art",
                   "meta-analysis", "review of methods", "survey of", "review paper"]
_DATASET_KW     = ["benchmark dataset", "labeled dataset", "annotated dataset",
                   "training dataset", "ground truth dataset", "flood dataset",
                   "open dataset", "publicly available dataset"]
_HYDRO_FORE_KW  = ["hydrological model", "rainfall-runoff", "flood prediction",
                   "flood forecasting", "hec-hms", "swat model", "discharge forecast",
                   "streamflow"]
_HYDRAULIC_KW   = ["hydraulic model", "hydrodynamic model", "hec-ras", "lisflood",
                   "flo-2d", "mike flood", "mike 21", "2d flood simulation",
                   "inundation model", "2d hydraulic"]
_OPERATIONAL_KW = ["copernicus ems", "emergency management service",
                   "operational flood", "early warning system",
                   "flood monitoring system", "rapid mapping service",
                   "near-real-time system", "automated flood detection pipeline"]
_DL_KW          = ["u-net", "unet", "convolutional neural network", "cnn",
                   "deep learning", "semantic segmentation", "encoder-decoder",
                   "resnet", "deeplab", "segnet", "vision transformer", "lstm",
                   "fully convolutional"]
_ML_KW          = ["random forest", "support vector machine", "svm",
                   "machine learning classification", "decision tree",
                   "gradient boosting", "xgboost", "maximum likelihood classification",
                   "naive bayes", "logistic regression"]


# ── Satellite / sensor keyword lists ─────────────────────────────────────────

_SAR_SATELLITES = [
    "sentinel-1", "terrasar-x", "tandem-x", "uavsar", "cosmo-skymed",
    "alos-2", "alos palsar", "radarsat-2", "radarsat-1", "radarsat",
    "ers-1", "ers-2", "envisat asar", "novasar", "iceye", "capella",
]
_OPT_SATELLITES = [
    "sentinel-2", "landsat-8", "landsat-9", "landsat-7", "landsat",
    "modis", "viirs", "worldview-2", "worldview-3", "worldview",
    "pleiades", "spot-6", "spot-7", "spot", "planet labs", "planet",
    "planetscope", "rapideye", "aerial photography", "google earth engine",
]

_DATA_PRODUCTS = [
    "grdh", "grds", "grd", "slc", "msi", "oli", "tirs", "oli/tirs",
    "level-1c", "level-2a", "level-1", "level-2", "ard",
    "iw grd", "iw mode", "sm mode",
]

# ── Methods keyword map: (canonical_name, [trigger_keywords]) ─────────────────

_METHODS_RULES: list[tuple[str, list[str]]] = [
    ("Thresholding",        ["threshold", "thresholding", "otsu", "bimodal threshold",
                             "empirical threshold", "histogram threshold", "constant threshold"]),
    ("Change detection",    ["change detection", "multi-temporal", "bitemporal",
                             "bi-temporal", "pre-flood and post-flood", "pre/post-flood"]),
    ("NDWI/MNDWI",          ["ndwi", "mndwi", "normalized difference water index",
                             "modified ndwi", "awei", "automated water extraction"]),
    ("Random Forest",       ["random forest", "rf classifier"]),
    ("SVM",                 ["support vector machine", r"\bsvm\b"]),
    ("Maximum likelihood",  ["maximum likelihood", r"\bmlc\b"]),
    ("U-Net",               ["u-net", "unet", "attention u-net", "attention unet",
                             "fsa-unet", "res-unet"]),
    ("CNN",                 ["convolutional neural network", r"\bcnn\b", "fcnn",
                             "fully convolutional", "segnet", "deeplab", "basnet"]),
    ("LSTM",                [r"\blstm\b", "long short-term memory"]),
    ("Transformer",         ["vision transformer", r"\bvit\b", "swin transformer"]),
    ("OBIA",                ["obia", "object-based image analysis", "object-oriented"]),
    ("Hydrodynamic model",  ["hydrodynamic model", "hydraulic model", "hec-ras",
                             "lisflood", "flo-2d", "mike flood", "mike 21",
                             "2d flood simulation", "inundation model"]),
    ("Operational workflow", ["copernicus ems", "emergency mapping", "rapid mapping",
                              "operational workflow", "automated pipeline"]),
]


# ── Study area keyword lists ──────────────────────────────────────────────────

_KNOWN_COUNTRIES = [
    "australia", "bangladesh", "brazil", "china", "croatia", "england",
    "france", "germany", "india", "indonesia", "iran", "italy", "japan",
    "mexico", "moldova", "morocco", "myanmar", "netherlands", "nigeria",
    "pakistan", "poland", "romania", "serbia", "slovenia", "spain",
    "thailand", "ukraine", "united kingdom", "united states", "usa", "uk",
    "vietnam", "global",
]

# ── Ukraine-specific geographic keywords ─────────────────────────────────────
# Any of these triggers ukraine_relevance = True

_UKRAINE_KW = {
    "ukraine", "ukrainian",
    "dnipro", "dnieper",
    "kakhovka",                # Kakhovka reservoir / dam
    "prut",                    # Prut River (western Ukraine / Moldova)
    "dniester", "dnister",     # Dniester River
    "tisza", "tysa",           # Tisza / Tysa River (Zakarpattia)
    "carpathian", "carpathians",
    "bukovyna", "bukovina",
    "zakarpattia", "transcarpathia",
    "kherson", "kyiv", "odesa", "mykolaiv",
    "zaporizhzhia", "poltava", "vinnytsia",
    "chernihiv", "sumy", "zhytomyr",
    "desna",                   # Desna River
    "pivdennyi buh", "southern bug",
}

_RIVER_BASINS = [
    "amazon basin", "congo basin", "ganges", "brahmaputra", "mekong",
    "nile basin", "mississippi", "missouri river", "ohio river",
    "rhine basin", "danube basin", "po valley", "elbe", "oder",
    "loire", "thames", "indus basin", "yangtze", "yellow river",
    "niger basin", "zambezi", "murray-darling", "colorado river",
    "columbia river", "volga", "dnipro basin", "dnieper basin",
    "dniester basin", "prut basin", "tisza basin",
    "carpathian basin", "irrawaddy", "red river",
]

# ── Individual river names (finer than basin) ────────────────────────────────

_RIVER_NAMES = [
    # Ukrainian / Eastern European rivers
    "dnipro", "dnieper", "dniester", "dnister", "prut", "tisza", "tysa",
    "desna", "southern bug", "pivdennyi buh", "siversky donets",
    # Other European rivers
    "danube", "rhine", "elbe", "oder", "vistula", "thames", "seine",
    "po", "tiber", "rhone", "loire", "meuse", "main", "mosel",
    "drava", "sava", "tisa", "morava",
    # Asian rivers
    "ganges", "brahmaputra", "mekong", "yangtze", "yellow river",
    "irrawaddy", "indus", "godavari", "mahanadi", "brahmaputra",
    # Americas
    "mississippi", "missouri", "amazon", "colorado", "columbia",
    # Africa
    "nile", "niger", "zambezi", "congo",
    # Other
    "volga", "ob", "lena", "yenisei",
]

_REGION_CONTEXT_PAT = _r(
    r"(?:study\s+area|located\s+in|region\s+of|located\s+at|in\s+the)"
    r"[^.]{0,100}?(?:" + "|".join(re.escape(c) for c in _KNOWN_COUNTRIES) + r")"
)


# ── Timeliness patterns ───────────────────────────────────────────────────────

_NRT_PAT = _r(
    r"near[\s-]?real[\s-]?time|near\s+real\s+time|\bNRT\b"
    r"|rapid\s+(?:flood\s+)?mapping|real[\s-]time\s+flood"
    r"|operational\s+flood\s+monitoring"
)
_LATENCY_PAT = _r(
    r"(?:latency|delivered\s+within|available\s+within|within\s+less\s+than|"
    r"processing\s+time\s+of)\s*[:\-]?\s*(\d+(?:\.\d+)?)\s*(hours?|days?|h\b|d\b)"
)
_REVISIT_PAT = _r(
    r"(?:revisit\s+time|repeat\s+pass|repeat\s+cycle)\s*(?:of\s*)?(\d+(?:\.\d+)?)\s*(days?|hours?)"
)


# ── Bibliographic helpers ─────────────────────────────────────────────────────

_ABSTRACT_HEADER = re.compile(r"\bAbstract\b\s*[:\-—]?\s*", re.IGNORECASE)
_DOI_RE          = re.compile(r"10\.\d{4,9}/[-._;()/:A-Za-z0-9]+")
_YEAR_RE         = re.compile(r"\b(19[89]\d|20[0-3]\d)\b")
_AUTHOR_PATTERN  = _r(
    r"^([A-Z][a-zA-ZÀ-ɏ\-]+(?:\s+[A-Z][a-zA-ZÀ-ɏ\-]+)?"
    r"(?:\s+et\s+al\.?)?)\s*[,\(]?\s*(?:19|20)\d{2}"
)


def _extract_title(text: str, source_file: str) -> str:
    for line in text[:600].splitlines():
        line = line.strip()
        if 10 < len(line) < 200 and not line.lower().startswith(
            ("abstract", "introduction", "doi", "http", "www", "©", "keywords")
        ):
            return line
    return source_file.replace(".pdf", "").replace("_", " ")


def _extract_abstract(text: str) -> str:
    m = _ABSTRACT_HEADER.search(text)
    if not m:
        return ""
    after = text[m.end(): m.end() + 2000].strip()
    paras = re.split(r"\n\s*\n|\n(?=[1-9A-Z][.\s])", after)
    candidate = paras[0].strip() if paras else ""
    return candidate[:1200]


def _extract_doi(text: str) -> str:
    m = _DOI_RE.search(text[:3000])
    return m.group(0) if m else ""


def _extract_year(text: str, source_file: str) -> str:
    # prefer year from first 1000 chars (title/author area)
    m = _YEAR_RE.search(text[:1000])
    if m:
        return m.group(1)
    # fall back to filename
    m = _YEAR_RE.search(source_file)
    return m.group(1) if m else ""


def _extract_authors(text: str, source_file: str) -> str:
    head = text[:1500]
    for line in head.splitlines():
        m = _AUTHOR_PATTERN.match(line.strip())
        if m:
            return m.group(1).strip()
    stem = source_file.replace(".pdf", "").replace("_", " ").replace("-", " ")
    parts = stem.split()
    return " ".join(parts[:3]) if parts else "Unknown"


# ── Study type classification ─────────────────────────────────────────────────

def _classify_study_type(text: str) -> str:
    tl = text.lower()

    def _hits(kw_list: list[str]) -> int:
        return sum(1 for kw in kw_list if kw in tl)

    if _hits(_REVIEW_KW) >= 2:
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


# ── Satellite / sensor detection ──────────────────────────────────────────────

def _detect_satellites(text: str) -> tuple[str, str]:
    """Return (satellite_names_csv, sensor_type)."""
    tl = text.lower()

    sar_found = [s for s in _SAR_SATELLITES if s in tl]
    opt_found = [s for s in _OPT_SATELLITES if s in tl]

    # deduplicate preserving order
    def _dedup(lst: list[str]) -> list[str]:
        seen: set[str] = set()
        out = []
        for x in lst:
            if x not in seen:
                seen.add(x)
                out.append(x)
        return out

    sar_found = _dedup(sar_found)
    opt_found = _dedup(opt_found)

    all_found = [s.title() for s in sar_found + opt_found]
    names_csv = ", ".join(all_found[:6]) if all_found else ""

    if sar_found and opt_found:
        sensor_type = "Multi-sensor"
    elif sar_found:
        sensor_type = "SAR"
    elif opt_found:
        sensor_type = "Optical"
    else:
        sensor_type = ""

    return names_csv, sensor_type


def _detect_data_product(text: str) -> str:
    tl = text.lower()
    found = [p for p in _DATA_PRODUCTS if p.lower() in tl]
    # deduplicate
    seen: set[str] = set()
    unique = [p.upper() for p in found if p not in seen and not seen.add(p)]  # type: ignore
    return ", ".join(unique[:4]) if unique else ""


# ── Methods detection ─────────────────────────────────────────────────────────

def _detect_methods(text: str) -> str:
    tl = text.lower()
    found = []
    for canonical, triggers in _METHODS_RULES:
        for kw in triggers:
            if re.search(kw, tl):
                found.append(canonical)
                break
    # deduplicate preserving order
    seen: set[str] = set()
    unique = [m for m in found if m not in seen and not seen.add(m)]  # type: ignore
    return ", ".join(unique) if unique else ""


# ── Study area detection ──────────────────────────────────────────────────────

def _detect_country(text: str) -> str:
    tl = text.lower()
    found = [c.title() for c in _KNOWN_COUNTRIES if c in tl]
    # prefer a contextual match
    m = _REGION_CONTEXT_PAT.search(text)
    if m:
        phrase = m.group(0).strip()[:80]
        return phrase
    seen: set[str] = set()
    unique = [c for c in found if c not in seen and not seen.add(c)]  # type: ignore
    return ", ".join(unique[:4]) if unique else ""


def _detect_river_basin(text: str) -> str:
    tl = text.lower()
    found = [b.title() for b in _RIVER_BASINS if b in tl]
    seen: set[str] = set()
    unique = [b for b in found if b not in seen and not seen.add(b)]  # type: ignore
    return ", ".join(unique[:3]) if unique else ""


_CITY_EVENT_PAT = _r(
    r"(?:flooding\s+(?:in|of|event\s+in)|flood\s+event\s+(?:in|of)|"
    r"flood\s+of|case\s+study\s+(?:in|of))\s+([A-Z][a-zA-Z\s]{2,40})"
)


def _detect_city_event(text: str) -> str:
    m = _CITY_EVENT_PAT.search(text[:3000])
    return m.group(1).strip()[:60] if m else ""


# ── Ukraine relevance ─────────────────────────────────────────────────────────

def _detect_ukraine_relevance(text: str) -> bool:
    """True if any Ukraine-specific geographic keyword appears in *text*."""
    tl = text.lower()
    return any(kw in tl for kw in _UKRAINE_KW)


# ── River name detection ──────────────────────────────────────────────────────

# Pre-compiled patterns: short names (≤3 chars) need \b word boundaries to avoid
# matching inside other words ("ob" inside "global", "po" inside "poland", etc.)
_RIVER_PATTERNS: list[tuple[str, re.Pattern]] = [
    (name, re.compile(
        r"\b" + re.escape(name) + r"\b" if len(name) <= 3
        else re.escape(name),
        re.IGNORECASE,
    ))
    for name in _RIVER_NAMES
]


def _detect_river_name(text: str) -> str:
    """Extract specific river names with word-boundary protection for short names."""
    found = []
    for canonical, pat in _RIVER_PATTERNS:
        if pat.search(text):
            found.append(canonical.title())
    seen: set[str] = set()
    unique = [r for r in found if r not in seen and not seen.add(r)]  # type: ignore
    return ", ".join(unique[:5]) if unique else ""


# ── Timeliness detection ──────────────────────────────────────────────────────

def _detect_timeliness(text: str) -> tuple[str, str, bool | None]:
    """Return (latency, revisit_time, near_real_time)."""
    nrt: bool | None = None
    if _NRT_PAT.search(text):
        nrt = True

    latency = ""
    m = _LATENCY_PAT.search(text)
    if m:
        latency = f"{m.group(1)} {m.group(2)}"

    revisit = ""
    m = _REVISIT_PAT.search(text)
    if m:
        revisit = f"{m.group(1)} {m.group(2)}"

    return latency, revisit, nrt


# ── Confidence scoring ────────────────────────────────────────────────────────

def _score_confidence(r: ExtractionResult) -> float:
    score = 0.0
    score += 0.20 if r.title     else 0.0
    score += 0.15 if r.satellite_names else 0.0
    score += 0.15 if r.country        else 0.0
    score += 0.15 if r.methods        else 0.0
    score += 0.10 if r.study_type     else 0.0
    score += 0.10 if r._num_metrics() > 0 else 0.0
    score += 0.10 if r.authors        else 0.0
    score += 0.05 if r.river_basin    else 0.0
    return min(round(score, 2), 1.0)


# ── Evidence collection ───────────────────────────────────────────────────────

_EVIDENCE_PAT = _r(
    r"(?:accuracy|OA|F1|IoU|kappa|precision|recall|near-real-time|"
    r"Sentinel-1|Sentinel-2|SAR|flood mapping)"
    r"[^.]{0,150}(?:[0-9]+\.?[0-9]*\s*%?)"
)


def _collect_evidence(text: str) -> list[str]:
    return [m.group(0).strip() for m in _EVIDENCE_PAT.finditer(text)][:6]


# ── Metric parsing ────────────────────────────────────────────────────────────

def _parse_number(raw: str) -> float:
    num = float(raw.replace(",", "."))
    return num / 100.0 if num > 1.0 else num


def _extract_metric(text: str, patterns: list[re.Pattern]) -> float | None:
    for pat in patterns:
        for m in pat.finditer(text):
            groups = [g for g in m.groups() if g is not None]
            if len(groups) == 2:
                try:
                    lo, hi = _parse_number(groups[0]), _parse_number(groups[1])
                    if 0.0 <= lo <= hi <= 1.0:
                        return round(mean([lo, hi]), 4)
                except ValueError:
                    continue
            elif len(groups) == 1:
                try:
                    val = _parse_number(groups[0])
                    if 0.0 <= val <= 1.0:
                        return round(val, 4)
                except ValueError:
                    continue
    return None


# ── Main extractor class ──────────────────────────────────────────────────────

class RegexExtractor(BaseExtractor):
    """
    Offline rule-based extractor for satellite flood mapping literature.
    """

    def extract(
        self,
        chunks: list[dict],
        source_file: str,
    ) -> ExtractionResult:
        combined = "\n\n".join(c["text"] for c in chunks)
        result   = ExtractionResult(source_file=source_file)

        # Bibliographic
        result.title    = _extract_title(combined, source_file)
        result.abstract = _extract_abstract(combined)
        result.doi      = _extract_doi(combined)
        result.year     = _extract_year(combined, source_file)
        result.authors  = _extract_authors(combined, source_file)
        result.full_text = combined

        # Study type
        result.study_type = _classify_study_type(combined)

        # Satellite / sensor
        result.satellite_names, result.sensor_type = _detect_satellites(combined)
        result.data_product = _detect_data_product(combined)

        # Study area
        result.country            = _detect_country(combined)
        result.river_basin        = _detect_river_basin(combined)
        result.river_name         = _detect_river_name(combined)
        result.city_event         = _detect_city_event(combined)
        result.ukraine_relevance  = _detect_ukraine_relevance(combined)

        # Methods
        result.methods = _detect_methods(combined)

        # Metrics — only extract if paper likely reports them
        if result.study_type in ("ML/DL classification", "Satellite flood mapping"):
            result.oa    = _extract_metric(combined, OA_PATTERNS)
            result.f1    = _extract_metric(combined, F1_PATTERNS)
            result.iou   = _extract_metric(combined, IOU_PATTERNS)
            result.kappa = _extract_metric(combined, KAPPA_PATTERNS)

        # Timeliness
        result.latency, result.revisit_time, result.near_real_time = _detect_timeliness(combined)

        result.evidence   = _collect_evidence(combined)
        result.confidence = _score_confidence(result)

        return result.finalize()
