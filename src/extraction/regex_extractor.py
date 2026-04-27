"""
Rule-based information extractor for flood-mapping papers.

Parses accuracy metrics (OA, F1, IoU, Kappa), method, sensor, region,
and author directly from text using curated regex patterns.

Works entirely offline — no LLM required.
"""
from __future__ import annotations

import logging
import re
from statistics import mean

from src.extraction.base import BaseExtractor, ExtractionResult

logger = logging.getLogger(__name__)


# ── Accuracy patterns ────────────────────────────────────────────────────────
#
# Each tuple is (metric_name, compiled_pattern).
# Groups: (value) OR (lo, hi) for ranges.
# Patterns listed from most specific to least specific.

def _r(pattern: str) -> re.Pattern:
    return re.compile(pattern, re.IGNORECASE)


_NUM   = r"([\d]+\.?[\d]*)"          # integer or decimal
_PCT   = r"%?"                         # optional percent sign
_SEP   = r"\s*[=:>]?\s*"              # separator token
_RANGE = rf"{_NUM}\s*[-–]\s*{_NUM}"   # lo-hi range

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


# ── Keyword banks ─────────────────────────────────────────────────────────────

_DL_KW = [
    "u-net", "unet", "convolutional neural network", "cnn", "fcnn",
    "resnet", "efficientnet", "segnet", "basnet", "fsa-unet", "deeplab",
    "vision transformer", "vit", "lstm", "attention unet", "fully convolutional",
    "deep learning", "semantic segmentation",
]
_ML_KW = [
    "random forest", "support vector machine", "svm", "decision tree",
    "gradient boosting", "xgboost", "naive bayes", "logistic regression",
    "maximum likelihood", "mlc", "cart", "k-nearest", "k-means",
    "machine learning", "multilayer perceptron", "mlp",
]
_SAR_KW = [
    "thresholding", "otsu", "object-based", "obia", "insar",
    "backscatter", "change detection", "multi-temporal", "hysteresis",
    "histogram threshold", "rapid", "rst-flood", "gumbel",
    "repeat-pass sar", "depth estimation",
]

_SAR_SENSORS = [
    "sentinel-1", "terrasar-x", "uavsar", "cosmo-skymed", "alos",
    "radarsat", "ers", "envisat", "novasar",
]
_OPT_SENSORS = [
    "sentinel-2", "landsat", "modis", "viirs", "worldview",
    "pleiades", "spot", "planet", "aerial", "google earth",
]

_KNOWN_REGIONS = [
    "australia", "bangladesh", "china", "croatia", "england", "france",
    "germany", "india", "italy", "japan", "mexico", "moldova",
    "morocco", "myanmar", "netherlands", "pakistan", "poland",
    "romania", "slovenia", "spain", "ukraine", "united kingdom",
    "united states", "usa", "uk", "conus", "texas", "california",
    "florida", "vietnam", "global",
]

_REGION_PATTERN = _r(
    r"(?:study\s+area|region|location|site)[^.]{0,120}?(?:"
    + "|".join(re.escape(r) for r in _KNOWN_REGIONS)
    + r")"
)


# ── Author heuristics ─────────────────────────────────────────────────────────

# "Smith et al. (2024)" or "Smith, J., & Jones, K. (2023)"
_AUTHOR_PATTERN = _r(
    r"^([A-Z][a-zA-ZÀ-ɏ\-]+(?:\s+[A-Z][a-zA-ZÀ-ɏ\-]+)?"
    r"(?:\s+et\s+al\.?)?)\s*[,\(]?\s*(?:19|20)\d{2}"
)


# ── Main extractor class ───────────────────────────────────────────────────────

class RegexExtractor(BaseExtractor):
    """
    Offline rule-based extractor.
    Runs on concatenated chunk text.
    """

    def extract(
        self,
        chunks: list[dict],
        source_file: str,
    ) -> ExtractionResult:
        combined = "\n\n".join(c["text"] for c in chunks)
        result   = ExtractionResult(source_file=source_file)

        result.oa    = _extract_metric(combined, OA_PATTERNS)
        result.f1    = _extract_metric(combined, F1_PATTERNS)
        result.iou   = _extract_metric(combined, IOU_PATTERNS)
        result.kappa = _extract_metric(combined, KAPPA_PATTERNS)

        result.method = _classify_method(combined)
        result.sensor = _detect_sensors(combined)
        result.region = _detect_region(combined)
        result.author = _detect_author(combined, source_file)

        result.accuracy_desc = _build_desc(result)
        result.evidence = _collect_evidence(combined)
        result.confidence = _score_confidence(result)

        return result.finalize()


# ── Metric parsing helpers ────────────────────────────────────────────────────

def _parse_number(raw: str) -> float:
    num = float(raw.replace(",", "."))
    return num / 100.0 if num > 1.0 else num


def _extract_metric(
    text: str,
    patterns: list[re.Pattern],
) -> float | None:
    """Try every pattern; return the first successful parse (normalised 0–1)."""
    for pat in patterns:
        for m in pat.finditer(text):
            groups = [g for g in m.groups() if g is not None]
            if len(groups) == 2:          # range → mean
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


# ── Method classification ─────────────────────────────────────────────────────

def _kw_hits(text: str, keywords: list[str]) -> int:
    tl = text.lower()
    return sum(1 for kw in keywords if kw in tl)


def _classify_method(text: str) -> str:
    dl_score  = _kw_hits(text, _DL_KW)
    ml_score  = _kw_hits(text, _ML_KW)
    sar_score = _kw_hits(text, _SAR_KW)

    scores = {"DL": dl_score, "ML": ml_score, "SAR": sar_score}
    top    = max(scores, key=scores.get)   # type: ignore[arg-type]

    # Extract the specific method name from the text (first 3000 chars)
    snippet  = text[:3000].lower()
    for label, kw_list in [("DL", _DL_KW), ("ML", _ML_KW), ("SAR", _SAR_KW)]:
        if label == top:
            for kw in kw_list:
                if kw in snippet:
                    return kw.title()

    return top if scores[top] > 0 else "Unknown"


# ── Sensor detection ─────────────────────────────────────────────────────────

def _detect_sensors(text: str) -> str:
    tl = text.lower()
    found = [s.title() for s in _SAR_SENSORS + _OPT_SENSORS if s in tl]
    # deduplicate while preserving order
    seen: set[str] = set()
    unique = []
    for s in found:
        if s not in seen:
            seen.add(s)
            unique.append(s)
    return ", ".join(unique[:5]) if unique else "Unknown"


# ── Region detection ──────────────────────────────────────────────────────────

def _detect_region(text: str) -> str:
    tl = text.lower()
    found = [r.title() for r in _KNOWN_REGIONS if r in tl]
    # try the longer pattern first for a richer phrase
    m = _REGION_PATTERN.search(text)
    if m:
        phrase = m.group(0).strip()[:80]
        return phrase
    seen: set[str] = set()
    unique = [r for r in found if r not in seen and not seen.add(r)]   # type: ignore
    return ", ".join(unique[:4]) if unique else "Unknown"


# ── Author detection ──────────────────────────────────────────────────────────

def _detect_author(text: str, source_file: str) -> str:
    # search first 1500 chars (title / abstract section)
    head = text[:1500]
    for line in head.splitlines():
        m = _AUTHOR_PATTERN.match(line.strip())
        if m:
            return m.group(1).strip()

    # fallback: filename stem (e.g. "Smith_2024_flood.pdf" → "Smith 2024")
    stem = source_file.replace(".pdf", "").replace("_", " ").replace("-", " ")
    parts = stem.split()
    if parts:
        return " ".join(parts[:3])
    return "Unknown"


# ── Helpers ───────────────────────────────────────────────────────────────────

def _build_desc(r: ExtractionResult) -> str:
    parts = []
    if r.oa    is not None: parts.append(f"OA={r.oa:.3f}")
    if r.f1    is not None: parts.append(f"F1={r.f1:.3f}")
    if r.iou   is not None: parts.append(f"IoU={r.iou:.3f}")
    if r.kappa is not None: parts.append(f"Kappa={r.kappa:.3f}")
    return ", ".join(parts) if parts else "No numeric metrics extracted"


_EVIDENCE_PAT = _r(
    r"(?:accuracy|OA|F1|IoU|kappa|precision|recall)"
    r"[^.]{0,150}(?:[0-9]+\.?[0-9]*\s*%?)"
)


def _collect_evidence(text: str) -> list[str]:
    return [m.group(0).strip() for m in _EVIDENCE_PAT.finditer(text)][:6]


def _score_confidence(r: ExtractionResult) -> float:
    score = 0.0
    score += 0.25 * min(r._num_metrics(), 2)          # up to 0.5 for metrics
    score += 0.1  if r.method   != "Unknown" else 0.0
    score += 0.1  if r.sensor   != "Unknown" else 0.0
    score += 0.1  if r.region   != "Unknown" else 0.0
    score += 0.1  if r.author   != "Unknown" else 0.0
    score += 0.1  if r.evidence else 0.0
    return min(round(score, 2), 1.0)
