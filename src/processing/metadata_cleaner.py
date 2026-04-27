"""
Post-extraction metadata validation, cleaning, and normalization.

Validates Abstract/DOI, normalizes Method/Sensor/Region,
enforces region detection, scores rows, and filters low-quality records.
Never drops rows unless they fall below the quality threshold.
"""
from __future__ import annotations

import logging
import re

import pandas as pd

logger = logging.getLogger(__name__)


# ── Abstract validation ───────────────────────────────────────────────────────

_ABSTRACT_KEYWORDS = ["study", "analysis", "results", "method", "approach", "objective"]
_BAD_STARTS        = ["introduction", "methods", "materials", "results"]


def is_valid_abstract(text: str | None) -> bool:
    if not text or not isinstance(text, str):
        return False
    text_clean = text.strip()
    text_low   = text_clean.lower()
    if len(text_clean) < 300 or len(text_clean) > 1200:
        return False
    if any(text_low.startswith(x) for x in _BAD_STARTS):
        return False
    return sum(k in text_low for k in _ABSTRACT_KEYWORDS) >= 2


def get_valid_abstract(
    text: str | None,
    fallback_text: str | None,
) -> tuple[str | None, bool]:
    if is_valid_abstract(text):
        return text, True
    fb = fallback_text if isinstance(fallback_text, str) else None
    return (fb[:1200] if fb else None), False


# ── DOI validation ────────────────────────────────────────────────────────────

_DOI_PATTERN = re.compile(r"10\.\d{4,9}/[-._;()/:A-Z0-9]+", re.IGNORECASE)


def is_valid_doi(doi: str | None) -> bool:
    if not doi or not isinstance(doi, str):
        return False
    return bool(_DOI_PATTERN.match(doi))


# ── Region force-detection ────────────────────────────────────────────────────

# Ordered: first match wins — more specific entries go first
_REGION_RULES: list[tuple[str, str]] = [
    ("ukraine",        "Ukraine"),
    ("carpathian",     "Carpathians"),
    ("dnipro",         "Dnipro Basin"),
    ("eastern europe", "Eastern Europe"),
]


def force_region_detection(text: str | None) -> str | None:
    """Scan raw text for known geographic keywords; return canonical region name."""
    if not isinstance(text, str):
        return None
    text_low = text.lower()
    for keyword, region in _REGION_RULES:
        if keyword in text_low:
            return region
    return None


# ── Method normalization ──────────────────────────────────────────────────────

_METHOD_MAP: dict[str, str] = {
    "u-net":                        "U-Net",
    "unet":                         "U-Net",
    "attention unet":               "Attention U-Net",
    "attention u-net":              "Attention U-Net",
    "random forest":                "Random Forest",
    "cnn":                          "CNN",
    "convolutional neural network": "CNN",
    "fcnn":                         "CNN",
    "fully convolutional":          "CNN",
    "svm":                          "SVM",
    "support vector machine":       "SVM",
    "threshold":                    "Thresholding",
    "thresholding":                 "Thresholding",
    "otsu":                         "Thresholding",
    "histogram threshold":          "Thresholding",
    "deeplab":                      "DeepLab",
    "segnet":                       "SegNet",
    "resnet":                       "ResNet",
    "vision transformer":           "ViT",
    "vit":                          "ViT",
}


def normalize_method(value: str | None) -> str | None:
    if not isinstance(value, str) or not value.strip():
        return None
    return _METHOD_MAP.get(value.strip().lower(), value.strip())


# ── Sensor normalization ──────────────────────────────────────────────────────

_SAR_KEYWORDS = {
    "sentinel-1", "sar", "terrasar", "cosmo-skymed",
    "uavsar", "alos", "radarsat", "novasar", "ers", "envisat",
}
_OPT_KEYWORDS = {
    "sentinel-2", "landsat", "modis", "viirs", "worldview",
    "pleiades", "spot", "planet", "aerial", "google earth",
}


def normalize_sensor(value: str | None) -> str | None:
    if not isinstance(value, str) or not value.strip():
        return None
    v = value.lower()
    has_sar = any(kw in v for kw in _SAR_KEYWORDS)
    has_opt = any(kw in v for kw in _OPT_KEYWORDS)
    if has_sar and has_opt:
        return "Multi"
    if has_sar:
        return "SAR"
    if has_opt:
        return "Optical"
    return value.strip()


# ── Region cleaning ───────────────────────────────────────────────────────────

_REGION_NOISE = re.compile(
    r"\b(study\s+area|this\s+study|region)\b",
    re.IGNORECASE,
)


def clean_region(value: str | None) -> str | None:
    if not isinstance(value, str) or not value.strip():
        return None
    cleaned = _REGION_NOISE.sub("", value).strip().strip(",;. ")
    return cleaned if cleaned else None


# ── DataFrame-level operations ────────────────────────────────────────────────

def clean_metadata_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Validate Abstract and DOI.  Adds Abstract_Valid / DOI_Valid columns.
    Falls back abstract to Full_Text[:1200] when invalid.
    Never drops rows.
    """
    for col in ("Abstract", "DOI", "Full_Text"):
        if col not in df.columns:
            df[col] = None

    df["Abstract"], df["Abstract_Valid"] = zip(*df.apply(
        lambda row: get_valid_abstract(row["Abstract"], row.get("Full_Text")),
        axis=1,
    ))

    df["DOI_Valid"] = df["DOI"].apply(is_valid_doi)
    df["DOI"] = df.apply(
        lambda row: row["DOI"] if row["DOI_Valid"] else None,
        axis=1,
    )
    return df


def normalize_fields_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize Method, Sensor, and Region values.
    Fills null Regions via force_region_detection on Full_Text.
    """
    if "Method" in df.columns:
        df["Method"] = df["Method"].apply(normalize_method)

    if "Sensor" in df.columns:
        df["Sensor"] = df["Sensor"].apply(normalize_sensor)

    if "Region" in df.columns:
        df["Region"] = df["Region"].apply(clean_region)
        if "Full_Text" in df.columns:
            null_mask = df["Region"].isna() | (df["Region"].astype(str).str.strip() == "")
            df.loc[null_mask, "Region"] = (
                df.loc[null_mask, "Full_Text"].apply(force_region_detection)
            )

    return df


# ── Consistency filter + extraction scoring ───────────────────────────────────

_METRIC_COLS = ("OA", "F1", "IoU", "Kappa")


def _has_value(series_or_val) -> bool:
    """True if value is non-null and non-empty."""
    if isinstance(series_or_val, pd.Series):
        val = series_or_val.iloc[0] if not series_or_val.empty else None
    else:
        val = series_or_val
    return bool(pd.notna(val) and str(val).strip() not in ("", "None", "nan", "Unknown"))


def _score_row(row: pd.Series) -> int:
    score = 0
    if _has_value(row.get("Method")):   score += 1
    if _has_value(row.get("Sensor")):   score += 1
    if any(_has_value(row.get(m)) for m in _METRIC_COLS): score += 2
    if _has_value(row.get("Accuracy_Desc")): score += 1
    return score


def apply_consistency_filter(
    df: pd.DataFrame,
    min_score: int = 2,
) -> pd.DataFrame:
    """
    Add Extraction_Score column and drop rows that are neither
    (title AND method present) nor (score >= min_score).
    Uses OR logic to preserve dataset diversity.
    """
    df = df.copy()
    df["Extraction_Score"] = df.apply(_score_row, axis=1)

    def _col_has_value(series: pd.Series) -> pd.Series:
        return series.apply(lambda v: _has_value(v))

    has_title  = (_col_has_value(df["Title"])  if "Title"  in df.columns
                  else pd.Series(False, index=df.index))
    has_method = (_col_has_value(df["Method"]) if "Method" in df.columns
                  else pd.Series(False, index=df.index))

    keep    = (has_title & has_method) | (df["Extraction_Score"] >= min_score)
    dropped = int((~keep).sum())
    if dropped:
        logger.info(
            "Consistency filter removed %d low-quality rows (min_score=%d).",
            dropped, min_score,
        )

    return df[keep].reset_index(drop=True)


# ── Report ────────────────────────────────────────────────────────────────────

def print_report(df: pd.DataFrame) -> None:
    total           = len(df)
    valid_abstracts = int(df["Abstract_Valid"].sum()) if "Abstract_Valid" in df.columns else 0
    valid_dois      = int(df["DOI_Valid"].sum())      if "DOI_Valid"      in df.columns else 0

    print("=" * 50)
    print(f"  Total papers:       {total}")
    print(f"  Valid abstracts:    {valid_abstracts}")
    print(f"  Fallback abstracts: {total - valid_abstracts}")
    print(f"  Valid DOIs:         {valid_dois}")
    print("=" * 50)
