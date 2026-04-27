"""
Literature review analytics for flood mapping papers.
Produces summary tables and journal-ready scientific text.
"""
from __future__ import annotations

import pandas as pd


# ── Method categorization ─────────────────────────────────────────────────────

_DL_METHODS = {
    "u-net", "attention u-net", "cnn", "deeplab", "segnet", "resnet",
    "vit", "vision transformer", "lstm", "deep learning",
    "semantic segmentation", "fully convolutional",
}
_ML_METHODS = {
    "random forest", "svm", "machine learning", "logistic regression",
    "decision tree", "gradient boosting", "xgboost", "mlp",
    "naive bayes", "maximum likelihood", "k-nearest",
}
_SAR_METHODS = {
    "thresholding", "obia", "object-based", "change detection",
    "multi-temporal", "backscatter", "insar", "otsu",
}


def categorize_method(method: str | None) -> str:
    if not isinstance(method, str) or not method.strip():
        return "Unknown"
    m = method.strip().lower()
    if any(kw in m for kw in _DL_METHODS):
        return "DL"
    if any(kw in m for kw in _ML_METHODS):
        return "ML"
    if any(kw in m for kw in _SAR_METHODS):
        return "SAR"
    return "Other"


# ── Region categorization ─────────────────────────────────────────────────────

def categorize_region(region: str | None) -> str:
    if not isinstance(region, str) or not region.strip():
        return "Unspecified"
    r = region.strip().lower()
    if r in ("unknown", "nan", "none", "—", ""):
        return "Unspecified"
    if any(kw in r for kw in ("ukraine", "dnipro", "carpathian")):
        return "Ukraine"
    if any(kw in r for kw in ("eastern europe", "europe")):
        return "Eastern Europe"
    return "Global"


# ── Summary computation ───────────────────────────────────────────────────────

def compute_summary(df: pd.DataFrame) -> dict:
    total = len(df)

    level_counts  = (df["Accuracy_Level"].value_counts().to_dict()
                     if "Accuracy_Level" in df.columns else {})
    quantitative  = level_counts.get("Quantitative", 0)
    semi          = level_counts.get("Semi-quantitative", 0)
    qualitative   = level_counts.get("Qualitative", 0)

    region_dist = (
        df["Region"].apply(categorize_region).value_counts().to_dict()
        if "Region" in df.columns else {}
    )
    method_dist = (
        df["Method"].apply(categorize_method).value_counts().to_dict()
        if "Method" in df.columns else {}
    )
    sensor_dist = (
        df["Sensor"].value_counts().to_dict()
        if "Sensor" in df.columns else {}
    )

    return {
        "total":             total,
        "quantitative":      quantitative,
        "semi_quantitative": semi,
        "qualitative":       qualitative,
        "region_distribution": region_dist,
        "method_distribution": method_dist,
        "sensor_distribution": sensor_dist,
    }


# ── Display ───────────────────────────────────────────────────────────────────

def _pct(n: int, total: int) -> str:
    return f"{round(100 * n / total)}%" if total else "N/A"


def _table(title: str, data: dict, total: int) -> str:
    lines = [f"\n{'─' * 44}", f"  {title}", f"{'─' * 44}"]
    for key, count in sorted(data.items(), key=lambda x: -x[1]):
        lines.append(f"  {key:<28} {count:>4}  ({_pct(count, total)})")
    return "\n".join(lines)


def print_summary_tables(df: pd.DataFrame) -> None:
    s = compute_summary(df)
    total = s["total"]

    print("\n" + "=" * 44)
    print("  LITERATURE REVIEW SUMMARY")
    print("=" * 44)

    print(f"\n{'─' * 44}")
    print("  TABLE 1 — Study Overview")
    print(f"{'─' * 44}")
    print(f"  {'Total studies':<28} {total:>4}")
    print(f"  {'Quantitative (OA/F1/IoU)':<28} {s['quantitative']:>4}  ({_pct(s['quantitative'], total)})")
    print(f"  {'Semi-quantitative':<28} {s['semi_quantitative']:>4}  ({_pct(s['semi_quantitative'], total)})")
    print(f"  {'Qualitative':<28} {s['qualitative']:>4}  ({_pct(s['qualitative'], total)})")

    print(_table("TABLE 2 — Region Distribution",  s["region_distribution"],  total))
    print(_table("TABLE 3 — Method Distribution",  s["method_distribution"],  total))
    print(_table("TABLE 4 — Sensor Distribution",  s["sensor_distribution"],  total))

    print(f"\n{'─' * 44}")
    print("  SCIENTIFIC TEXT (journal-ready)")
    print(f"{'─' * 44}")
    print(_generate_scientific_text(s))
    print("=" * 44)


def _generate_scientific_text(s: dict) -> str:
    total = s["total"]
    q     = s["quantitative"]
    sq    = s["semi_quantitative"]
    ql    = s["qualitative"]
    rd    = s["region_distribution"]
    md    = s["method_distribution"]
    sd    = s["sensor_distribution"]

    top_region = max(rd, key=rd.get) if rd else "unspecified regions"
    top_sensor = max(sd, key=sd.get) if sd else "unspecified sensors"
    dl  = md.get("DL",  0)
    ml  = md.get("ML",  0)
    sar = md.get("SAR", 0)

    return (
        f"A systematic review of {total} flood mapping studies was conducted. "
        f"Of the reviewed papers, {q} ({_pct(q, total)}) provided quantitative accuracy "
        f"metrics (overall accuracy, F1-score, or IoU), {sq} ({_pct(sq, total)}) reported "
        f"semi-quantitative performance, and {ql} ({_pct(ql, total)}) offered qualitative "
        f"assessments only. "
        f"Geographically, {top_region} represented the most frequently studied area, "
        f"accounting for {rd.get(top_region, 0)} ({_pct(rd.get(top_region, 0), total)}) "
        f"of all studies. "
        f"Regarding methodology, deep learning approaches were applied in {dl} studies "
        f"({_pct(dl, total)}), classical machine learning in {ml} ({_pct(ml, total)}), "
        f"and SAR-based thresholding or OBIA methods in {sar} ({_pct(sar, total)}). "
        f"The dominant remote sensing sensor type was {top_sensor}, used in "
        f"{sd.get(top_sensor, 0)} ({_pct(sd.get(top_sensor, 0), total)}) of cases."
    )
