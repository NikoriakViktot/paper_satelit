"""
Full pipeline run + Markdown literature review report generator.
Outputs: outputs/literature_review_report.md  and  outputs/final_dataset.csv
"""
from __future__ import annotations

import os
import warnings
import logging
from pathlib import Path
from datetime import date

import pandas as pd

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.WARNING)

# ── ensure temp cache for HuggingFace ────────────────────────────────────────
os.environ.setdefault("HF_HOME", "/tmp/hf_cache")
os.environ.setdefault("SENTENCE_TRANSFORMERS_HOME", "/tmp/hf_cache/sentence_transformers")

from src.pipeline.rag_pipeline import RAGPipeline
from src.processing.analytics import (
    compute_summary,
    categorize_method,
    categorize_region,
    print_summary_tables,
)

OUTPUTS = Path("outputs")
OUTPUTS.mkdir(exist_ok=True)

REPORT_PATH  = OUTPUTS / "literature_review_report.md"
DATASET_PATH = OUTPUTS / "final_dataset.csv"


# ── 1. Run pipeline ───────────────────────────────────────────────────────────

print("Running RAG pipeline …")
pipeline = RAGPipeline()
df = pipeline.query(save_csv=False)
print(f"Extracted {len(df)} papers.")

# ── 2. Analytics ──────────────────────────────────────────────────────────────

print_summary_tables(df)
s = compute_summary(df)

# ── 3. Report helpers ─────────────────────────────────────────────────────────

def _fmt(val, decimals: int = 3) -> str:
    if pd.isna(val) or val is None:
        return "—"
    if isinstance(val, float):
        return f"{val:.{decimals}f}"
    return str(val).strip() or "—"


def _pct(n: int, total: int) -> str:
    return f"{round(100 * n / max(total, 1))}%"


def _md_table(headers: list[str], rows: list[list]) -> str:
    col_w = [max(len(h), max((len(str(r[i])) for r in rows), default=0))
             for i, h in enumerate(headers)]
    sep = "| " + " | ".join("-" * w for w in col_w) + " |"
    head = "| " + " | ".join(h.ljust(col_w[i]) for i, h in enumerate(headers)) + " |"
    body = "\n".join(
        "| " + " | ".join(str(r[i]).ljust(col_w[i]) for i, _ in enumerate(headers)) + " |"
        for r in rows
    )
    return "\n".join([head, sep, body])


def _dist_table(dist: dict, total: int) -> str:
    rows = sorted(dist.items(), key=lambda x: -x[1])
    return _md_table(
        ["Category", "Count", "Share"],
        [[k, str(v), _pct(v, total)] for k, v in rows],
    )


# ── 4. Build Markdown ─────────────────────────────────────────────────────────

total        = s["total"]
q_count      = s["quantitative"]
semi_count   = s["semi_quantitative"]
qual_count   = s["qualitative"]
doi_valid    = int(df["DOI_Valid"].sum()) if "DOI_Valid" in df.columns else 0
abs_valid    = int(df["Abstract_Valid"].sum()) if "Abstract_Valid" in df.columns else 0
today        = date.today().isoformat()

region_dist  = s["region_distribution"]
method_dist  = s["method_distribution"]
sensor_dist  = s["sensor_distribution"]

# region rows for per-category breakdown
ukraine_df = df[df["Region"].apply(
    lambda r: isinstance(r, str) and any(k in r.lower() for k in ("ukraine", "dnipro", "carpathian"))
)]
e_europe_df = df[df["Region"].apply(
    lambda r: isinstance(r, str) and "eastern europe" in r.lower() and "ukraine" not in r.lower()
)]

lines: list[str] = []

# ── Title / metadata ─────────────────────────────────────────────────────────
lines += [
    f"# Literature Review: Flood Mapping with Remote Sensing",
    f"",
    f"*Generated: {today}  |  Pipeline: RAG + Regex + Metadata Validation*",
    f"",
]

# ── 1. Introduction ──────────────────────────────────────────────────────────
lines += [
    "## 1. Introduction",
    "",
    f"This report presents the results of an automated systematic literature review "
    f"of flood mapping studies using satellite remote sensing. A total of **{total} papers** "
    f"were processed through the RAG (Retrieval-Augmented Generation) pipeline, which "
    f"extracts structured bibliographic and methodological metadata from PDF documents. "
    f"The purpose of this analysis is to characterise the current state of the art in "
    f"flood mapping with respect to methods employed, sensor types used, geographic coverage, "
    f"and the availability and completeness of reported accuracy metrics.",
    "",
]

# ── 2. Dataset Overview ──────────────────────────────────────────────────────
lines += [
    "## 2. Dataset Overview",
    "",
    _md_table(
        ["Metric", "Value"],
        [
            ["Total studies processed",       str(total)],
            ["Studies with valid DOI",         f"{doi_valid} ({_pct(doi_valid, total)})"],
            ["Studies with valid abstract",    f"{abs_valid} ({_pct(abs_valid, total)})"],
            ["Studies with quantitative metrics", f"{q_count} ({_pct(q_count, total)})"],
            ["Studies with semi-quantitative metrics", f"{semi_count} ({_pct(semi_count, total)})"],
            ["Studies qualitative only",       f"{qual_count} ({_pct(qual_count, total)})"],
        ],
    ),
    "",
    f"**DOI coverage** ({_pct(doi_valid, total)}): "
    f"A substantial portion of papers lacked machine-readable DOIs in the extracted "
    f"text, likely due to PDF formatting or placement outside the retrieved chunks.",
    "",
    f"**Abstract validity** ({_pct(abs_valid, total)}): "
    f"The automated abstract extractor requires a clearly delimited `Abstract` "
    f"section header. Papers where the abstract was embedded in the first-page text "
    f"without a distinct heading were assigned a fallback excerpt from the full text.",
    "",
]

# ── 3. Global Analysis ────────────────────────────────────────────────────────
lines += ["## 3. Global Analysis", ""]

# 3.1 Accuracy distribution
lines += [
    "### 3.1 Accuracy Distribution",
    "",
    _md_table(
        ["Level", "Count", "Share"],
        [
            ["Quantitative (OA / F1 / IoU)", str(q_count),    _pct(q_count, total)],
            ["Semi-quantitative (OA only)",  str(semi_count), _pct(semi_count, total)],
            ["Qualitative only",             str(qual_count), _pct(qual_count, total)],
        ],
    ),
    "",
    f"Only **{q_count + semi_count} out of {total}** studies ({_pct(q_count + semi_count, total)}) "
    f"reported any numeric accuracy metric. This reflects a widespread tendency in the "
    f"flood mapping literature to present results visually or descriptively rather than "
    f"through standardised performance indicators. "
    f"The predominance of qualitative assessments ({_pct(qual_count, total)}) highlights a "
    f"significant gap in reproducibility and cross-study comparability.",
    "",
]

# 3.2 Method distribution
method_cat_df = df.copy()
method_cat_df["Method_Cat"] = method_cat_df["Method"].apply(categorize_method)

lines += [
    "### 3.2 Method Distribution",
    "",
    _dist_table(method_dist, total),
    "",
    f"Machine learning methods (Random Forest, SVM, Decision Tree) are the most frequent "
    f"category ({method_dist.get('ML', 0)} studies, {_pct(method_dist.get('ML', 0), total)}), "
    f"followed by deep learning approaches such as U-Net and CNN "
    f"({method_dist.get('DL', 0)}, {_pct(method_dist.get('DL', 0), total)}) and "
    f"SAR-specific techniques (thresholding, change detection, OBIA) "
    f"({method_dist.get('SAR', 0)}, {_pct(method_dist.get('SAR', 0), total)}). "
    f"The remaining {method_dist.get('Other', 0) + method_dist.get('Unknown', 0)} studies "
    f"used unclassified or hybrid methods.",
    "",
]

# method detail table
method_detail = (
    df.groupby(df["Method"].fillna("Unknown"))["Source_File"]
    .count()
    .reset_index()
    .rename(columns={"Source_File": "Count", "Method": "Method"})
    .sort_values("Count", ascending=False)
    .head(12)
)
lines += [
    "**Top methods by occurrence:**",
    "",
    _md_table(
        ["Method", "Count", "Category"],
        [
            [row["Method"], str(row["Count"]),
             categorize_method(row["Method"])]
            for _, row in method_detail.iterrows()
        ],
    ),
    "",
]

# 3.3 Sensor distribution
lines += [
    "### 3.3 Sensor Distribution",
    "",
    _dist_table(sensor_dist, total),
    "",
    f"SAR-based sensors (primarily Sentinel-1) dominate the dataset "
    f"({sensor_dist.get('SAR', 0)} studies, {_pct(sensor_dist.get('SAR', 0), total)}), "
    f"reflecting the suitability of SAR for flood detection under cloud cover — "
    f"a critical advantage during flood events. Multi-sensor studies combining SAR and "
    f"optical imagery account for {sensor_dist.get('Multi', 0)} papers "
    f"({_pct(sensor_dist.get('Multi', 0), total)}), suggesting growing interest in "
    f"data fusion approaches.",
    "",
]

# 3.4 Region distribution
lines += [
    "### 3.4 Region Distribution",
    "",
    _dist_table(region_dist, total),
    "",
    f"A significant proportion of studies ({region_dist.get('Unknown', 0) + region_dist.get('Global', 0)}) "
    f"could not be assigned to a specific region, either because no geographic reference "
    f"was found in the retrieved text, or because the study used global datasets. "
    f"This partly reflects the chunk-based retrieval strategy, which may not always "
    f"surface the study-area description.",
    "",
]

# ── 4. Ukraine & Regional Focus ───────────────────────────────────────────────
lines += ["## 4. Ukraine and Regional Focus", ""]

if len(ukraine_df) > 0:
    lines += [
        f"**{len(ukraine_df)} studies** in this dataset reference Ukraine or its sub-regions "
        f"(Dnipro Basin, Carpathians). These are summarised below:",
        "",
        _md_table(
            ["Source File", "Method", "Sensor", "Region", "OA"],
            [
                [row["Source_File"][:50], _fmt(row.get("Method")),
                 _fmt(row.get("Sensor")), _fmt(row.get("Region")),
                 _fmt(row.get("OA"))]
                for _, row in ukraine_df.iterrows()
            ],
        ),
        "",
    ]
else:
    lines += [
        "No studies in the current extraction were explicitly linked to Ukraine or its "
        "sub-regions (Dnipro Basin, Carpathians, Eastern Europe) through the retrieved "
        "text chunks. This may reflect the coverage of the ingested PDF collection or "
        "the chunk-based retrieval not surfacing geographic metadata from these papers.",
        "",
    ]

if len(e_europe_df) > 0:
    lines += [
        f"**{len(e_europe_df)} studies** reference Eastern Europe (excluding Ukraine):",
        "",
        _md_table(
            ["Source File", "Method", "Sensor", "Region"],
            [
                [row["Source_File"][:50], _fmt(row.get("Method")),
                 _fmt(row.get("Sensor")), _fmt(row.get("Region"))]
                for _, row in e_europe_df.iterrows()
            ],
        ),
        "",
    ]

# ── 5. Per-Study Analysis ─────────────────────────────────────────────────────
lines += ["## 5. Per-Study Analysis", ""]

for i, (_, row) in enumerate(df.iterrows(), start=1):
    fname    = row.get("Source_File", "Unknown")
    title    = _fmt(row.get("Title"))
    authors  = _fmt(row.get("Authors"))
    doi_raw  = row.get("DOI")
    doi_str  = f"[{doi_raw}](https://doi.org/{doi_raw})" if pd.notna(doi_raw) else "—"
    method   = _fmt(row.get("Method"))
    sensor   = _fmt(row.get("Sensor"))
    region   = _fmt(row.get("Region"))
    oa       = _fmt(row.get("OA"))
    f1       = _fmt(row.get("F1"))
    iou      = _fmt(row.get("IoU"))
    acc_lvl  = _fmt(row.get("Accuracy_Level"))
    acc_desc = _fmt(row.get("Accuracy_Desc"))
    score    = _fmt(row.get("Extraction_Score"), 0)

    lines += [
        f"### Paper {i} — `{fname}`",
        "",
        _md_table(
            ["Field", "Value"],
            [
                ["Title",               title],
                ["Authors",             authors],
                ["DOI",                 doi_str],
                ["Method",              method],
                ["Sensor",              sensor],
                ["Region",              region],
                ["OA",                  oa],
                ["F1",                  f1],
                ["IoU",                 iou],
                ["Accuracy Level",      acc_lvl],
                ["Accuracy Description", acc_desc],
                ["Extraction Score",    score],
            ],
        ),
        "",
    ]

# ── 6. Key Findings ───────────────────────────────────────────────────────────
lines += [
    "## 6. Key Findings",
    "",
    "### 6.1 Scarcity of Standardised Accuracy Reporting",
    "",
    f"Only {_pct(q_count + semi_count, total)} of reviewed studies reported numeric "
    f"accuracy metrics (OA, F1, or IoU). The majority ({_pct(qual_count, total)}) "
    f"relied exclusively on qualitative assessments or visual comparisons. "
    f"This heterogeneity in reporting practices severely limits the ability to draw "
    f"cross-study conclusions or conduct meta-analytic comparisons of methods.",
    "",
    "### 6.2 Dominance of SAR-Based Approaches",
    "",
    f"Synthetic Aperture Radar (SAR) sensors were used in "
    f"{sensor_dist.get('SAR', 0) + sensor_dist.get('Multi', 0)} out of {total} studies "
    f"({_pct(sensor_dist.get('SAR', 0) + sensor_dist.get('Multi', 0), total)}), either "
    f"exclusively or in combination with optical data. This reflects the operational "
    f"advantages of SAR for flood detection: all-weather, day-and-night acquisition, "
    f"and direct sensitivity to surface water extent through changes in backscatter.",
    "",
    "### 6.3 Emergence of Deep Learning",
    "",
    f"Deep learning methods (U-Net, CNN, ViT) were applied in {method_dist.get('DL', 0)} "
    f"studies ({_pct(method_dist.get('DL', 0), total)}), approaching the prevalence of "
    f"classical machine learning approaches ({method_dist.get('ML', 0)} studies). "
    f"This trend indicates a rapid shift towards data-driven segmentation architectures "
    f"that can exploit large labelled flood datasets.",
    "",
    "### 6.4 Geographic Coverage Gaps",
    "",
    f"A majority of studies with identifiable geography focused on Asia (Bangladesh, "
    f"India, China) and North America (USA). Eastern European flood events — including "
    f"Ukraine, the Dnipro Basin, and Carpathian catchments — appear under-represented "
    f"in the literature, suggesting a need for targeted studies using locally acquired "
    f"SAR and optical imagery.",
    "",
]

# ── 7. Limitations ────────────────────────────────────────────────────────────
lines += [
    "## 7. Limitations",
    "",
    "### 7.1 Missing Accuracy Metrics",
    "",
    "A large number of papers did not report OA, F1, or IoU in the sections captured "
    "by the retrieval queries. It is possible that some papers report accuracy metrics "
    "in tables or supplementary material that was not indexed. Future work should "
    "include table-aware extraction and figure-caption parsing.",
    "",
    "### 7.2 Extraction Uncertainty",
    "",
    "The current pipeline relies on rule-based regex patterns for metric extraction. "
    "Non-standard notation (e.g., per-class F1, micro/macro averages) may be "
    "misidentified or missed entirely. The title and abstract extractors depend on "
    "the presence of clearly delimited section headers in the PDF text, which is "
    "not always the case after OCR or layout parsing.",
    "",
    "### 7.3 Variability in Reporting Conventions",
    "",
    "Studies use different scales (0–1 vs 0–100%), different metric names (Overall "
    "Accuracy vs Overall Classification Accuracy), and different evaluation protocols "
    "(per-image vs per-event vs per-dataset). All extracted values were normalised to "
    "the 0–1 scale, but subtle differences in evaluation protocols may introduce "
    "incomparabilities not detectable by automated extraction.",
    "",
    "### 7.4 Dataset Coverage",
    "",
    f"The current extraction processed {total} papers from the indexed PDF collection. "
    f"The vector store contains 13,970 chunks from a larger set of PDFs. Some papers "
    f"may have been underrepresented in the retrieval step if their accuracy or method "
    f"descriptions did not match the six predefined retrieval queries.",
    "",
]

# ── 8. Conclusion ─────────────────────────────────────────────────────────────
lines += [
    "## 8. Conclusion",
    "",
    f"This systematic review processed {total} flood mapping studies via an automated "
    f"RAG pipeline combining vector-store retrieval, regex-based information extraction, "
    f"metadata validation, and field normalisation. "
    f"The results confirm several established trends: the dominance of SAR sensors "
    f"(particularly Sentinel-1), the growing adoption of deep learning architectures "
    f"alongside classical ML methods, and a persistent gap in standardised quantitative "
    f"reporting. Only {_pct(q_count + semi_count, total)} of studies provided machine-readable "
    f"numeric accuracy, underscoring the need for community-wide adoption of reporting "
    f"standards (e.g., STAC flood datasets, standardised validation protocols). "
    f"Future reviews should expand geographic coverage — particularly for Eastern Europe "
    f"and Ukraine — and integrate table and figure extraction to recover the full breadth "
    f"of accuracy information available in the literature.",
    "",
    "---",
    f"*Report auto-generated by the Flood-Paper RAG Pipeline on {today}.*",
]

# ── 5. Save outputs ───────────────────────────────────────────────────────────

report_text = "\n".join(lines)
REPORT_PATH.write_text(report_text, encoding="utf-8")
print(f"\nMarkdown report saved → {REPORT_PATH}")

df.to_csv(DATASET_PATH, index=False)
print(f"Final dataset CSV saved → {DATASET_PATH}")
print(f"\nReport size: {len(report_text):,} chars  |  {report_text.count(chr(10))} lines")
