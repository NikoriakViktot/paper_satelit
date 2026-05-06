
"""
pipeline.py  —  GeoHydroAI Extraction Pipeline
================================================
Refactored version of tei_to_sections.py.

Architecture layers:
  1. Knowledge Layer   → KnowledgeBase (JSON-driven, no hardcoded patterns)
  2. Normalization     → Normalizer (alias resolution)
  3. Extraction        → EntityExtractor (KB-driven pattern matching)
  4. Method Matching   → MethodMatcher (ontology-driven)
  5. Geo / NER         → spaCy + Geonames (unchanged)
  6. Classification    → embedding cosine similarity (unchanged)
  7. Validation        → OllamaJudge LLM (unchanged)
  8. Output            → structured JSON (identical schema to v1)

What changed vs tei_to_sections.py:
  - SATELLITE_PATTERNS, DEM_DATASETS, METHOD_PATTERNS → KB-driven
  - All entities enriched with KB metadata (full_name, domain, used_for, etc.)
  - METRIC_PATTERNS kept (value-capture groups need explicit regex)
  - COUNTRY_PATTERNS kept (geo, not ontology)
  - All geo / embedding / LLM logic is preserved exactly
"""

from __future__ import annotations

import sys
import subprocess

# When run directly (python src/ingestion/pipeline.py), ensure the project root
# is on sys.path so that absolute imports of src.* packages work.
if __name__ == "__main__":
    _root = str(__import__("pathlib").Path(__file__).resolve().parents[2])
    if _root not in sys.path:
        sys.path.insert(0, _root)
import re
import json
import hashlib
import urllib.parse
import logging
import os
import traceback
from pathlib import Path
from typing import Optional

import numpy as np
import requests
import spacy
import time
from lxml import etree
from sklearn.metrics.pairwise import cosine_similarity

# ── knowledge layer ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    # support direct execution: python src/ingestion/pipeline.py
    import importlib, sys as _sys
    _pkg = importlib.import_module("src.ingestion.knowledge")
    KnowledgeBase      = _pkg.KnowledgeBase
    load_knowledge_base = _pkg.load_knowledge_base
    Normalizer         = _pkg.Normalizer
    EntityExtractor    = _pkg.EntityExtractor
    MethodMatcher      = _pkg.MethodMatcher
else:
    from .knowledge import (
        KnowledgeBase,
        load_knowledge_base,
        Normalizer,
        EntityExtractor,
        MethodMatcher,
    )

log = logging.getLogger(__name__)

# ── paths ─────────────────────────────────────────────────────────────────────
os.environ["HF_HOME"]            = "/home/viktornikoriak/paper_satelit/.hf_cache"
os.environ["TRANSFORMERS_CACHE"] = "/home/viktornikoriak/paper_satelit/.hf_cache"

BASE    = Path("/home/viktornikoriak/paper_satelit/data/literature")
XML_DIR = BASE / "grobid_xml"
OUT_DIR = BASE / "paper_json"
OUT_DIR.mkdir(exist_ok=True)

NS = {"tei": "http://www.tei-c.org/ns/1.0"}
GEO_CACHE: dict[str, Optional[dict]] = {}
LAST_CALL = 0.0

# ─────────────────────────────────────────────────────────────────────────────
# KNOWLEDGE BASE  (loaded once at module import)
# ─────────────────────────────────────────────────────────────────────────────
_KB: Optional[KnowledgeBase] = None


def _get_kb() -> KnowledgeBase:
    global _KB
    if _KB is None:
        _KB = load_knowledge_base()
    return _KB


# ─────────────────────────────────────────────────────────────────────────────
# GEO PATTERNS  (kept as-is — geographic data, not ontology)
# ─────────────────────────────────────────────────────────────────────────────

COUNTRY_PATTERNS: dict[str, str] = {
    "Ukraine":        r"\bukraine\b|\bukrainian\b|\bkyiv\b|\bkherson\b|\bkakhovka\b|\bzakarpattia\b|\bcarpathians?\b",
    "Poland":         r"\bpoland\b|\bpolish\b",
    "Germany":        r"\bgermany\b|\bgerman\b",
    "France":         r"\bfrance\b|\bfrench\b",
    "Italy":          r"\bitaly\b|\bitalian\b",
    "Spain":          r"\bspain\b|\bspanish\b",
    "Portugal":       r"\bportugal\b|\bportuguese\b",
    "Netherlands":    r"\bnetherlands\b|\bdutch\b",
    "Belgium":        r"\bbelgium\b",
    "Austria":        r"\baustria\b",
    "Switzerland":    r"\bswitzerland\b",
    "Slovenia":       r"\bslovenia\b",
    "Croatia":        r"\bcroatia\b",
    "Hungary":        r"\bhungary\b",
    "Czech Republic": r"\bczech\b",
    "Slovakia":       r"\bslovakia\b",
    "Romania":        r"\bromania\b",
    "Bulgaria":       r"\bbulgaria\b",
    "Greece":         r"\bgreece\b",
    "Turkey":         r"\bturkey\b",
    "UK":             r"\buk\b|united kingdom|england|scotland|wales",
    "Ireland":        r"\bireland\b",
    "USA":            r"\busa\b|united states|louisiana|california|texas|florida",
    "Canada":         r"\bcanada\b",
    "Mexico":         r"\bmexico\b",
    "Brazil":         r"\bbrazil\b",
    "Argentina":      r"\bargentina\b",
    "Chile":          r"\bchile\b",
    "Peru":           r"\bperu\b",
    "India":          r"\bindia\b|\bkerala\b",
    "China":          r"\bchina\b",
    "Japan":          r"\bjapan\b",
    "South Korea":    r"\bkorea\b",
    "Vietnam":        r"\bvietnam\b",
    "Thailand":       r"\bthailand\b",
    "Indonesia":      r"\bindonesia\b",
    "Malaysia":       r"\bmalaysia\b",
    "Pakistan":       r"\bpakistan\b",
    "Bangladesh":     r"\bbangladesh\b",
    "Uzbekistan":     r"\buzbekistan\b",
    "Kazakhstan":     r"\bkazakhstan\b",
    "Iran":           r"\biran\b",
    "Iraq":           r"\biraq\b",
    "Australia":      r"\baustralia\b|new south wales|queensland",
    "New Zealand":    r"\bnew zealand\b",
    "South Africa":   r"\bsouth africa\b",
    "Egypt":          r"\begypt\b",
    "Morocco":        r"\bmorocco\b",
    "Madagascar":     r"\bmadagascar\b",
    "Nigeria":        r"\bnigeria\b",
    "Philippines":    r"\bphilippines\b|\bluzon\b",
}

RIVER_TO_COUNTRY: dict[str, str] = {
    "Dnipro":        "Ukraine", "Dnieper": "Ukraine",
    "Prut":          "Ukraine/Romania", "Dniester": "Ukraine/Moldova",
    "Tisza":         "Ukraine/Hungary", "Southern Bug": "Ukraine",
    "Desna":         "Ukraine", "Moshchunka": "Ukraine",
    "Danube":        "Multiple", "Rhine": "Germany/France/Netherlands",
    "Elbe":          "Germany/Czech Republic", "Seine": "France",
    "Loire":         "France", "Thames": "UK",
    "Po":            "Italy", "Ebro": "Spain",
    "Tagus":         "Spain/Portugal", "Duero": "Spain/Portugal",
    "Carrion":       "Spain", "Krka": "Slovenia",
    "Sava":          "Slovenia/Croatia", "Drava": "Austria/Slovenia/Croatia",
    "Vistula":       "Poland", "Oder": "Germany/Poland",
    "Ganges":        "India/Bangladesh", "Indus": "India/Pakistan",
    "Yangtze":       "China", "Yellow River": "China",
    "Mekong":        "Multiple", "Irrawaddy": "Myanmar",
    "Mississippi":   "USA", "Colorado": "USA",
    "Amazon":        "Brazil/Peru", "Orinoco": "Venezuela",
    "Parana":        "Argentina/Brazil",
    "Nile":          "Multiple", "Congo": "DR Congo",
    "Niger":         "Multiple", "Zambezi": "Multiple",
    "Darling":       "Australia", "Murray": "Australia",
    "Cagayan":       "Philippines", "Cagayan River": "Philippines",
}

RIVER_PATTERNS: list[str] = [
    r"\b([A-Z][a-zA-Z\-]+(?:\s+[A-Z][a-zA-Z\-]+){0,3})\s+River basin\b",
    r"\b([A-Z][a-zA-Z\-]+(?:\s+[A-Z][a-zA-Z\-]+){0,3})\s+River\b",
    r"\b(Dnipro|Dnieper|Prut|Dniester|Danube|Tisza|Moshchunka|Krka|Sava|Darling|Carrion|Duero|Indus|Ganges)\b",
]

REGION_PATTERNS: dict[str, str] = {
    "Luzon":                r"\bluzon\b",
    "Cagayan Valley":       r"\bcagayan valley\b",
    "Northern Philippines": r"\bnorthern philippines\b",
    "Steppe zone":          r"\bsteppe zone\b",
    "Forest-steppe zone":   r"\bforest[\-\s]?steppe\b",
    "Ukrainian Carpathians":r"\bukrainian carpathians\b|\bcarpathians\b",
    "Crimean Mountains":    r"\bcrimean mountains\b",
    "Kyiv Oblast":          r"\bkyiv oblast\b",
    "Bucha district":       r"\bbucha district\b",
    "Zakarpattia":          r"\bzakarpattia\b|\btranscarpathia\b",
    "Kherson":              r"\bkherson\b",
    "Krka floodplain":      r"\bkrka river floodplain\b|\bkrka floodplain\b",
    "Lower Krka":           r"\blower krka\b",
    "Krakovo forest":       r"\bkrakovo forest\b",
    "Kerala":               r"\bkerala\b",
    "Fishlake":             r"\bfishlake\b",
    "Pontypridd":           r"\bpontypridd\b",
    "Rhondda Cynon Taf":    r"\brhondda cynon taf\b",
}

INVALID_LOCATIONS: set[str] = {
    "earth", "world", "globe", "surface",
    "region", "area", "study", "model",
    "figure", "table", "section", "introduction",
    "results", "discussion", "dsm", "worlddem",
    "sentinel", "sentinel-1", "sentinel-2",
    "sar", "gee", "google earth engine",
    "north america", "western europe",
    "the middle east", "middle east",
}

GENERIC_REGION_NOISE: set[str] = {
    "specific region", "study region", "target region",
    "selected region", "this region", "the region",
    "region", "area", "geographical region",
    "geographical regions", "selected locations",
    "selected regions", "region a", "region b", "region c",
}

# ─────────────────────────────────────────────────────────────────────────────
# CLASSIFICATION PROTOTYPES
# ─────────────────────────────────────────────────────────────────────────────

STUDY_TYPE_PROTOTYPES: dict[str, list[str]] = {
    "case_study": [
        "study conducted in a specific river basin",
        "analysis of a flood event in a region",
        "case study of a watershed",
    ],
    "multi_site": [
        "evaluation across multiple locations",
        "analysis on many flood events",
        "tested on various regions",
    ],
    "global_algorithmic": [
        "global flood mapping algorithm",
        "method applied worldwide",
        "large scale automatic system",
    ],
    "regional": [
        "analysis across a country",
        "study using multiple stations in one country",
        "regional climate analysis",
    ],
    "review": [
        "this paper reviews existing methods",
        "we survey the literature",
        "systematic review of flood mapping techniques",
        "comparison of different approaches",
        "literature review on remote sensing",
    ],
}

TASK_PROTOTYPES: dict[str, list[str]] = {
    "flood_mapping_satellite": [
        "flood mapping using Sentinel-1 SAR imagery",
        "water extent extraction using satellite data",
        "flood detection using remote sensing",
        "inundation mapping using SAR data",
        "satellite based flood monitoring",
        "flood extent mapping from SAR or optical satellite imagery",
    ],
    "flood_modeling_hydraulic": [
        "flood simulation using HEC-RAS",
        "2D hydraulic flood modeling",
        "hydrodynamic flood simulation",
        "river flow modeling using hydraulic equations",
        "flood depth and velocity simulation",
        "inundation modeling using hydraulic models",
    ],
    "hydrological_modeling": [
        "rainfall runoff modeling using SWAT",
        "hydrological simulation using HEC-HMS",
        "basin scale runoff modeling",
        "precipitation discharge modeling",
        "streamflow simulation",
        "catchment runoff forecasting",
    ],
    "spectral_index_analysis": [
        "NDVI NDWI NDII RDI spectral index analysis",
        "water and vegetation indices for land surface monitoring",
        "remote sensing indices for soil moisture and drought",
        "vegetation and moisture index analysis",
        "spectral indices for land surface analysis",
        "water index based surface condition monitoring",
    ],
    "drought_monitoring": [
        "drought monitoring using MODIS vegetation indices",
        "soil moisture assessment using NDII and RDI",
        "desertification monitoring using remote sensing",
        "climate change impact on terrestrial ecosystems",
        "aridity assessment using satellite indices",
        "soil moisture and vegetation stress monitoring",
    ],
    "land_cover_classification": [
        "land cover classification using remote sensing",
        "image classification using machine learning",
        "supervised classification of satellite imagery",
        "LULC classification using optical satellite data",
    ],
    "land_use_change_detection": [
        "land use land cover change detection",
        "land cover transition analysis using satellite imagery",
        "ecosystem change detection using remote sensing",
        "land use change monitoring in disaster affected regions",
        "vegetation and forest cover dynamics",
    ],
    "flood_damage_assessment": [
        "flood damage assessment using satellite imagery",
        "agricultural loss assessment after flood",
        "infrastructure damage mapping after flood",
        "rapid flood damage estimation using SAR data",
    ],
    "flood_susceptibility_mapping": [
        "flood susceptibility mapping using machine learning",
        "flood hazard mapping using topographic and hydrological factors",
        "urban flood susceptibility analysis",
        "flood risk mapping using GIS and remote sensing factors",
    ],
    "terrain_dem_analysis": [
        "digital elevation model analysis",
        "terrain analysis using DEM",
        "slope aspect elevation modeling",
        "topographic analysis for flood modeling",
        "geomorphometric analysis using elevation data",
    ],
    "dem_validation": [
        "digital elevation model validation using ICESat-2",
        "DEM vertical accuracy assessment",
        "terrain model comparison using reference elevation data",
        "elevation error assessment using LiDAR or ICESat-2",
    ],
    "review": [
        "review of flood mapping methods",
        "survey of remote sensing approaches",
        "overview of flood detection techniques",
        "literature review on flood monitoring",
        "systematic review of flood susceptibility mapping",
    ],
    "unknown": [
        "unknown task",
        "insufficient evidence to classify the scientific task",
    ],
}

VALID_TASK_LABELS   = set(TASK_PROTOTYPES)
VALID_STUDY_TYPES   = {
    "case_study", "regional", "multi_site",
    "global_algorithmic", "review", "unknown",
}

INLINE_HEADINGS: list[tuple[str, str]] = [
    ("introduction",  r"\bIntroduction\.\s+"),
    ("methods",       r"\bMaterials and methods\.\s+"),
    ("methods",       r"\bMaterial and methods\.\s+"),
    ("methods",       r"\bMethods\.\s+"),
    ("results",       r"\bResults(?: and discussion)?\.\s+"),
    ("discussion",    r"\bDiscussion\.\s+"),
    ("conclusion",    r"\bConclusions?\.\s+"),
]

# ─────────────────────────────────────────────────────────────────────────────
# EMBEDDING MODEL
# ─────────────────────────────────────────────────────────────────────────────

def load_embedding_model():
    from dotenv import load_dotenv
    from sentence_transformers import SentenceTransformer
    load_dotenv()
    hf_token = os.getenv("HF_TOKEN")
    if not hf_token:
        raise ValueError("HF_TOKEN not found in environment")
    os.environ["HF_HOME"]            = "./.hf_cache"
    os.environ["TRANSFORMERS_CACHE"] = "./.hf_cache"
    return SentenceTransformer(
        "sentence-transformers/all-MiniLM-L6-v2",
        device="cpu",
        use_auth_token=hf_token,
    )

model = load_embedding_model()

# ─────────────────────────────────────────────────────────────────────────────
# NLP
# ─────────────────────────────────────────────────────────────────────────────

def load_spacy(model_name: str = "en_core_web_md"):
    try:
        return spacy.load(model_name)
    except OSError:
        print(f"⚠️ spaCy model '{model_name}' not found. Installing...")
        subprocess.run([sys.executable, "-m", "spacy", "download", model_name], check=True)
        return spacy.load(model_name)

nlp = load_spacy("en_core_web_trf")

# ─────────────────────────────────────────────────────────────────────────────
# PIPELINE CONTEXT
# ─────────────────────────────────────────────────────────────────────────────

class PipelineContext:
    def __init__(self, sections: dict):
        self.sections    = ensure_dict(sections)
        self.abstract    = self.sections.get("abstract", "")
        self.introduction = self.sections.get("introduction", "")
        self.study_area  = self.sections.get("study_area", "")
        self.data_sources = self.sections.get("data_sources", "")
        self.methods     = self.sections.get("methods", "")
        self.results     = self.sections.get("results", "")
        self.conclusion  = self.sections.get("conclusion", "")
        self.other       = self.sections.get("other", "")
        self.full_text   = build_full_text(self.sections)

        self.study_country_text = " ".join([
            self.abstract, self.study_area,
            self.data_sources, self.methods, self.other,
        ])
        self.satellite_text = " ".join([
            self.abstract, self.data_sources, self.methods, self.other,
        ])
        self.method_text = " ".join([
            self.abstract, self.data_sources, self.methods,
            self.results, self.other,
        ])
        self.metric_text = " ".join([
            self.abstract, self.results, self.conclusion,
        ])

# ─────────────────────────────────────────────────────────────────────────────
# UTILITY FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────

def clean_text(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()

def snippet(text: str, start: int, end: int, window: int = 120) -> str:
    return clean_text(text[max(0, start - window): min(len(text), end + window)])

def make_hash(text: str) -> str:
    return hashlib.md5(text.encode("utf-8")).hexdigest()

def doi_to_url(doi: Optional[str]) -> Optional[str]:
    return f"https://doi.org/{doi}" if doi else None

def scholar_url(title: Optional[str]) -> Optional[str]:
    if not title:
        return None
    return "https://scholar.google.com/scholar?q=" + urllib.parse.quote(title)

def ensure_dict(value, default=None):
    if isinstance(value, dict):
        return value
    return default if default is not None else {}

def json_safe(obj):
    if isinstance(obj, dict):
        return {str(k): json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [json_safe(v) for v in obj]
    if isinstance(obj, (np.float32, np.float64)):
        return float(obj)
    if isinstance(obj, (np.int32, np.int64)):
        return int(obj)
    if isinstance(obj, Path):
        return str(obj)
    return obj

def build_full_text(sections: dict) -> str:
    return " ".join([
        sections.get("abstract", ""),
        sections.get("introduction", ""),
        sections.get("study_area", ""),
        sections.get("data_sources", ""),
        sections.get("methods", ""),
        sections.get("results", ""),
        sections.get("other", ""),
    ])

def first_text(root, xpath: str) -> Optional[str]:
    values = root.xpath(xpath, namespaces=NS)
    if not values:
        return None
    return clean_text(str(values[0]))

def all_text(node, xpath: str) -> str:
    return clean_text(" ".join(node.xpath(xpath, namespaces=NS)))

# ─────────────────────────────────────────────────────────────────────────────
# XML PARSING
# ─────────────────────────────────────────────────────────────────────────────

def parse_authors(root) -> list[dict]:
    authors = []
    for a in root.xpath("//tei:sourceDesc//tei:analytic/tei:author", namespaces=NS):
        first = clean_text(" ".join(a.xpath(".//tei:forename/text()", namespaces=NS)))
        last  = clean_text(" ".join(a.xpath(".//tei:surname/text()",  namespaces=NS)))
        email = first_text(a, ".//tei:email/text()")
        orcid = first_text(a, ".//tei:idno[@type='ORCID']/text()")
        affiliation = clean_text(" ".join(a.xpath(".//tei:affiliation//text()", namespaces=NS)))
        countries = [
            normalize_country_name(c)
            for c in a.xpath(".//tei:country/text()", namespaces=NS)
        ]
        countries = [c for c in countries if c]
        full_name = clean_text(f"{first} {last}") if first or last else None
        if full_name:
            authors.append({
                "first_name": first or None, "last_name": last or None,
                "full_name": full_name, "email": email, "orcid": orcid,
                "affiliation": affiliation or None,
                "affiliation_countries": countries,
            })
    return authors

def parse_metadata(root, xml_path: Path) -> dict:
    title = (
        first_text(root, "//tei:titleStmt/tei:title/text()")
        or first_text(root, "//tei:sourceDesc//tei:analytic/tei:title/text()")
    )
    doi     = first_text(root, "//tei:idno[@type='DOI']/text()")
    year    = first_text(root, "//tei:date/@when")
    if year:
        year = year[:4]
    journal   = first_text(root, "//tei:sourceDesc//tei:monogr/tei:title/text()")
    publisher = (
        first_text(root, "//tei:sourceDesc//tei:monogr//tei:publisher/text()")
        or first_text(root, "//tei:publicationStmt/tei:publisher/text()")
    )
    url = doi_to_url(doi) or first_text(root, "//tei:ptr/@target") or scholar_url(title)
    return {
        "paper_id":  xml_path.stem.replace(".tei", ""),
        "source_xml": str(xml_path),
        "title":     title, "doi": doi, "url": url,
        "year":      year,  "journal": journal, "publisher": publisher,
        "authors":   parse_authors(root),
    }

def section_tags(head_text: Optional[str], n_attr: Optional[str], content: str) -> set[str]:
    head = (head_text or "").lower()
    n    = (n_attr or "").strip()
    tags: set[str] = set()

    if "summary" in head or "abstract" in head:
        tags.add("abstract")
    if "intro" in head or n == "1" or n.startswith("1."):
        tags.add("introduction")
    if any(k in head for k in ["study area", "study region", "study site",
                                 "area of interest", "study area and data",
                                 "study region and data"]):
        tags.add("study_area")
    if any(k in head for k in ["dataset", "datasets", "data sources",
                                 "data", "materials"]):
        tags.add("data_sources")
    if any(k in head for k in ["method", "methods", "methodology", "workflow",
                                 "processing", "model", "models", "simulation",
                                 "algorithm", "algorithms",
                                 "change detection algorithms"]):
        tags.add("methods")
    if "data and methods" in head or "materials and methods" in head:
        tags.update({"methods", "data_sources"})
    if any(k in head for k in ["result", "results", "accuracy",
                                 "evaluation", "assessment"]):
        tags.add("results")
    if "discussion" in head:
        tags.add("discussion")
    if "conclusion" in head:
        tags.add("conclusion")
    if not tags:
        tags.add("other")
    return tags

def split_inline_sections(content: str) -> dict:
    content = content or ""
    found: list[tuple[int, int, str]] = []
    for name, pattern in INLINE_HEADINGS:
        for m in re.finditer(pattern, content, flags=re.I):
            found.append((m.start(), m.end(), name))
    if not found:
        return {"other": content}
    found.sort()
    result: dict[str, str] = {}
    prefix = clean_text(content[:found[0][0]])
    if prefix:
        result.setdefault("other", "")
        result["other"] += prefix + "\n"
    for i, (start, end, name) in enumerate(found):
        next_start = found[i + 1][0] if i + 1 < len(found) else len(content)
        block = clean_text(content[end:next_start])
        if block:
            result.setdefault(name, "")
            result[name] += block + "\n"
    return result

def parse_sections(root) -> dict:
    sections = {
        "abstract": "", "introduction": "", "study_area": "",
        "data_sources": "", "methods": "", "results": "",
        "discussion": "", "conclusion": "", "other": "",
    }
    abstract = root.xpath("//tei:abstract//tei:p//text()", namespaces=NS)
    if abstract:
        sections["abstract"] += clean_text(" ".join(abstract)) + "\n"
    for div in root.xpath("//tei:body//tei:div", namespaces=NS):
        head    = all_text(div, "./tei:head//text()")
        n_attr  = div.get("n")
        content = clean_text(" ".join(div.xpath(".//tei:p//text()", namespaces=NS)))
        if not content:
            continue
        tags = section_tags(head, n_attr, content)
        if tags == {"other"}:
            inline_sections = split_inline_sections(content)
            for tag, block in inline_sections.items():
                if tag in sections:
                    sections[tag] += block + "\n"
        else:
            for tag in tags:
                if tag in sections:
                    sections[tag] += content + "\n"
    return sections

# ─────────────────────────────────────────────────────────────────────────────
# GEO UTILITIES
# ─────────────────────────────────────────────────────────────────────────────

def normalize_country_name(name: str) -> str:
    if not isinstance(name, str):
        return ""
    return clean_text(name).replace(";", "").strip()

def ensure_country_dict(c) -> dict:
    if isinstance(c, dict):
        return {**c, "name": normalize_country_name(c.get("name")),
                "source": c.get("source", "unknown"),
                "confidence": c.get("confidence", 0.5)}
    return {"name": normalize_country_name(c), "code": None,
            "source": "unknown", "confidence": 0.5}

def ensure_list_of_country_dicts(items) -> list[dict]:
    if not items:
        return []
    result, seen = [], set()
    for item in items:
        c = ensure_country_dict(item)
        name = c.get("name")
        if not name or name in seen:
            continue
        seen.add(name)
        result.append(c)
    return result

def ensure_river_dict(r) -> dict:
    if isinstance(r, dict):
        return {**r, "name": clean_text(r.get("name")),
                "type": r.get("type", "river"),
                "source": r.get("source", "unknown")}
    return {"name": clean_text(str(r)), "type": "river", "source": "unknown"}

def ensure_list_of_river_dicts(items) -> list[dict]:
    if not items:
        return []
    result, seen = [], set()
    for item in items:
        r = ensure_river_dict(item)
        name = r.get("name")
        if not name or name in seen:
            continue
        seen.add(name)
        result.append(r)
    return result

def default_study_geo() -> dict:
    return {
        "primary_country": None, "countries": [],
        "regions": [], "rivers": [], "river_country_links": [],
        "locations": [], "coordinates": [], "confidence": 0.0,
    }

def is_valid_place(name: str) -> bool:
    if not name:
        return False
    n = name.strip().lower()
    if len(n) < 3:
        return False
    if n in INVALID_LOCATIONS:
        return False
    if any(k in n for k in ["earth", "world", "surface"]):
        return False
    return True

def classify_location(name: str) -> str:
    n = name.lower()
    if "river" in n:
        return "river"
    if "lake" in n:
        return "lake"
    if any(k in n for k in ["basin", "catchment", "delta"]):
        return "basin"
    if any(k in n for k in ["watershed"]):
        return "watershed"
    return "region"

def is_valid_region_name(name: str) -> bool:
    if not name:
        return False
    n = clean_text(name).lower()
    if len(n) < 3:
        return False
    if n in GENERIC_REGION_NOISE:
        return False
    return True

def normalize_river_name(name: str) -> str:
    name = clean_text(name)
    name = re.sub(r"^(the|a|an)\s+", "", name, flags=re.I)
    name = re.sub(r"'s$", "", name)
    name = name.replace(" River basin", "").replace(" river basin", "")
    name = name.replace(" River", "").replace(" river", "")
    name = clean_text(name)
    if name.lower() == "cagayan":
        return "Cagayan River"
    return name

def normalize_country_item(item):
    if isinstance(item, dict):
        name = clean_text(item.get("name"))
        if not name:
            return None
        return {**item, "name": name,
                "source": item.get("source", "unknown"),
                "confidence": float(item.get("confidence", 0.5))}
    name = clean_text(str(item))
    if not name:
        return None
    return {"name": name, "code": None, "source": "ner", "confidence": 0.55}

def merge_countries(existing, ner_countries) -> list[dict]:
    merged = {}
    for item in existing or []:
        c = normalize_country_item(item)
        if c:
            merged[c["name"]] = c
    for item in ner_countries or []:
        c = normalize_country_item(item)
        if not c:
            continue
        name = c["name"]
        if name not in merged:
            merged[name] = c
        else:
            merged[name]["confidence"] = max(
                float(merged[name].get("confidence", 0.5)),
                float(c.get("confidence", 0.55)),
            )
    return list(merged.values())

def compute_geo_confidence(study_geo: dict) -> float:
    geo = ensure_dict(study_geo)
    score = 0.0
    if geo.get("primary_country"):   score += 0.35
    if geo.get("countries"):         score += 0.15
    if geo.get("regions"):           score += 0.20
    if geo.get("rivers"):            score += 0.15
    if geo.get("locations"):         score += 0.10
    if geo.get("coordinates"):       score += 0.05
    return round(min(score, 1.0), 4)

def geonames_type(fcode: str) -> str:
    if not fcode:       return "unknown"
    if fcode.startswith("H"): return "river"
    if fcode.startswith("P"): return "city"
    if fcode.startswith("A"): return "admin"
    if fcode.startswith("T"): return "terrain"
    return "other"

def geonames_lookup(name: str) -> Optional[dict]:
    global LAST_CALL
    if not name:
        return None
    delay   = 1.0
    elapsed = time.time() - LAST_CALL
    if elapsed < delay:
        time.sleep(delay - elapsed)
    LAST_CALL = time.time()
    url = "http://api.geonames.org/searchJSON"
    params = {"q": name, "maxRows": 5, "username": "viktornikoriak"}
    try:
        r = requests.get(url, params=params, timeout=5)
        if r.status_code != 200:
            return None
        results = r.json().get("geonames", [])
        if not results:
            return None
        best = next(
            (g for g in results if g.get("fcode", "").startswith(("P","A","H","T"))),
            results[0]
        )
        lat, lon = best.get("lat"), best.get("lng")
        if lat is None or lon is None:
            return None
        return {
            "name": best.get("name"), "lat": float(lat), "lon": float(lon),
            "type": geonames_type(best.get("fcode")),
            "country": best.get("countryName"),
            "feature": best.get("fcodeName"),
            "feature_code": best.get("fcode"), "source": "geonames",
        }
    except Exception:
        return None

def geocode_place(name: str) -> Optional[dict]:
    if not is_valid_place(name):
        return None
    if name in GEO_CACHE:
        return GEO_CACHE[name]
    try:
        g = geonames_lookup(name)
        if g:
            res = {"lat": g["lat"], "lon": g["lon"],
                   "country": g.get("country"), "feature": g.get("feature"),
                   "source": "geonames"}
            GEO_CACHE[name] = res
            return res
    except Exception:
        pass
    try:
        url    = "https://nominatim.openstreetmap.org/search"
        params = {"q": name, "format": "json", "limit": 1}
        r      = requests.get(url, params=params, headers={"User-Agent": "geo-parser"})
        data   = r.json()
        if data:
            res = {"lat": float(data[0]["lat"]), "lon": float(data[0]["lon"]),
                   "source": "nominatim"}
            GEO_CACHE[name] = res
            return res
    except Exception:
        pass
    GEO_CACHE[name] = None
    return None

# ─────────────────────────────────────────────────────────────────────────────
# NER
# ─────────────────────────────────────────────────────────────────────────────

def extract_geo_ner(text: str) -> tuple[list[str], list[str]]:
    doc = nlp(text)
    countries, locations = set(), set()
    for ent in doc.ents:
        name = ent.text.strip()
        if not is_valid_place(name):
            continue
        if ent.label_ == "GPE":
            countries.add(name)
        elif ent.label_ == "LOC":
            locations.add(name)
    return list(countries), list(locations)

def extract_tei_countries(root) -> list[dict]:
    found = {}
    for c in root.xpath("//tei:country", namespaces=NS):
        name = normalize_country_name(clean_text(" ".join(c.xpath(".//text()"))))
        code = c.get("key")
        if name:
            found[name] = {"name": name, "code": code,
                           "source": "tei", "confidence": 0.95}
    return list(found.values())

def author_name_set(metadata: dict) -> set[str]:
    names = set()
    for author in metadata.get("authors", []):
        for key in ["first_name", "last_name", "full_name"]:
            value = author.get(key)
            if not value:
                continue
            for part in str(value).split():
                p = clean_text(part).lower()
                if len(p) > 2:
                    names.add(p)
    return names

def extract_author_geo(metadata: dict) -> list[dict]:
    countries = {}
    for author in metadata.get("authors", []):
        for country in author.get("affiliation_countries", []):
            name = normalize_country_name(country)
            if name:
                countries[name] = {"name": name,
                                   "source": "tei_author_affiliation",
                                   "confidence": 0.95}
    return list(countries.values())

# ─────────────────────────────────────────────────────────────────────────────
# EMBEDDING CLASSIFICATION
# ─────────────────────────────────────────────────────────────────────────────

def classify_with_embeddings(text: str):
    text_emb = model.encode([text or ""])[0]
    scores   = {}
    for label, examples in STUDY_TYPE_PROTOTYPES.items():
        sim = cosine_similarity([text_emb], model.encode(examples)).mean()
        scores[label] = float(sim)
    best = max(scores, key=scores.get)
    return best, scores

def classify_task_with_embeddings(text: str):
    text_emb = model.encode([text or ""])[0]
    scores   = {}
    for label, examples in TASK_PROTOTYPES.items():
        sim = cosine_similarity([text_emb], model.encode(examples)).mean()
        scores[label] = float(sim)
    best = max(scores, key=scores.get)
    return best, scores

def embedding_score(text: str, label_examples: list) -> float:
    text_emb = model.encode([text])[0]
    ex_embs  = model.encode(label_examples)
    sims     = cosine_similarity([text_emb], ex_embs)[0]
    return float(max(sims))

def region_context_score(region_name: str, text: str) -> float:
    text        = text or ""
    region_name = region_name or ""
    matches     = list(re.finditer(re.escape(region_name), text, re.I))
    if not matches:
        return 0.0
    query    = f"actual study region or study area: {region_name}"
    emb_q    = model.encode([query])[0]
    best     = 0.0
    for m in matches[:5]:
        ctx    = snippet(text, m.start(), m.end(), window=500)
        emb_c  = model.encode([ctx])[0]
        sim    = cosine_similarity([emb_q], [emb_c])[0][0]
        best   = max(best, float(sim))
    return float(best)

def refine_study_area_with_embeddings(name: str, text: str) -> Optional[str]:
    if not name:
        return name
    sim = cosine_similarity(
        [model.encode([name])[0]],
        [model.encode([text])[0]]
    )[0][0]
    return name if sim >= 0.2 else None

# ─────────────────────────────────────────────────────────────────────────────
# ENTITY SCORING & ROLE DETECTION
# ─────────────────────────────────────────────────────────────────────────────

def best_mention_context(name: str, text: str, window: int = 500) -> str:
    if not name or not text:
        return ""
    matches = list(re.finditer(re.escape(name), text, flags=re.I))
    if not matches:
        return ""
    scored = []
    for m in matches[:10]:
        ctx   = snippet(text, m.start(), m.end(), window=window)
        score = sum(1 for k in [
            "study area", "study site", "tested on", "test site",
            "flood event", "flood events", "occurred in", "located in",
            "case study", "data sources", "ground truth", "validation data",
        ] if k in ctx.lower())
        scored.append((score, ctx))
    scored.sort(key=lambda x: x[0], reverse=True)
    return scored[0][1]

# ─────────────────────────────────────────────────────────────────────────────
# SCS DISAMBIGUATION
# ─────────────────────────────────────────────────────────────────────────────

_SCS_HYDRO_SIGNALS = {
    "curve number", "scs-cn", "scs cn", "unit hydrograph",
    "soil conservation service", "nrcs", "rainfall-runoff",
    "loss estimation", "runoff estimation", "green and ampt",
    "muskingum", "hec-hms", "hec hms",
}

_SCS_HYDRO_META = {
    "full_name":  "Soil Conservation Service",
    "type":       "method",
    "type_group": "method",
    "domain":     "hydrology",
    "definition": "USDA method for estimating surface runoff from rainfall "
                  "using the Curve Number (CN) approach.",
    "used_for":   ["rainfall-runoff modelling", "loss estimation",
                   "runoff estimation", "hydrological modelling"],
    "related":    ["HEC-HMS", "SWAT", "CN", "NRCS"],
    "contexts":   ["hydrology", "watershed"],
    "source_kb":  "disambiguation",
    "disambiguated": "hydrology",
}


def disambiguate_scs(entity: dict, ctx_text: str) -> dict:
    """
    SCS can mean either:
      - Spectral Correlation Similarity  (remote sensing)
      - Soil Conservation Service / SCS-CN  (hydrology)
    Select the correct interpretation from local context.
    """
    if entity.get("name", "").upper() != "SCS":
        return entity
    ctx = best_mention_context("SCS", ctx_text, window=400).lower()
    if any(sig in ctx for sig in _SCS_HYDRO_SIGNALS):
        entity.setdefault("kb_metadata", {}).update(_SCS_HYDRO_META)
    return entity


# ─────────────────────────────────────────────────────────────────────────────
# SECTION-AWARE IMPORTANCE BOOSTING
# ─────────────────────────────────────────────────────────────────────────────

_SECTION_BOOSTS: dict[str, float] = {
    "abstract":    0.20,
    "methods":     0.25,
    "conclusion":  0.15,
    "study_area":  0.10,
    "data_sources": 0.10,
}
# title boost handled separately via metadata


def compute_section_boost(name: str, ctx: "PipelineContext", title: str = "") -> float:
    """Return additive score boost based on how many key sections mention the entity."""
    name_l = name.lower()
    boost  = 0.0
    if title and name_l in title.lower():
        boost += 0.25
    for section, weight in _SECTION_BOOSTS.items():
        text = getattr(ctx, section, "")
        if text and name_l in text.lower():
            boost += weight
    return boost  # caller clamps to 1.0


def add_context_score(entity: dict, ctx_text: str) -> dict:
    if not entity.get("name") or not ctx_text:
        return entity
    ctx   = best_mention_context(entity["name"], ctx_text)
    score = 0
    ctx_l = ctx.lower()
    if any(k in ctx_l for k in ["study area", "case study", "located in",
                                  "study site", "basin"]):
        score += 0.5
    if any(k in ctx_l for k in ["used", "applied", "analysis", "simulation"]):
        score += 0.3
    entity["scores"]["context"] = score
    entity["evidence"] = ctx or entity["evidence"]
    return entity

def detect_role(entity: dict, ctx_text: str) -> dict:
    ctx   = best_mention_context(entity["name"], ctx_text)
    ctx_l = ctx.lower()
    if any(k in ctx_l for k in ["used", "applied", "model", "analysis",
                                  "derived from", "calculated"]):
        entity["role"] = "used"
    else:
        entity["role"] = "mentioned"
    return entity

def compute_entity_score(entity: dict) -> dict:
    s     = entity["scores"]
    score = (
        s.get("pattern",   0) * 0.3 +
        s.get("context",   0) * 0.3 +
        s.get("embedding", 0) * 0.2 +
        s.get("llm",       0) * 0.2
    )
    entity["final_score"] = round(score, 4)
    return entity

def decide_entity(entity: dict, threshold: float = 0.6) -> dict:
    entity = compute_entity_score(entity)
    entity["accepted"] = entity["final_score"] >= threshold
    return entity

def apply_llm_judge(entity: dict, judge_result: dict) -> dict:
    if not judge_result:
        return entity
    accepted   = judge_result.get("accepted", True)
    confidence = judge_result.get("confidence", 0.5)
    entity["scores"]["llm"] = confidence
    if not accepted:
        entity["final_score"] *= 0.5
        entity["accepted"]     = False
    return entity

def resolve_geo_entity(entity: dict, context_text: str) -> dict:
    geo = geonames_lookup(entity["name"])
    if not geo:
        return entity
    score = 0
    if geo["country"] and geo["country"].lower() in context_text.lower():
        score += 0.5
    if geo["type"] == entity.get("type"):
        score += 0.3
    entity["geo"] = geo
    entity["scores"]["context"] += score
    return entity

# ─────────────────────────────────────────────────────────────────────────────
# STUDY TYPE & TASK DETECTION
# ─────────────────────────────────────────────────────────────────────────────

def is_valid_study_type_label(label: Optional[str]) -> bool:
    return label in VALID_STUDY_TYPES

def is_study_area_context(ctx: str) -> bool:
    ctx_l    = (ctx or "").lower()
    positive = any(k in ctx_l for k in [
        "study area", "study site", "test site", "tested on", "tested using",
        "we tested", "we applied", "case study", "flood event", "flood events",
        "occurred in", "located in", "ground truth data", "validation data",
        "data sources", "two urban flood events",
    ])
    negative = any(k in ctx_l for k in [
        "previous studies", "several studies", "many studies", "literature",
        "ref.", "et al.", "state of the art", "existing methods",
        "in contrast", "for example",
    ])
    return positive and not negative

def sanitize_study_type(study_type_obj: dict) -> dict:
    study_type_obj = ensure_dict(study_type_obj)
    label          = study_type_obj.get("label")
    if label not in VALID_STUDY_TYPES:
        return {"label": "unknown", "confidence": 0.4,
                "source": "sanitized_invalid_study_type",
                "previous": label, "needs_judge": True}
    return study_type_obj

def detect_study_type(sections: dict, title: str = "") -> dict:
    sections  = ensure_dict(sections)
    title_abs = f"{title} {sections.get('abstract','')}".lower()
    strong    = f"{title} {sections.get('abstract','')} {sections.get('study_area','')} {sections.get('methods','')[:1500]} {sections.get('results','')[:1000]}".lower()

    flood_case    = "flood" in strong and any(k in strong for k in [
        "tested on","tested using","we tested","we applied","applied to",
        "validated on","evaluated on","study area","study site","case study",
        "case area","test site","pilot area","event occurred","occurred in",
        "flood event","flood events","flooded area","flood extent",
        "ground truth data","post-flood","preflood","pre-flood",
    ])
    general_case  = any(k in strong for k in [
        "tested on","tested using","we tested","we applied","applied to",
        "validated on","evaluated on","study area","study site","case study",
        "case area","test site","pilot area","event occurred","occurred in",
        "study was conducted","dataset was collected",
    ])
    regional      = any(k in strong for k in [
        "regional scale","national scale","country scale","across the country",
        "entire country","whole country","territory of","large region",
        "administrative region","natural zones","multiple stations",
        "weather stations","meteorological stations",
    ])
    multi_site    = any(k in strong for k in [
        "multiple case studies","several case studies","five case studies",
        "multiple locations","multiple sites","different locations",
        "different regions","various regions","multiple flood events",
        "several flood events","two flood events","different test sites",
        "benchmark sites",
    ])
    global_ev     = any(k in strong for k in [
        "global scale","global-scale","worldwide","anywhere in the world",
        "global application","global flood monitoring","global basis",
        "near real-time on a global basis","globally available",
        "global datasets","worldwide application",
    ])
    review_ev     = any(k in title_abs for k in [
        "systematic review","literature review","review paper","this review",
        "we review","review of","state-of-the-art review","survey of",
        "overview of","meta-analysis",
    ])
    weak_review   = any(k in strong for k in [
        "previous studies","several studies","many studies","existing methods",
        "state of the art","related work","literature","literature review",
    ])

    if review_ev and not (flood_case or general_case):
        return {"label": "review", "confidence": 0.95,
                "source": "rules_title_abstract", "needs_judge": False}
    if multi_site:
        return {"label": "multi_site", "confidence": 0.88,
                "source": "rules", "needs_judge": False}
    if flood_case:
        return {"label": "case_study", "confidence": 0.92,
                "source": "rules_flood_case", "needs_judge": False}
    if regional and not general_case:
        return {"label": "regional", "confidence": 0.88,
                "source": "rules", "needs_judge": False}
    if general_case:
        return {"label": "case_study", "confidence": 0.9,
                "source": "rules", "needs_judge": False}
    if global_ev:
        return {"label": "global_algorithmic", "confidence": 0.85,
                "source": "rules", "needs_judge": bool(weak_review)}

    label, scores = classify_with_embeddings(strong)
    confidence    = float(scores[label])
    if label not in VALID_STUDY_TYPES:
        return {"label": "unknown", "confidence": 0.4,
                "source": "embeddings_invalid_label",
                "needs_judge": True, "scores": scores}
    return {"label": label, "confidence": confidence,
            "source": "embeddings",
            "needs_judge": confidence < 0.85 or weak_review, "scores": scores}

def make_task(label: str, confidence: float, source: str = "rules") -> dict:
    if label not in VALID_TASK_LABELS:
        raise ValueError(f"Unknown task label: {label}")
    return {"label": label, "confidence": float(confidence), "source": source}

# Strong hydrological modelling signals — checked BEFORE generic flood rules.
# Papers using these tools are rainfall-runoff / hydrological, not SAR flood mappers.
_HYDRO_MODEL_SIGNALS = [
    "hec-hms", "hec hms",
    "scs-cn", "scs cn", "curve number",
    "unit hydrograph", "muskingum",
    "rainfall-runoff modelling", "rainfall runoff modelling",
    "rainfall-runoff modeling", "rainfall runoff modeling",
]


def classify_task(ctx: PipelineContext) -> dict:
    text = ctx.full_text.lower()

    # Priority 1: strong hydrological-modelling signals override everything
    if any(k in text for k in _HYDRO_MODEL_SIGNALS):
        return make_task("hydrological_modeling", 0.92)

    # Priority 2: explicit hydraulic simulation (HEC-RAS / 2-D models)
    if any(k in text for k in ["hec-ras", "2d flood model", "hydraulic simulation",
                                "hydraulic flood", "1d/2d"]):
        return make_task("flood_modeling_hydraulic", 0.88)

    # Priority 3: SWAT / generic runoff-watershed combination
    if any(k in text for k in ["swat"]) or (
        "runoff" in text and any(k in text for k in ["watershed", "streamflow", "hydrograph"])
    ):
        return make_task("hydrological_modeling", 0.88)

    # Priority 4: satellite-based flood mapping (only after ruling out hydro models)
    if "flood" in text and any(k in text for k in [
        "mapping", "extent", "inundation", "water extent"
    ]):
        return make_task("flood_mapping_satellite", 0.9)

    # Priority 5: spectral index analysis
    if any(k in text for k in ["ndvi", "ndwi", "ndii", "rdi"]):
        return make_task("spectral_index_analysis", 0.88)

    label, scores = classify_task_with_embeddings(text)
    if label not in VALID_TASK_LABELS:
        return make_task("unknown", 0.4)
    return make_task(label, scores[label], source="embeddings")

# ─────────────────────────────────────────────────────────────────────────────
# GEO EXTRACTION
# ─────────────────────────────────────────────────────────────────────────────

def detect_countries(text: str, root) -> list[dict]:
    found = {}
    for c in extract_tei_countries(root):
        c = ensure_country_dict(c)
        found[c["name"]] = c
    for name, pattern in COUNTRY_PATTERNS.items():
        m = re.search(pattern, text, re.I)
        if m and name not in found:
            found[name] = {"name": name, "code": None, "source": "regex",
                           "confidence": 0.75,
                           "evidence": snippet(text, m.start(), m.end())}
    return ensure_list_of_country_dicts(list(found.values()))

def detect_study_countries_from_text(text: str) -> list[dict]:
    found = {}
    for name, pattern in COUNTRY_PATTERNS.items():
        m = re.search(pattern, text, re.I)
        if m and name not in found:
            found[name] = {"name": name, "code": None,
                           "source": "regex_study_text", "confidence": 0.75,
                           "evidence": snippet(text, m.start(), m.end())}
    return ensure_list_of_country_dicts(list(found.values()))

def extract_rivers(text: str) -> list[dict]:
    text     = text or ""
    entities = {}
    for pattern in RIVER_PATTERNS:
        for m in re.finditer(pattern, text):
            raw  = clean_text(m.group(1))
            name = normalize_river_name(raw)
            if len(name) < 3:
                continue
            if name not in entities:
                entities[name] = {
                    "name": name, "type": "river",
                    "evidence": snippet(text, m.start(), m.end()),
                    "sources": ["regex"],
                    "scores": {"pattern": 0.8, "context": 0.0, "embedding": 0.0, "llm": 0.0},
                    "final_score": 0.0, "accepted": None, "role": None,
                }
    return list(entities.values())

# ─────────────────────────────────────────────────────────────────────────────
# RIVER SCORING
# ─────────────────────────────────────────────────────────────────────────────

# Signals that indicate the river is cited from prior literature, not the study area
_RIVER_CITATION_SIGNALS = [
    "et al", "ismail (", "modi (", "by adopting", "modelled by",
    "previous study", "studied by", "proposed by", "introduced by",
    "according to", "in contrast to", "in the study of",
]
# Signals that indicate the river IS the actual study watershed
_RIVER_STUDY_SIGNALS = [
    "study area", "study basin", "study watershed", "located in",
    "situated in", "the watershed", "our study", "this study",
    "case study", "urban catchment", "the basin", "river basin is",
    "study site", "the study", "research area",
]


def score_rivers(rivers: list[dict], ctx: "PipelineContext") -> list[dict]:
    """
    Score and filter rivers.
    - Rivers mentioned only in citation/literature context are rejected.
    - Rivers with study-area evidence from abstract/study_area/methods are accepted.
    """
    core_text = " ".join(filter(None, [
        ctx.abstract, ctx.study_area, ctx.methods, ctx.data_sources,
    ])).lower()

    result = []
    for r in rivers:
        ev    = r.get("evidence", "").lower()
        name  = r.get("name", "")
        name_l = name.lower()

        is_citation  = any(sig in ev for sig in _RIVER_CITATION_SIGNALS)
        is_study     = any(sig in ev for sig in _RIVER_STUDY_SIGNALS)
        in_core_text = name_l in core_text

        ctx_score = 0.0
        if is_study and not is_citation:
            ctx_score = 0.9
        elif in_core_text and not is_citation:
            ctx_score = 0.5

        r["scores"]["context"] = ctx_score
        r["source"] = "study_area" if (is_study and not is_citation) else (
            "citation" if is_citation else "unknown"
        )
        r = compute_entity_score(r)
        r["accepted"] = ctx_score > 0.4
        r["role"]     = "study_watershed" if r["accepted"] else "citation_reference"
        result.append(r)
    return result


def enrich_river_country(rivers: list[dict]) -> list[dict]:
    links = []
    for river in ensure_list_of_river_dicts(rivers):
        name = river.get("name")
        if name in RIVER_TO_COUNTRY:
            links.append({"river": name, "country": RIVER_TO_COUNTRY[name],
                          "source": "lookup", "confidence": 0.85})
    return links

def extract_regions(
    text: str,
    ner_locations: Optional[list[str]] = None,
    invalid_names: Optional[set[str]] = None,
) -> list[dict]:
    text          = text or ""
    invalid_names = invalid_names or set()
    candidates    = {}

    for name, pattern in REGION_PATTERNS.items():
        m = re.search(pattern, text, re.I)
        if m and is_valid_region_name(name):
            candidates[name] = {"name": name, "type": "region",
                                "source": "regex_region", "confidence": 0.85,
                                "evidence": snippet(text, m.start(), m.end())}

    for loc in ner_locations or []:
        loc_name = clean_text(loc)
        if not loc_name or loc_name.lower() in invalid_names:
            continue
        if not is_valid_place(loc_name) or not is_valid_region_name(loc_name):
            continue
        loc_type = classify_location(loc_name)
        if loc_type not in {"region", "basin", "watershed"}:
            continue
        ctx = best_mention_context(loc_name, text, window=500)
        if not is_study_area_context(ctx):
            continue
        candidates.setdefault(loc_name, {
            "name": loc_name, "type": loc_type,
            "source": "ner_region_context", "confidence": 0.7, "evidence": ctx,
        })

    results = []
    for item in candidates.values():
        score = region_context_score(item["name"], text)
        item["semantic_score"] = round(float(score), 4)
        if item["source"] == "regex_region":
            item["confidence"] = min(float(item["confidence"]) + 0.05, 0.95)
            results.append(item)
        elif score >= 0.35:
            item["confidence"] = min(float(item["confidence"]) + float(score), 0.9)
            results.append(item)

    return sorted(results, key=lambda x: x["confidence"], reverse=True)

def normalize_regions(regions) -> list[dict]:
    if not regions:
        return []
    result = []
    for r in regions:
        if isinstance(r, dict):
            name = clean_text(r.get("name"))
            item = {**r, "name": name, "type": r.get("type", "region"),
                    "source": r.get("source", "unknown"),
                    "confidence": float(r.get("confidence", 0.5))}
        else:
            name = clean_text(str(r))
            item = {"name": name, "type": "region",
                    "source": "unknown", "confidence": 0.5}
        if not is_valid_region_name(name):
            continue
        result.append(item)
    return result

def extract_study_area_structured(text: str) -> Optional[dict]:
    result = {}
    m = re.search(r"([A-Z][a-zA-Z\s\-]+(?:Reserve|Park|Basin|Catchment|Region))", text)
    if m:
        result["name"] = m.group(1).strip()
    for country, pattern in COUNTRY_PATTERNS.items():
        if re.search(pattern, text, re.I):
            result["country"] = country
            break
    region_match = re.search(r"([A-Z][a-z]+ region|[A-Z][a-z]+ oblast)", text, re.I)
    if region_match:
        result["region"] = region_match.group(1)
    coord_match = re.search(r"(\d{2})[°\s]+(\d{2}).*?N.*?(\d{2})[°\s]+(\d{2}).*?E", text)
    if coord_match:
        lat1, lat2, lon1, lon2 = coord_match.groups()
        result["coordinates"] = {
            "lat_min": float(f"{lat1}.{lat2}"),
            "lat_max": float(f"{lat1}.{int(lat2)+5}"),
            "lon_min": float(f"{lon1}.{lon2}"),
            "lon_max": float(f"{lon1}.{int(lon2)+5}"),
        }
    if result and result.get("name"):
        result["name"] = refine_study_area_with_embeddings(result["name"], text)
    return result if result else None

def extract_data_geo(sections: dict) -> dict:
    text = " ".join([
        sections.get("abstract", ""), sections.get("introduction", ""),
        sections.get("methods", ""),  sections.get("results", ""),
    ]).lower()
    flags = []
    if "global coverage" in text or "earth's entire surface" in text or "global scale" in text:
        flags.append("global")
    if "multiple flood images" in text or "hundreds of" in text or "eight data sets" in text:
        flags.append("multi_site")
    if "modis archive" in text or "earth engine" in text:
        flags.append("global_satellite_archive")
    return {"scope": flags or ["unknown"], "source": "text_semantic_rules",
            "confidence": 0.85 if flags else 0.4}

def validate_with_ner(study_geo: dict, ner_countries, ner_locations) -> dict:
    validated = ensure_dict(study_geo, default_study_geo()).copy()
    validated.pop("study_geo", None)
    if ner_countries:
        validated["countries"] = merge_countries(
            validated.get("countries", []), ner_countries
        )
    validated_locations = []
    for loc in validated.get("locations", []):
        loc = ensure_dict(loc)
        if not loc:
            continue
        if loc.get("source") == "ner_study_area_context" and loc.get("evidence"):
            loc["validated"]   = True
            loc["confidence"]  = min(1.0, float(loc.get("confidence", 0.7)) + 0.1)
            validated_locations.append(loc)
    validated["locations"]   = validated_locations
    validated["confidence"]  = compute_geo_confidence(validated)
    return validated

def enrich_with_coordinates(geo: dict) -> dict:
    enriched, seen = [], set()
    geo["countries"] = ensure_list_of_country_dicts(geo.get("countries", []))
    geo["rivers"]    = ensure_list_of_river_dicts(geo.get("rivers", []))

    for c in geo["countries"]:
        name = c.get("name")
        if not is_valid_place(name) or name in seen:
            continue
        seen.add(name)
        res = geocode_place(name)
        if res:
            enriched.append({"name": name, "type": "country",
                             "lat": res["lat"], "lon": res["lon"],
                             "source": res.get("source")})

    for loc in geo.get("locations", []):
        if not isinstance(loc, dict):
            continue
        name = loc.get("name")
        conf = float(loc.get("confidence", 0.0))
        if conf < 0.7:
            continue
        if loc.get("source") == "ner" and not loc.get("evidence"):
            continue
        if not is_valid_place(name) or name in seen:
            continue
        seen.add(name)
        res = geocode_place(name)
        if res:
            enriched.append({"name": name, "type": loc.get("type"),
                             "lat": res["lat"], "lon": res["lon"],
                             "source": res.get("source")})

    for r in geo["rivers"]:
        name = r.get("name")
        if not is_valid_place(name) or name in seen:
            continue
        seen.add(name)
        res = geocode_place(name)
        if res:
            enriched.append({"name": name, "type": "river",
                             "lat": res["lat"], "lon": res["lon"],
                             "source": res.get("source")})

    geo["coordinates"] = enriched
    return geo

# ─────────────────────────────────────────────────────────────────────────────
# ENTITY EXTRACTION  (KB-DRIVEN)
# ─────────────────────────────────────────────────────────────────────────────

def extract_geo(root, ctx: PipelineContext, metadata: dict) -> dict:
    text          = ctx.full_text
    study_type_obj = sanitize_study_type(
        detect_study_type(ctx.sections, metadata.get("title", ""))
    )
    author_geo     = ensure_list_of_country_dicts(extract_author_geo(metadata))
    data_geo       = extract_data_geo(ctx.sections)
    authors        = author_name_set(metadata)

    ner_countries, ner_locations = extract_geo_ner(text)

    detected_countries  = ensure_list_of_country_dicts(
        detect_study_countries_from_text(ctx.study_country_text)
    )
    study_area_struct = extract_study_area_structured(ctx.study_area)
    rivers      = ensure_list_of_river_dicts(extract_rivers(text))
    rivers      = score_rivers(rivers, ctx)
    river_links = enrich_river_country([r for r in rivers if r.get("accepted")])
    regions           = normalize_regions(
        extract_regions(text, ner_locations, invalid_names=authors)
    )

    locations = []
    for loc in ner_locations:
        loc_name = clean_text(loc)
        if not loc_name or loc_name.lower() in authors:
            continue
        if not is_valid_place(loc_name):
            continue
        ctx_snippet = best_mention_context(loc_name, text, window=500)
        if not is_study_area_context(ctx_snippet):
            continue
        locations.append({"name": loc_name, "type": classify_location(loc_name),
                          "source": "ner_study_area_context", "confidence": 0.75,
                          "evidence": ctx_snippet})

    primary_country = None
    if study_area_struct and study_area_struct.get("country"):
        primary_country = study_area_struct["country"]
    elif detected_countries:
        primary_country = detected_countries[0]["name"]
    elif river_links:
        primary_country = river_links[0]["country"]

    study_geo = default_study_geo()
    study_geo.update({
        "primary_country": primary_country,
        "countries":       detected_countries,
        "regions":         regions,
        "rivers":          rivers,
        "river_country_links": river_links,
        "locations":       locations,
        "coordinates":     [],
        "confidence":      0.0,
    })
    study_geo = validate_with_ner(study_geo, ner_countries, ner_locations)

    return {"study_type": study_type_obj, "author_geo": author_geo,
            "study_geo": study_geo, "data_geo": data_geo,
            "note": "context-aware geo extraction"}


def extract_entities(root, ctx: PipelineContext, metadata: dict, title: str = "") -> dict:
    """
    Main extraction entry point.
    Layers: deterministic (KB patterns) → probabilistic (scoring) → (validation via judge at call site).
    """
    kb        = _get_kb()
    extractor = EntityExtractor(kb)

    geo = extract_geo(root, ctx, metadata)

    # deterministic layer: KB-driven pattern matching
    satellites = extractor.extract_satellites(ctx.satellite_text)
    dems       = extractor.extract_dems(ctx.satellite_text)
    methods    = extractor.extract_methods(ctx.method_text)
    metrics    = extractor.extract_metrics(ctx.metric_text)

    # probabilistic layer: context + section-boost + embedding + disambiguation
    satellites = run_entity_pipeline(satellites, ctx, title=title)
    dems       = run_entity_pipeline(dems,       ctx, title=title)
    methods    = run_entity_pipeline(methods,    ctx, title=title)

    return {
        "geo":        geo,
        "satellites": satellites,
        "dems":       dems,
        "methods":    methods,
        "metrics":    metrics,
    }

# ─────────────────────────────────────────────────────────────────────────────
# ENTITY PIPELINE  (scoring + LLM judge)
# ─────────────────────────────────────────────────────────────────────────────

def run_entity_pipeline(
    entities: list[dict],
    ctx: "PipelineContext",
    judge=None,
    title: str = "",
) -> list[dict]:
    result = []
    for e in entities:
        # deterministic layer: context scoring + role
        e = add_context_score(e, ctx.full_text)
        e = detect_role(e, ctx.full_text)

        # disambiguation: SCS hydrology vs remote sensing
        e = disambiguate_scs(e, ctx.full_text)

        # probabilistic layer: embedding score vs methods text
        if e["scores"].get("embedding", 0) == 0 and ctx.method_text:
            e["scores"]["embedding"] = embedding_score(
                e["name"], [ctx.method_text[:2000]]
            )

        # scoring: pattern*0.3 + context*0.3 + embedding*0.2 + llm*0.2
        e = compute_entity_score(e)

        # section-aware importance boost (added on top, clamped to 1.0)
        boost = compute_section_boost(e["name"], ctx, title=title)
        e["final_score"] = min(1.0, round(e["final_score"] + boost, 4))

        e["accepted"] = e["final_score"] >= 0.3
        result.append(e)
    return result

# ─────────────────────────────────────────────────────────────────────────────
# OLLAMA JUDGE
# ─────────────────────────────────────────────────────────────────────────────

class OllamaJudge:
    def __init__(self, base_url="http://localhost:11434",
                 model="llama3.1:8b", timeout=120):
        self.url     = f"{base_url.rstrip('/')}/api/generate"
        self.model   = model
        self.timeout = timeout

    def judge(self, candidate_json: dict, sections: dict) -> dict:
        prompt   = self._build_prompt(candidate_json, sections)
        response = requests.post(
            self.url,
            json={"model": self.model, "prompt": prompt, "stream": False,
                  "format": "json", "options": {"temperature": 0, "top_p": 0.2}},
            timeout=self.timeout,
        )
        response.raise_for_status()
        return self._parse_json(response.json().get("response", ""))

    def _parse_json(self, text: str) -> dict:
        text = text.strip()
        # strip <think>...</think> blocks emitted by some reasoning models
        text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
        # strip markdown fences
        text = re.sub(r"```[a-z]*\n?", "", text).strip()
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            match = re.search(r"\{.*\}", text, re.DOTALL)
            if not match:
                log.warning("Ollama returned non-JSON: %s", text[:200])
                return {}
            try:
                return json.loads(match.group(0))
            except json.JSONDecodeError:
                log.warning("Ollama JSON parse failed: %s", text[:200])
                return {}

    def _build_prompt(self, candidate_json: dict, sections: dict) -> str:
        title    = candidate_json.get("title") or sections.get("title") or ""
        abstract = (sections.get("abstract") or "")[:2500]
        methods  = "\n".join(filter(None, [
            sections.get("methods", ""),
            sections.get("study_area", ""),
            sections.get("data_sources", ""),
        ]))[:3500]
        return f"""You are a scientific validation system.

STRICT RULES:
- Respond ONLY in English. Do NOT use any other language.
- Return ONLY valid JSON. Do NOT include markdown. Do NOT include explanations outside JSON.
- You are NOT allowed to invent new data.
- You are ONLY allowed to validate or correct existing extracted data.
- If uncertain, return "unknown".

Allowed study_type values: case_study | regional | multi_site | global_algorithmic | review | unknown
Allowed task values: flood_mapping_satellite | flood_modeling_hydraulic | hydrological_modeling | spectral_index_analysis | drought_monitoring | land_cover_classification | land_use_change_detection | flood_damage_assessment | flood_susceptibility_mapping | terrain_dem_analysis | dem_validation | review | unknown

Validation rules:
1. Study country must come from study area/title/abstract/methods — NOT author affiliation.
2. Rivers accepted only if they are the actual study watershed, not a citation reference.
3. Sentinel/Landsat/MODIS/RADARSAT are satellites; SRTM/ASTER/FABDEM/Copernicus DEM are DEM datasets.
4. If accepted=true, corrected_value MUST be null. If corrected_value differs from original_value, accepted MUST be false.

INPUT:
{json.dumps(candidate_json, ensure_ascii=False, indent=2)}

EVIDENCE:
TITLE: {title[:800]}
ABSTRACT: {abstract}
METHODS: {methods}

OUTPUT JSON ONLY:
{{
  "paper_id": "...",
  "study_type": {{"accepted": true, "original_value": "...", "corrected_value": null, "confidence": 0.0}},
  "study_country": {{"accepted": true, "original_value": "...", "corrected_value": null, "confidence": 0.0}},
  "rivers": [],
  "data_sources": [],
  "task": {{"accepted": true, "original_value": "...", "corrected_value": null, "confidence": 0.0}}
}}"""


def is_ollama_available(base_url="http://localhost:11434") -> bool:
    try:
        r = requests.get(f"{base_url.rstrip('/')}/api/tags", timeout=5)
        return r.status_code == 200
    except Exception:
        return False

def is_valid_judge_task_verdict(task_verdict: dict) -> bool:
    task_verdict = ensure_dict(task_verdict)
    corrected    = task_verdict.get("corrected_value")
    accepted     = task_verdict.get("accepted")
    original     = task_verdict.get("original_value")
    if corrected and corrected not in VALID_TASK_LABELS:
        return False
    if accepted is True and corrected not in {None, original}:
        return False
    return True

def is_valid_judge_study_type_verdict(verdict: dict) -> bool:
    verdict   = ensure_dict(verdict)
    accepted  = verdict.get("accepted")
    original  = verdict.get("original_value")
    corrected = verdict.get("corrected_value")
    if original  and original  not in VALID_STUDY_TYPES: return False
    if corrected and corrected not in VALID_STUDY_TYPES: return False
    if accepted is True and corrected not in {None, original}: return False
    return True

def needs_judge(paper: dict) -> bool:
    entities    = ensure_dict(paper.get("entities"))
    geo         = ensure_dict(entities.get("geo"))
    study_geo   = ensure_dict(geo.get("study_geo"))
    study_type  = ensure_dict(geo.get("study_type"))
    task        = ensure_dict(entities.get("task"))
    task_label  = task.get("label", "")
    methods     = entities.get("methods", [])

    accepted_method_names = {
        m.get("name", "").upper() for m in methods if m.get("accepted")
    }
    _HYDRO_METHODS = {"HEC-HMS", "SWAT", "SCS", "SCS-CN", "HEC-RAS"}

    # Ambiguous acronym: SCS could be Soil Conservation Service or Spectral Correlation
    if "SCS" in accepted_method_names:
        scs_meta = next(
            (m.get("kb_metadata", {}) for m in methods if m.get("name","").upper() == "SCS"),
            {},
        )
        if scs_meta.get("disambiguated") != "hydrology":
            return True

    # Task conflicts with dominant methods
    if task_label == "flood_mapping_satellite" and _HYDRO_METHODS & accepted_method_names:
        return True

    # Standard checks
    if study_type.get("needs_judge") is True:                     return True
    if study_type.get("confidence", 1.0) < 0.85:                  return True
    if study_type.get("label") in {"global_algorithmic", "multi_site"}:
        if study_geo.get("primary_country") or study_geo.get("rivers"):
            return True
    if study_geo.get("confidence", 1.0) < 0.75:                   return True
    accepted_rivers = [r for r in study_geo.get("rivers", []) if r.get("accepted")]
    if accepted_rivers and study_geo.get("confidence", 1.0) < 0.9: return True
    if entities.get("dems"):                                        return True
    if entities.get("satellites") and entities.get("dems"):         return True
    if task.get("confidence", 1.0) < 0.85:                         return True
    return False

def judge_paper_with_ollama(paper: dict) -> dict:
    base_url   = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    model_name = os.getenv("OLLAMA_MODEL", "mistral-nemo:12b")
    if not is_ollama_available(base_url):
        return {"status": "skipped", "reason": f"Ollama not available at {base_url}"}
    sections  = ensure_dict(paper.get("sections"))
    metadata  = ensure_dict(paper.get("metadata"))
    entities  = ensure_dict(paper.get("entities"))
    candidate = {
        "paper_id":    metadata.get("paper_id"),
        "title":       metadata.get("title"),
        "study_type":  ensure_dict(ensure_dict(entities.get("geo")).get("study_type")),
        "study_geo":   ensure_dict(ensure_dict(entities.get("geo")).get("study_geo")),
        "task":        ensure_dict(entities.get("task")),
        "author_geo":  ensure_dict(entities.get("geo")).get("author_geo", []),
        "satellites":  entities.get("satellites", []),
        "dems":        entities.get("dems", []),
        "methods":     entities.get("methods", []),
    }
    judge = OllamaJudge(base_url=base_url, model=model_name, timeout=180)
    return judge.judge(candidate, sections)

def apply_judge_verdict(paper: dict, verdict: dict) -> dict:
    verdict  = ensure_dict(verdict)
    if verdict.get("status") in {"skipped", "failed"}:
        return paper
    entities  = paper.setdefault("entities", {})
    geo       = entities.setdefault("geo", {})
    geo["study_geo"] = ensure_dict(geo.get("study_geo"), default_study_geo())
    study_geo = geo["study_geo"]
    entities.setdefault("validation_warnings", [])

    st_verdict = ensure_dict(verdict.get("study_type"))
    if st_verdict:
        if not is_valid_judge_study_type_verdict(st_verdict):
            entities["validation_warnings"].append(
                {"type": "invalid_llm_study_type_verdict", "verdict": st_verdict}
            )
        elif not st_verdict.get("accepted", True):
            corrected = st_verdict.get("corrected_value")
            conf      = float(st_verdict.get("confidence", 0.7))
            if corrected and conf >= 0.8:
                geo["study_type"] = {
                    "label": corrected, "confidence": conf,
                    "source": "ollama_judge",
                    "previous": st_verdict.get("original_value"),
                    "reason": st_verdict.get("reason"),
                }

    c_verdict = ensure_dict(verdict.get("study_country"))
    if c_verdict:
        accepted      = c_verdict.get("accepted")
        original      = c_verdict.get("original_value")
        corrected     = c_verdict.get("corrected_value")
        conf          = float(c_verdict.get("confidence", 0.7))
        final_country = corrected if corrected else original
        if final_country and conf >= 0.8 and not study_geo.get("primary_country"):
            study_geo["primary_country"] = final_country
            study_geo["countries"]       = [{
                "name": final_country, "source": "ollama_judge",
                "confidence": conf, "reason": c_verdict.get("reason"),
            }]

    t_verdict = ensure_dict(verdict.get("task"))
    if t_verdict:
        if not is_valid_judge_task_verdict(t_verdict):
            entities["validation_warnings"].append(
                {"type": "invalid_llm_task_verdict", "verdict": t_verdict}
            )
        elif not t_verdict.get("accepted", True):
            corrected = t_verdict.get("corrected_value")
            conf      = float(t_verdict.get("confidence", 0.7))
            if corrected and conf >= 0.8:
                entities["task"] = {
                    "label": corrected, "confidence": conf,
                    "source": "ollama_judge",
                    "previous": t_verdict.get("original_value"),
                    "reason": t_verdict.get("reason"),
                }
    return paper

# ─────────────────────────────────────────────────────────────────────────────
# CONSTRAINT LAYER  (deterministic post-extraction rules)
# ─────────────────────────────────────────────────────────────────────────────

_HARD_HYDRO_METHODS = {"HEC-HMS", "SWAT", "HEC-RAS"}
_SCS_CN_SIGNALS     = {"SCS", "SCS-CN"}
# Satellite sensors that produce flood-extent maps via water-body extraction
_FLOOD_MAPPING_SATS = {"SAR", "Sentinel-1", "RADARSAT", "COSMO-SKYMED", "TERRASAR-X"}


def apply_constraints(paper: dict) -> dict:
    """
    Lightweight deterministic constraint layer applied after all extraction and scoring.

    Rules (in priority order):
    1. If HEC-HMS / SWAT accepted and used → task = hydrological_modeling
    2. If SCS-CN / SCS (hydro disambiguated) accepted → task = hydrological_modeling
    3. If DEM accepted alongside a hydrological model → DEM role = terrain_input
    4. If task = flood_mapping_satellite but no flood-extent satellite is accepted
       → downgrade to hydrological_modeling if hydro methods present
    """
    entities = paper.setdefault("entities", {})
    methods  = entities.get("methods", [])
    dems     = entities.get("dems", [])
    sats     = entities.get("satellites", [])
    task     = entities.get("task", {})

    accepted_methods = {
        m["name"].upper()
        for m in methods
        if m.get("accepted") and m.get("role") == "used"
    }
    scs_hydro = any(
        m.get("name","").upper() == "SCS"
        and m.get("kb_metadata", {}).get("disambiguated") == "hydrology"
        and m.get("accepted")
        for m in methods
    )

    has_hard_hydro = bool(_HARD_HYDRO_METHODS & accepted_methods)
    has_scs_cn     = bool(_SCS_CN_SIGNALS & accepted_methods) or scs_hydro

    # Rule 1 + 2: force task to hydrological_modeling
    if (has_hard_hydro or has_scs_cn) and task.get("label") != "hydrological_modeling":
        entities["task"] = make_task("hydrological_modeling", 0.95, source="constraint")
        task = entities["task"]

    # Rule 3: DEM used alongside hydrological model → terrain_input role
    if has_hard_hydro:
        for dem in dems:
            if dem.get("accepted") and dem.get("role") in {"used", "mentioned", None}:
                dem["role"] = "terrain_input"

    # Rule 4: flood_mapping_satellite requires a real flood-mapping satellite
    if task.get("label") == "flood_mapping_satellite":
        accepted_sat_names = {s["name"] for s in sats if s.get("accepted")}
        has_flood_sat = bool(_FLOOD_MAPPING_SATS & accepted_sat_names)
        if not has_flood_sat and (has_hard_hydro or has_scs_cn):
            entities["task"] = make_task("hydrological_modeling", 0.88, source="constraint")

    return paper


# ─────────────────────────────────────────────────────────────────────────────
# MAIN BUILD
# ─────────────────────────────────────────────────────────────────────────────

def build_paper_json(xml_path: Path) -> dict:
    tree     = etree.parse(str(xml_path))
    root     = tree.getroot()
    metadata = parse_metadata(root, xml_path)
    sections = parse_sections(root)
    ctx      = PipelineContext(sections)
    title    = metadata.get("title", "")

    entities = extract_entities(root, ctx, metadata, title=title)
    kb       = _get_kb()
    extractor = EntityExtractor(kb)
    entities["sensor_types"] = extractor.infer_sensor_types(
        entities.get("satellites", []),
        entities.get("dems", []),
    )
    entities["task"] = classify_task(ctx)

    paper = {
        "metadata": {
            **metadata,
            "content_hash": make_hash(ctx.full_text),
        },
        "sections":  sections,
        "entities":  entities,
        "llm_judge": None,
        "provenance": {
            "parser":      "grobid_tei_kb_v2",
            "source_xml":  str(xml_path),
            "has_geo":     bool(
                entities.get("geo", {})
                .get("study_geo", {})
                .get("primary_country")
            ),
            "judge_used":  False,
            "judge_model": None,
        },
    }

    # Constraint layer: deterministic post-extraction rules
    paper = apply_constraints(paper)

    # Validation layer: LLM judge for ambiguous cases
    if needs_judge(paper):
        verdict = judge_paper_with_ollama(paper)
        paper   = apply_judge_verdict(paper, verdict)
        if verdict.get("status") not in {"skipped", "failed", None}:
            paper["llm_judge"]                = verdict
            paper["provenance"]["judge_used"] = True
            paper["provenance"]["judge_model"] = os.getenv(
                "OLLAMA_MODEL", "mistral-nemo:12b"
            )

    return json_safe(paper)


def run():
    xml_files = sorted(XML_DIR.glob("*.tei.xml"))
    print(f"Found {len(xml_files)} XML files")

    for xml_file in xml_files:
        try:
            print(f"Building JSON: {xml_file.name}")
            paper    = build_paper_json(xml_file)
            out_path = OUT_DIR / f"{xml_file.stem}.paper.json"
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(json_safe(paper), f, indent=2, ensure_ascii=False)
            print(f"Saved: {out_path.name}")
        except Exception as e:
            print(f"Error in {xml_file.name}: {type(e).__name__}: {e}")
            traceback.print_exc()


if __name__ == "__main__":
    run()
