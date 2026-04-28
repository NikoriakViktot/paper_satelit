"""
Section-aware LLM extractor for satellite flood mapping literature.

Architecture:
    PDF chunks (page-ordered) → split_sections → 3 targeted LLM calls → merge

Three LLM calls per paper:
    1. Abstract   → study_type, satellite_names, sensor_type, data_product,
                    country, region, river_basin, city_event, methods,
                    near_real_time, latency
    2. Methods    → methods, satellite_names, sensor_type, data_product (override)
    3. Results    → OA, F1, IoU, Kappa (validated: number must appear in text)

Merge priority: Results > Methods > Abstract > regex fallback
"""
from __future__ import annotations

import json
import logging
import re
import textwrap
from typing import Any

import requests

from src.config import OLLAMA_BASE_URL, OLLAMA_MODEL, OLLAMA_TIMEOUT
from src.extraction.base import BaseExtractor, ExtractionResult
from src.extraction.regex_extractor import RegexExtractor
from src.processing.section_parser import describe_sections, split_sections

logger = logging.getLogger(__name__)

_FALLBACK = RegexExtractor()

# ── System prompt ─────────────────────────────────────────────────────────────

_SYS = textwrap.dedent("""\
    You are a scientific information extraction system for satellite-based flood mapping literature.
    Reply with valid JSON only — no prose, no markdown.
    Numeric accuracy values must be 0–1 scale (e.g. 0.95 not 95).
    Extract ONLY what is explicitly in the text. Never hallucinate.
""")

# ── Per-section prompts ───────────────────────────────────────────────────────

_ABSTRACT_PROMPT = textwrap.dedent("""\
    Extract from this paper ABSTRACT. Return JSON with exactly these keys:
      Study_Type, Satellite_Names, Sensor_Type, Data_Product,
      Country, Region, River_Basin, City_Event, Methods, Near_Real_Time, Latency

    Rules:
    - Study_Type: one of "Satellite flood mapping" | "ML/DL classification" |
      "Hydrological forecasting" | "Hydraulic modeling" |
      "Operational mapping system" | "Review paper" | "Dataset/benchmark paper"
    - Satellite_Names: comma-separated list (e.g. "Sentinel-1, Sentinel-2"); null if absent
    - Sensor_Type: "SAR" | "Optical" | "Multi-sensor"; null if absent
    - Data_Product: e.g. "GRD", "MSI", "OLI"; null if not mentioned
    - Country: country name (prefer country over region); null if absent
    - Region: sub-national region, continent, or general area; null if absent
    - River_Basin: river or basin name; null if absent
    - City_Event: city name or specific flood event; null if absent
    - Methods: comma-separated list from:
        Thresholding | Change detection | NDWI/MNDWI | Random Forest | SVM |
        Maximum likelihood | U-Net | CNN | LSTM | Transformer | OBIA |
        Hydrodynamic model | Operational workflow
      null if absent
    - Near_Real_Time: true | false | null
    - Latency: string like "<6 h" or "1–3 days"; null if absent

    ABSTRACT:
    {text}
""")

_METHODS_PROMPT = textwrap.dedent("""\
    Extract from this METHODS section. Return JSON with exactly these keys:
      Methods, Satellite_Names, Sensor_Type, Data_Product

    Rules:
    - Methods: comma-separated list of all processing approaches found:
        Thresholding | Change detection | NDWI/MNDWI | Random Forest | SVM |
        Maximum likelihood | U-Net | CNN | LSTM | Transformer | OBIA |
        Hydrodynamic model | Operational workflow
    - Satellite_Names: comma-separated list of satellite/sensor names; null if absent
    - Sensor_Type: "SAR" | "Optical" | "Multi-sensor"; null if absent
    - Data_Product: e.g. "GRD", "MSI", "OLI"; null if absent

    METHODS:
    {text}
""")

_RESULTS_PROMPT = textwrap.dedent("""\
    Extract ONLY explicit numeric accuracy metrics from this RESULTS section.
    Return JSON with keys: OA, F1, IoU, Kappa

    Rules:
    - OA / F1 / IoU / Kappa: float 0–1 only; range (0.84–0.95) → mean
    - Return null if the number is NOT explicitly stated in the text
    - NEVER guess or infer; "high accuracy" → null

    RESULTS:
    {text}
""")


# ── Metric validation ─────────────────────────────────────────────────────────

def _number_in_text(val: float, text: str) -> bool:
    candidates = [
        f"{val:.4f}", f"{val:.3f}", f"{val:.2f}",
        f"{val * 100:.2f}", f"{val * 100:.1f}",
        f"{val * 100:.0f}",
    ]
    for raw in candidates:
        s = raw.rstrip("0").rstrip(".")
        if not s:
            continue
        if re.search(r"(?<!\d)" + re.escape(s) + r"(?!\d)", text):
            return True
    return False


def _validate_metrics(result: ExtractionResult, text: str) -> ExtractionResult:
    for attr in ("oa", "f1", "iou", "kappa"):
        val = getattr(result, attr)
        if val is not None and not _number_in_text(val, text):
            logger.debug("Discarded hallucinated %s=%s (not in text)", attr, val)
            setattr(result, attr, None)
    return result


# ── SectionExtractor ──────────────────────────────────────────────────────────

class SectionExtractor(BaseExtractor):
    """
    Section-aware extractor for flood mapping papers.
    Falls back to RegexExtractor when Ollama is unreachable.
    """

    def __init__(
        self,
        base_url: str = OLLAMA_BASE_URL,
        model:    str = OLLAMA_MODEL,
        timeout:  int = OLLAMA_TIMEOUT,
        abstract_max: int = 1500,
        methods_max:  int = 3000,
        results_max:  int = 2000,
    ) -> None:
        self._url          = f"{base_url.rstrip('/')}/api/generate"
        self._model        = model
        self._timeout      = timeout
        self._abstract_max = abstract_max
        self._methods_max  = methods_max
        self._results_max  = results_max
        self._available: bool | None = None

    # ── BaseExtractor interface ───────────────────────────────────────────────

    def extract(
        self,
        chunks: list[dict],
        source_file: str,
    ) -> ExtractionResult:
        ordered  = sorted(chunks, key=lambda c: (c.get("page_start", 0), c.get("chunk_id", "")))
        full_text = "\n\n".join(c["text"] for c in ordered)

        regex_result = _FALLBACK.extract(chunks, source_file)

        sections = split_sections(full_text)
        found, missing = describe_sections(sections)
        self._debug_sections(source_file, found, missing)

        if not self._is_available():
            logger.warning("Ollama unavailable — regex fallback for %s", source_file)
            return regex_result

        result = ExtractionResult(source_file=source_file)
        sections_used: list[str] = []

        # Call 1: Abstract → full thematic profile
        abstract_text = sections.get("abstract") or regex_result.abstract or ""
        if abstract_text:
            abs_r = self._call_section(
                _ABSTRACT_PROMPT, abstract_text[:self._abstract_max], source_file
            )
            _merge(result, abs_r, (
                "study_type", "satellite_names", "sensor_type", "data_product",
                "country", "region", "river_basin", "city_event",
                "methods", "near_real_time", "latency",
            ))
            sections_used.append("abstract")

        # Call 2: Methods → precise methods + sensor (override)
        methods_text = sections.get("methods") or sections.get("data") or ""
        if methods_text:
            meth_r = self._call_section(
                _METHODS_PROMPT, methods_text[:self._methods_max], source_file
            )
            _merge(result, meth_r,
                   ("methods", "satellite_names", "sensor_type", "data_product"),
                   override=True)
            sections_used.append("methods")

        # Call 3: Results → metrics (only for classification/mapping papers)
        results_text = sections.get("results") or ""
        if results_text and result.study_type in (
            "ML/DL classification", "Satellite flood mapping", ""
        ):
            res_r = self._call_section(
                _RESULTS_PROMPT, results_text[:self._results_max], source_file
            )
            res_r = _validate_metrics(res_r, results_text)
            _merge(result, res_r, ("oa", "f1", "iou", "kappa"))
            sections_used.append("results")

        # Regex fills remaining gaps
        _merge(result, regex_result, (
            "oa", "f1", "iou", "kappa",
            "study_type", "satellite_names", "sensor_type", "data_product",
            "country", "river_basin", "city_event", "methods",
            "near_real_time", "latency", "revisit_time",
        ))

        # Bibliographic fields always from regex
        result.title     = regex_result.title
        result.abstract  = regex_result.abstract
        result.doi       = regex_result.doi
        result.year      = regex_result.year
        result.authors   = regex_result.authors
        result.full_text = regex_result.full_text
        result.evidence  = regex_result.evidence

        # Geography fields from keyword scan — always use regex (reliable)
        result.ukraine_relevance = regex_result.ukraine_relevance
        result.river_name        = regex_result.river_name or result.river_basin

        result.sections_used = sections_used
        result.confidence    = 0.90

        return result.finalize()

    # ── LLM helpers ───────────────────────────────────────────────────────────

    def _is_available(self) -> bool:
        if self._available is not None:
            return self._available
        try:
            resp = requests.get(
                self._url.replace("/api/generate", "/api/tags"),
                timeout=3,
            )
            self._available = resp.status_code == 200
        except requests.RequestException:
            self._available = False
        if self._available:
            logger.info("Ollama available at %s (model: %s)", self._url, self._model)
        return self._available

    def _call_section(
        self,
        prompt_template: str,
        text: str,
        source_file: str,
    ) -> ExtractionResult:
        if not text.strip():
            return ExtractionResult(source_file=source_file)

        payload = {
            "model":  self._model,
            "prompt": prompt_template.format(text=text),
            "system": _SYS,
            "stream": False,
            "format": "json",
            "options": {"temperature": 0.0, "num_predict": 400},
        }
        try:
            resp = requests.post(self._url, json=payload, timeout=self._timeout)
            resp.raise_for_status()
            raw = resp.json().get("response", "")
            return self._parse(raw, source_file)
        except Exception as exc:
            logger.warning("LLM call failed for %s: %s", source_file, exc)
            return ExtractionResult(source_file=source_file)

    def _parse(self, raw: str, source_file: str) -> ExtractionResult:
        raw = re.sub(r"```(?:json)?", "", raw).strip()
        try:
            data: dict[str, Any] = json.loads(raw)
        except json.JSONDecodeError:
            return ExtractionResult(source_file=source_file)

        def _s(key: str) -> str:
            val = data.get(key) or data.get(key.lower())
            return str(val).strip() if val and val != "null" else ""

        def _f(key: str) -> float | None:
            val = data.get(key) or data.get(key.lower())
            if val is None or val == "null":
                return None
            try:
                n = float(val)
                return round(n / 100.0 if n > 1.0 else n, 4)
            except (TypeError, ValueError):
                return None

        def _b(key: str) -> bool | None:
            val = data.get(key) or data.get(key.lower())
            if val is None or val == "null":
                return None
            if isinstance(val, bool):
                return val
            s = str(val).lower()
            if s in ("true", "yes", "1"):
                return True
            if s in ("false", "no", "0"):
                return False
            return None

        return ExtractionResult(
            source_file     = source_file,
            study_type      = _s("Study_Type"),
            satellite_names = _s("Satellite_Names"),
            sensor_type     = _s("Sensor_Type"),
            data_product    = _s("Data_Product"),
            country         = _s("Country"),
            region          = _s("Region"),
            river_basin     = _s("River_Basin"),
            city_event      = _s("City_Event"),
            methods         = _s("Methods"),
            near_real_time  = _b("Near_Real_Time"),
            latency         = _s("Latency"),
            oa              = _f("OA"),
            f1              = _f("F1"),
            iou             = _f("IoU"),
            kappa           = _f("Kappa"),
        )

    # ── Debug ─────────────────────────────────────────────────────────────────

    @staticmethod
    def _debug_sections(
        source_file: str,
        found: list[str],
        missing: list[str],
    ) -> None:
        logger.info("[%s]  found=%s  missing=%s", source_file, found, missing)
        print(f"\n  [{source_file}]")
        print(f"    Sections found:   {found or '—'}")
        print(f"    Missing sections: {missing or '—'}")


# ── Merge helper ──────────────────────────────────────────────────────────────

def _merge(
    target: ExtractionResult,
    source: ExtractionResult,
    fields: tuple[str, ...],
    override: bool = False,
) -> None:
    for f in fields:
        src_val = getattr(source, f, None)
        if src_val is None or src_val == "" or src_val is False:
            continue
        tgt_val = getattr(target, f, None)
        if override or not tgt_val:
            setattr(target, f, src_val)
