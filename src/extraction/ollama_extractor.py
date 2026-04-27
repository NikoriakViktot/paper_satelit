"""
LLM-based extractor using a locally running Ollama instance.

Falls back gracefully to the RegexExtractor when Ollama is unreachable
or returns unparseable output.

Usage (requires Ollama running):
    $ ollama pull llama3.2
    $ ollama serve          # already running as a service after install
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

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = textwrap.dedent("""\
    You are a scientific data completion and interpretation system for flood mapping literature.
    Your role: extract what is present, explain what is missing — never invent values.
    Always reply with valid JSON only — no prose, no markdown fences.
    Numeric accuracy values must be on a 0–1 scale (e.g. 0.95, not 95).
    NEVER hallucinate methods, sensors, regions, or numeric metrics.
""")

_USER_TEMPLATE = textwrap.dedent("""\
    Review the paper below and return a JSON object with these exact keys:
      Method, Sensor, Region, OA, F1, IoU, Accuracy_Description, Missing_Data_Explanation

    INPUT:
    TITLE: {title}
    ABSTRACT: {abstract}

    FIELD RULES:

    Method:
    - Extract specific model name if present: U-Net, Random Forest, CNN, SVM, etc.
    - If only a generic phrase: return it (e.g. "deep learning")
    - If absent: return exactly "Method not explicitly specified in abstract"

    Sensor:
    - Infer from keywords: SAR / Sentinel-1 → "SAR"; Sentinel-2 / Landsat → "Optical"
    - Both present → "Multi"
    - If absent: return exactly "Sensor type not reported"

    Region:
    - Priority: country > sub-region > basin/river
    - If Ukraine mentioned anywhere → return exactly "Ukraine"
    - If absent: return exactly "Study area not specified"

    OA / F1 / IoU:
    - Float 0–1, or null
    - Range (0.84–0.95) → return mean
    - Extract ONLY if an explicit number is present — NEVER guess
    - "high accuracy" → null (not numeric)

    Accuracy_Description:
    - If numeric metrics present → null
    - If no metrics but qualitative phrases exist → extract them verbatim
      (e.g. "high accuracy", "validated against ground truth")
    - If neither → return exactly "Study reports qualitative validation without explicit quantitative metrics"

    Missing_Data_Explanation:
    - Concise scientific sentence listing which fields are absent and why
    - Example: "OA, F1, IoU not reported; method described generically; study area inferred from context."
    - If nothing is missing → return null

    Return valid JSON only.
""")

_FALLBACK = RegexExtractor()

# Known explanation-string prefixes the LLM is instructed to return
_EXPLANATION_MARKERS = (
    "method not explicitly",
    "sensor type not",
    "study area not",
    "study reports qualitative",
    "no information",
    "not reported",
    "not specified",
    "not explicitly",
    "not determinable",
)


def _is_explanation(text: str) -> bool:
    """Return True when the LLM returned an explanation rather than a data value."""
    if not text:
        return False
    lower = text.lower()
    return any(marker in lower for marker in _EXPLANATION_MARKERS) or len(text) > 80


class OllamaExtractor(BaseExtractor):
    """
    LLM-backed extractor using Ollama's /api/generate endpoint.
    Input is title + abstract (not full text) for focused, reliable extraction.
    Falls back to RegexExtractor on any error.
    """

    def __init__(
        self,
        base_url: str = OLLAMA_BASE_URL,
        model: str = OLLAMA_MODEL,
        timeout: int = OLLAMA_TIMEOUT,
    ) -> None:
        self._url     = f"{base_url.rstrip('/')}/api/generate"
        self._model   = model
        self._timeout = timeout
        self._available: bool | None = None

    # ── public ────────────────────────────────────────────────────────────────

    def extract(
        self,
        chunks: list[dict],
        source_file: str,
    ) -> ExtractionResult:
        # always run regex first: supplies title/abstract for LLM and fills gaps
        regex_result = _FALLBACK.extract(chunks, source_file)

        if not self._is_available():
            logger.warning(
                "Ollama not reachable at %s — falling back to RegexExtractor",
                self._url,
            )
            return regex_result

        try:
            raw_json = self._call_ollama(regex_result.title, regex_result.abstract)
            result   = self._parse_response(raw_json, source_file)
        except Exception as exc:
            logger.warning("Ollama extraction failed (%s) — falling back to regex", exc)
            return regex_result

        return _fill_from_regex(result, regex_result).finalize()

    # ── private ───────────────────────────────────────────────────────────────

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
            logger.info("Ollama is available at %s (model: %s)", self._url, self._model)
        return self._available

    def _call_ollama(self, title: str, abstract: str) -> str:
        prompt = _USER_TEMPLATE.format(
            title=title or "(not available)",
            abstract=abstract or "(not available)",
        )
        payload = {
            "model":  self._model,
            "prompt": prompt,
            "system": _SYSTEM_PROMPT,
            "stream": False,
            "format": "json",
            "options": {"temperature": 0.0, "num_predict": 400},
        }
        resp = requests.post(self._url, json=payload, timeout=self._timeout)
        resp.raise_for_status()
        return resp.json().get("response", "")

    def _parse_response(
        self,
        raw: str,
        source_file: str,
    ) -> ExtractionResult:
        raw = re.sub(r"```(?:json)?", "", raw).strip()
        data: dict[str, Any] = json.loads(raw)

        def _raw_str(key: str) -> str:
            val = data.get(key) or data.get(key.lower())
            return str(val).strip() if val else ""

        def _float(key: str) -> float | None:
            val = data.get(key) or data.get(key.lower())
            if val is None:
                return None
            try:
                num = float(val)
                return round(num / 100.0 if num > 1.0 else num, 4)
            except (TypeError, ValueError):
                return None

        method_raw  = _raw_str("Method")
        sensor_raw  = _raw_str("Sensor")
        region_raw  = _raw_str("Region")
        acc_raw     = _raw_str("Accuracy_Description")
        mde_raw     = _raw_str("Missing_Data_Explanation")

        # Separate real values from explanation strings.
        # Explanation strings are routed to missing_data_explanation so that
        # the categorical fields stay clean for downstream normalisation.
        method, sensor, region = (
            ("" if _is_explanation(method_raw) else method_raw),
            ("" if _is_explanation(sensor_raw) else sensor_raw),
            ("" if _is_explanation(region_raw) else region_raw),
        )

        # Collect per-field explanations that were rerouted
        rerouted = [
            v for v in (
                method_raw if _is_explanation(method_raw) else "",
                sensor_raw if _is_explanation(sensor_raw) else "",
                region_raw if _is_explanation(region_raw) else "",
            ) if v
        ]
        if rerouted and not mde_raw:
            mde_raw = " | ".join(rerouted)
        elif rerouted:
            mde_raw = mde_raw + " | " + " | ".join(rerouted)

        # Standard accuracy fallback when no description was generated
        if not acc_raw and not any([_float("OA"), _float("F1"), _float("IoU")]):
            acc_raw = "Study reports qualitative validation without explicit quantitative metrics"

        return ExtractionResult(
            source_file              = source_file,
            method                   = method,
            sensor                   = sensor,
            region                   = region,
            oa                       = _float("OA"),
            f1                       = _float("F1"),
            iou                      = _float("IoU"),
            accuracy_desc            = acc_raw,
            missing_data_explanation = mde_raw,
            confidence               = 0.85,
        )


# ── merge helper ──────────────────────────────────────────────────────────────

def _fill_from_regex(
    llm: ExtractionResult,
    regex: ExtractionResult,
) -> ExtractionResult:
    """Fill any blank LLM fields with values from the regex pass."""
    if not llm.oa     and regex.oa:     llm.oa     = regex.oa
    if not llm.f1     and regex.f1:     llm.f1     = regex.f1
    if not llm.iou    and regex.iou:    llm.iou    = regex.iou
    if not llm.kappa  and regex.kappa:  llm.kappa  = regex.kappa
    if not llm.method and regex.method: llm.method = regex.method
    if not llm.sensor and regex.sensor: llm.sensor = regex.sensor
    if not llm.region and regex.region: llm.region = regex.region
    if not llm.author and regex.author: llm.author = regex.author

    # bibliographic fields come exclusively from regex
    llm.title     = regex.title
    llm.abstract  = regex.abstract
    llm.doi       = regex.doi
    llm.full_text = regex.full_text
    llm.evidence  = regex.evidence

    # preserve any explanation the LLM produced; don't overwrite with regex blank
    if not llm.missing_data_explanation and regex.missing_data_explanation:
        llm.missing_data_explanation = regex.missing_data_explanation

    return llm
