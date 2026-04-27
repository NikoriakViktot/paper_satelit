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
    You are a scientific information extraction assistant.
    You read passages from flood-mapping research papers and extract structured data.
    Always reply with valid JSON only — no prose, no markdown fences.
    Use null for fields that cannot be determined from the text.
    Numeric accuracy values must be on a 0–1 scale (e.g. 0.95, not 95).
""")

_USER_TEMPLATE = textwrap.dedent("""\
    Extract the following fields from the text below.
    Return ONLY a JSON object with these exact keys:
      author, method, sensor, region, oa, f1, iou, kappa, accuracy_desc

    Rules:
    - oa / f1 / iou / kappa: float 0–1, or null
    - If a range is given (e.g. 0.84–0.95), return the mean
    - author: first author surname and year if visible, else null
    - method: short label (e.g. "U-Net", "Random Forest", "Thresholding")
    - sensor: sensor names (e.g. "Sentinel-1", "Landsat-8")
    - region: study area name(s)
    - accuracy_desc: one-sentence summary of accuracy if numeric is absent

    TEXT:
    ---
    {text}
    ---
""")

_FALLBACK = RegexExtractor()


class OllamaExtractor(BaseExtractor):
    """
    LLM-backed extractor using Ollama's /api/generate endpoint.
    Falls back to RegexExtractor on any error.
    """

    def __init__(
        self,
        base_url: str = OLLAMA_BASE_URL,
        model: str = OLLAMA_MODEL,
        timeout: int = OLLAMA_TIMEOUT,
        context_chars: int = 6000,
    ) -> None:
        self._url     = f"{base_url.rstrip('/')}/api/generate"
        self._model   = model
        self._timeout = timeout
        self._ctx     = context_chars
        self._available: bool | None = None

    # ── public ────────────────────────────────────────────────────────────────

    def extract(
        self,
        chunks: list[dict],
        source_file: str,
    ) -> ExtractionResult:
        if not self._is_available():
            logger.warning(
                "Ollama not reachable at %s — falling back to RegexExtractor",
                self._url,
            )
            return _FALLBACK.extract(chunks, source_file)

        combined = "\n\n".join(c["text"] for c in chunks)
        text_ctx = combined[: self._ctx]

        try:
            raw_json = self._call_ollama(text_ctx)
            result   = self._parse_response(raw_json, source_file)
        except Exception as exc:
            logger.warning("Ollama extraction failed (%s) — falling back", exc)
            result = _FALLBACK.extract(chunks, source_file)

        # regex fills any fields the LLM missed
        result = _merge_with_regex(result, chunks, source_file)
        return result.finalize()

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

    def _call_ollama(self, text: str) -> str:
        payload = {
            "model":  self._model,
            "prompt": _USER_TEMPLATE.format(text=text),
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
        # strip markdown fences if model still emits them
        raw = re.sub(r"```(?:json)?", "", raw).strip()

        data: dict[str, Any] = json.loads(raw)

        def _f(key: str) -> float | None:
            val = data.get(key)
            if val is None:
                return None
            try:
                num = float(val)
                return round(num / 100.0 if num > 1.0 else num, 4)
            except (TypeError, ValueError):
                return None

        return ExtractionResult(
            source_file   = source_file,
            author        = str(data.get("author") or ""),
            method        = str(data.get("method") or ""),
            sensor        = str(data.get("sensor") or ""),
            region        = str(data.get("region") or ""),
            oa            = _f("oa"),
            f1            = _f("f1"),
            iou           = _f("iou"),
            kappa         = _f("kappa"),
            accuracy_desc = str(data.get("accuracy_desc") or ""),
            confidence    = 0.85,      # LLM output scored higher
        )


# ── merge helper ──────────────────────────────────────────────────────────────

def _merge_with_regex(
    llm_result: ExtractionResult,
    chunks: list[dict],
    source_file: str,
) -> ExtractionResult:
    """Fill missing LLM fields with regex results."""
    regex = _FALLBACK.extract(chunks, source_file)

    if not llm_result.oa     and regex.oa:     llm_result.oa     = regex.oa
    if not llm_result.f1     and regex.f1:     llm_result.f1     = regex.f1
    if not llm_result.iou    and regex.iou:    llm_result.iou    = regex.iou
    if not llm_result.kappa  and regex.kappa:  llm_result.kappa  = regex.kappa
    if not llm_result.method and regex.method: llm_result.method = regex.method
    if not llm_result.sensor and regex.sensor: llm_result.sensor = regex.sensor
    if not llm_result.region and regex.region: llm_result.region = regex.region
    if not llm_result.author and regex.author: llm_result.author = regex.author

    llm_result.evidence = regex.evidence
    return llm_result
