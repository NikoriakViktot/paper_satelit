"""
Section-aware parser for scientific PDF text.

Detects standard paper sections (Abstract, Introduction, Methods, Results,
Discussion, Conclusion) via regex on heading lines and splits the
concatenated document text into a keyed dict.
"""
from __future__ import annotations

import re

# ── Section heading regex ─────────────────────────────────────────────────────
#
# Matches a line that IS a section heading (after stripping leading
# section numbers).  Named groups identify the canonical section.

_HEADING_RE = re.compile(
    r"^\s*"
    r"(?:\d+(?:\.\d+)*\.?\s+)?"          # optional: "1.", "2.1.", "3.1.2 "
    r"(?:"
    r"(?P<abstract>abstract)"
    r"|(?P<introduction>introduction|background(?:\s+and\s+motivation)?)"
    r"|(?P<methods>"
        r"methods?|methodology"
        r"|materials\s+(?:and|&)\s+methods?"
        r"|experimental\s+(?:setup|design|methods?|procedures?)"
        r"|study\s+area(?:\s+and\s+methods?)?"
        r"|data\s+and\s+methods?"
        r"|proposed\s+method"
    r")"
    r"|(?P<data>"
        r"data(?:set)?(?:\s+(?:description|preparation|acquisition))?"
        r"|study\s+area"
        r"|research\s+area"
        r"|remote\s+sensing\s+data"
    r")"
    r"|(?P<results>"
        r"results?(?:\s+and\s+discussion)?"
        r"|experimental\s+results?"
        r"|accuracy\s+assessment"
        r"|performance\s+evaluation"
        r"|evaluation"
    r")"
    r"|(?P<discussion>"
        r"discussion(?:\s+and\s+(?:conclusion|analysis))?"
    r")"
    r"|(?P<conclusion>"
        r"conclusions?"
        r"|concluding\s+remarks?"
        r"|summary(?:\s+and\s+conclusion)?"
        r"|final\s+remarks?"
    r")"
    r")"
    r"\s*$",
    re.IGNORECASE,
)

# Canonical order — used to detect backward heading jumps (duplicate hits)
_SECTION_ORDER = [
    "abstract", "introduction", "methods", "data",
    "results", "discussion", "conclusion",
]


def split_sections(text: str) -> dict[str, str | None]:
    """
    Split *text* into a dict keyed by canonical section name.

    Returns
    -------
    {
        "abstract":     str | None,
        "introduction": str | None,
        "methods":      str | None,
        "data":         str | None,
        "results":      str | None,
        "discussion":   str | None,
        "conclusion":   str | None,
    }
    Lines that look like section headings (short, not ending in sentence
    punctuation) are used as split points.  Each section captures text up to
    the next recognised heading.
    """
    lines = text.splitlines()
    markers: list[tuple[int, str]] = []  # (line_index, canonical_section)

    for i, line in enumerate(lines):
        stripped = line.strip()
        if not stripped or len(stripped) > 100:
            continue
        # Reject lines ending with sentence-terminating punctuation
        if stripped.endswith((".", ",", ";", ":", "?", "—", "-")):
            continue
        m = _HEADING_RE.match(stripped)
        if not m:
            continue
        section = next(
            (name for name in _SECTION_ORDER if m.group(name)),
            None,
        )
        if section and (not markers or markers[-1][1] != section):
            markers.append((i, section))

    # Build result dict — default all to None
    result: dict[str, str | None] = {s: None for s in _SECTION_ORDER}

    for idx, (line_start, section) in enumerate(markers):
        line_end = markers[idx + 1][0] if idx + 1 < len(markers) else len(lines)
        body = "\n".join(lines[line_start + 1: line_end]).strip()
        if body and result[section] is None:
            result[section] = body

    return result


def describe_sections(sections: dict[str, str | None]) -> tuple[list[str], list[str]]:
    """Return (found, missing) section name lists."""
    found   = [k for k, v in sections.items() if v]
    missing = [k for k, v in sections.items() if not v]
    return found, missing
