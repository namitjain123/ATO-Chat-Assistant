"""
Rich per-chunk metadata, stored in each Qdrant point's payload and used for
query-time filtering (see qdrant_service._build_filter).

Everything here is deterministic and derived from the document itself,
except `topics`, which the contextualizer LLM assigns alongside its context
note (app/ingestion/contextualizer.py).
"""
import re

# The one piece of the pipeline tied to the ATO corpus — swap it with the
# knowledge base. The planner offers exactly these values as query filters.
TOPICS = (
    "deductions",
    "income",
    "tax_offsets",
    "tax_rates",
    "record_keeping",
    "lodgment",
    "super_and_investments",
    "business",
)

_HEADING_RE = re.compile(r"^#{1,6}\s+(.+?)\s*$", re.MULTILINE)
_MD_LINK_RE = re.compile(r"\[([^\]]*)\]\([^)]*\)")
# En dash / em dash / non-breaking hyphen / hyphen — ATO pages use all of them.
_INCOME_YEAR_RE = re.compile(r"\b(20\d{2})\s*[–—‑-]\s*(20\d{2}|\d{2})\b")


def _clean(text: str) -> str:
    text = _MD_LINK_RE.sub(r"\1", text)
    return re.sub(r"\s+", " ", text.replace("*", "")).strip()


def title_from_filename(filename: str) -> str:
    """'gifts-and-donations.md' -> 'Gifts and donations'. Crawled pages have no
    H1 (the pruning filter drops it), so the slug is the reliable title."""
    words = [w for w in filename.rsplit(".", 1)[0].replace("_", "-").split("-") if w]
    if len(words) > 1 and words[-1].isdigit():  # crawl.py's duplicate-slug suffix
        words = words[:-1]
    title = " ".join(words)
    return title[:1].upper() + title[1:]


def page_summary(document: str) -> str:
    """ATO pages open with a one-line description of the page — first line
    that isn't a heading or a list/link item."""
    for line in document.splitlines():
        stripped = line.strip()
        if stripped and not stripped.startswith(("#", "*", "-", "[")):
            return _clean(stripped)[:300]
    return ""


def heading_positions(document: str) -> list[tuple[int, str]]:
    return [(m.start(), _clean(m.group(1))) for m in _HEADING_RE.finditer(document)]


def section_at(headings: list[tuple[int, str]], position: int) -> str:
    """Nearest heading at or before `position`; '' if none (or position < 0)."""
    current = ""
    if position < 0:
        return current
    for pos, text in headings:
        if pos > position:
            break
        current = text
    return current


def locate_chunks(document: str, chunks: list[str]) -> list[int]:
    """Start offset of each chunk in `document` (-1 if not found). Searches
    forward from the previous match, since chunks arrive in document order
    and short repeated phrases would otherwise all resolve to the first hit."""
    positions, cursor = [], 0
    for chunk in chunks:
        probe = chunk[:80]
        pos = document.find(probe, cursor)
        if pos == -1:
            pos = document.find(probe)
        positions.append(pos)
        if pos != -1:
            cursor = pos + 1  # past this match's start, or an identical next chunk re-finds it
    return positions


def income_years(text: str) -> list[str]:
    """Financial years mentioned, normalized to 'YYYY-YY' ('2024–25' and
    '2024-2025' both -> '2024-25'). Non-consecutive ranges ('2021-2025') are
    date spans, not an income year, and are ignored."""
    years = set()
    for start, end in _INCOME_YEAR_RE.findall(text):
        if (int(start) + 1) % 100 == int(end[-2:]):
            years.add(f"{start}-{end[-2:]}")
    return sorted(years)


def normalize_topics(topics) -> list[str]:
    """Keep only known TOPICS (the LLM occasionally invents or re-spells one)."""
    result = []
    for topic in topics or []:
        key = str(topic).strip().lower().replace(" ", "_").replace("-", "_")
        if key in TOPICS and key not in result:
            result.append(key)
    return result
