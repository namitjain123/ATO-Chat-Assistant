import asyncio
import re
import sys
from pathlib import Path
from urllib.parse import urlparse

from docx import Document
from crawl4ai import AsyncWebCrawler, CrawlerRunConfig
from crawl4ai.content_filter_strategy import PruningContentFilter
from crawl4ai.markdown_generation_strategy import DefaultMarkdownGenerator
from crawl4ai.deep_crawling import (
    BFSDeepCrawlStrategy,
    FilterChain,
    DomainFilter,
    ContentTypeFilter,
)

sys.stdout.reconfigure(encoding="utf-8")

# Start from a content-rich section (not the homepage, which is mostly links).
START_URL = "https://www.ato.gov.au/individuals-and-families/income-deductions-offsets-and-records/deductions-you-can-claim"
OUTPUT_DIR = Path(__file__).parent / "DATA" / "ato_deductions"

MAX_PAGES = 25          # cap total pages crawled
MAX_DEPTH = 2           # how many link-hops from START_URL to follow
MIN_CONTENT_CHARS = 800  # skip thin nav/hub pages with little real text


# PruningContentFilter's density heuristic misses this site-wide chrome on a
# few longer pages (verified: 3/25 pages still carried it after pruning). Both
# patterns are exact ATO template boilerplate, not article content, so a
# direct strip is safe: "### Tools" reliably marks the start of the footer nav
# block (Tools / Tax information for / Help and support / copyright), and the
# "Log in to ATO online services" widget duplicates itself at the top of a
# couple of pages.
FOOTER_MARKER = "### Tools"
LOGIN_BLOCK_RE = re.compile(
    r"(?:Log in to online services ?Log in\s*\n)?"
    r"## Log in to ATO online services\n"
    r"(?:[^\n]*\n)*?"
    r"### \[Foreign investor[^\n]*\n",
)


SKIP_LINK_RE = re.compile(r"^\[Skip To[^\n]*\n?", re.MULTILINE)


def strip_residual_boilerplate(md: str) -> str:
    md = SKIP_LINK_RE.sub("", md)
    md = LOGIN_BLOCK_RE.sub("", md)
    idx = md.find(FOOTER_MARKER)
    if idx != -1:
        md = md[:idx]
    return md.strip()


def slugify(url: str) -> str:
    """Derive a filesystem-safe filename from the URL's last path segment."""
    path = urlparse(url).path.strip("/")
    last = path.split("/")[-1] if path else "index"
    slug = re.sub(r"[^a-zA-Z0-9-]", "-", last).strip("-").lower()
    return slug or "page"


async def main():
    strategy = BFSDeepCrawlStrategy(
        max_depth=MAX_DEPTH,
        max_pages=MAX_PAGES,
        include_external=False,  # never leave the starting site
        filter_chain=FilterChain([
            DomainFilter(allowed_domains=["www.ato.gov.au"]),  # main site only, no forum subdomain
            ContentTypeFilter(allowed_types=["text/html"]),  # skip PDFs/images
        ]),
    )
    # Same header/nav/footer/disclaimer HTML repeats on every ATO page, so without
    # pruning it gets chunked+embedded once per page (verified: 21 identical chunks
    # duplicated across pages, 2 of them present on all 25 — see notes.md). Pruning
    # scores DOM blocks by text density/link density and drops the low-density
    # boilerplate, keeping only the actual article body in fit_markdown.
    md_generator = DefaultMarkdownGenerator(
        content_filter=PruningContentFilter(threshold=0.48, threshold_type="fixed", min_word_threshold=5),
    )
    config = CrawlerRunConfig(deep_crawl_strategy=strategy, markdown_generator=md_generator, stream=False)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    saved = 0
    skipped = 0
    seen_slugs: dict[str, int] = {}

    async with AsyncWebCrawler() as crawler:
        results = await crawler.arun(url=START_URL, config=config)
        if not isinstance(results, list):
            results = [results]

        for result in results:
            if not getattr(result, "success", False):
                continue

            # Client-rendered search-tool pages (e.g. the Legal Database) have no
            # static article text at all — crawled markdown is 100% nav chrome
            # (verified: legaldatabase.md was still all header/footer links even
            # after boilerplate stripping).
            if "/single-page-applications/" in urlparse(result.url).path:
                skipped += 1
                continue

            # result.markdown is a MarkdownGenerationResult when a markdown_generator
            # is configured. fit_markdown is the pruned (boilerplate-stripped) body;
            # fall back to raw_markdown if pruning ever strips a page down to nothing.
            md_result = result.markdown
            md = (getattr(md_result, "fit_markdown", None) or "").strip()
            if len(md) < MIN_CONTENT_CHARS:
                md = (getattr(md_result, "raw_markdown", None) or str(md_result) or "").strip()
            md = strip_residual_boilerplate(md)
            if len(md) < MIN_CONTENT_CHARS:
                skipped += 1
                continue

            slug = slugify(result.url)
            # de-duplicate filenames when two URLs share a last segment
            n = seen_slugs.get(slug, 0)
            seen_slugs[slug] = n + 1
            if n:
                slug = f"{slug}-{n}"

            (OUTPUT_DIR / f"{slug}.md").write_text(md, encoding="utf-8")

            doc = Document()
            for line in md.splitlines():
                doc.add_paragraph(line)
            doc.save(OUTPUT_DIR / f"{slug}.docx")

            saved += 1
            print(f"Saved [{saved}] {slug}  ({len(md)} chars)")

    print(f"\nDone. Saved {saved} content pages, skipped {skipped} thin pages.")
    print(f"Output: {OUTPUT_DIR}")


if __name__ == "__main__":
    asyncio.run(main())
