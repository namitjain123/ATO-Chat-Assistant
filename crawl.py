import asyncio
import re
import sys
from pathlib import Path
from urllib.parse import urlparse

from docx import Document
from crawl4ai import AsyncWebCrawler, CrawlerRunConfig
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
    config = CrawlerRunConfig(deep_crawl_strategy=strategy, stream=False)

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

            md = result.markdown or ""
            if len(md.strip()) < MIN_CONTENT_CHARS:
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
