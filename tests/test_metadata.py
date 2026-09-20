"""
Tests for app/ingestion/metadata.py — pure functions, no I/O.
"""
from app.ingestion.metadata import (
    TOPICS, title_from_filename, page_summary, heading_positions, section_at,
    locate_chunks, income_years, normalize_topics,
)

DOC = """Deductions for gifts or donations you make to deductible gift recipients.
  * [When a gift is deductible](https://example.com#a)

##  When a gift or donation is deductible
You can claim a tax deduction for a gift.

### [Removal of the $2 donation threshold](https://example.com/x)
The threshold no longer applies.
"""


def test_title_from_filename():
    assert title_from_filename("gifts-and-donations.md") == "Gifts and donations"


def test_title_drops_crawl_duplicate_suffix():
    assert title_from_filename("deductions-you-can-claim-1.md") == "Deductions you can claim"


def test_page_summary_is_first_prose_line():
    assert page_summary(DOC) == "Deductions for gifts or donations you make to deductible gift recipients."


def test_page_summary_skips_headings_and_lists():
    assert page_summary("## Heading\n  * item\nReal first sentence.") == "Real first sentence."


def test_headings_are_cleaned_of_markdown_links_and_whitespace():
    assert [h for _, h in heading_positions(DOC)] == [
        "When a gift or donation is deductible",
        "Removal of the $2 donation threshold",
    ]


def test_section_is_nearest_preceding_heading():
    headings = heading_positions(DOC)
    assert section_at(headings, DOC.index("You can claim")) == "When a gift or donation is deductible"
    assert section_at(headings, DOC.index("The threshold no longer")) == "Removal of the $2 donation threshold"


def test_section_before_first_heading_or_unlocated_is_empty():
    headings = heading_positions(DOC)
    assert section_at(headings, 0) == ""
    assert section_at(headings, -1) == ""


def test_locate_chunks_searches_forward_for_repeated_text():
    doc = "intro. repeated phrase here. middle. repeated phrase here. end."
    first, second = locate_chunks(doc, ["repeated phrase here.", "repeated phrase here."])
    assert first < second


def test_locate_chunks_returns_minus_one_when_missing():
    assert locate_chunks("abc", ["zzz"]) == [-1]


def test_income_years_normalises_dash_variants_and_long_form():
    assert income_years("from 2026–27, also 2024-2025 and 2025 ‑ 26") == ["2024-25", "2025-26", "2026-27"]


def test_income_years_ignores_non_consecutive_date_spans():
    assert income_years("rates from 2001-2025 apply") == []


def test_income_years_empty_for_bare_calendar_year():
    assert income_years("tax slab for 2025") == []


def test_normalize_topics_keeps_only_known_and_dedupes():
    assert normalize_topics(["Tax Rates", "tax-rates", "made_up", None]) == ["tax_rates"]
    assert set(TOPICS) >= {"deductions", "record_keeping"}
