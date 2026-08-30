"""
Tests for crawl.py's boilerplate-stripping and slug-generation helpers.

strip_residual_boilerplate exists because crawl4ai's PruningContentFilter
alone didn't catch everything — verified during the ATO re-crawl that 3/25
pages still carried duplicated login-widget/footer chrome after pruning
(see notes.md). These tests pin down exactly what it's supposed to remove.
"""
from crawl import strip_residual_boilerplate, slugify


def test_strips_footer_from_marker_onward():
    md = "Real article content here.\n\n### Tools\n  * Media centre\n  * Forms"
    result = strip_residual_boilerplate(md)
    assert result == "Real article content here."
    assert "### Tools" not in result


def test_strips_skip_to_link():
    md = "[Skip To Alex Virtual Assistant](https://example.com/page#alex)\nReal content follows."
    result = strip_residual_boilerplate(md)
    assert "Skip To" not in result
    assert "Real content follows." in result


def test_strips_login_widget_block():
    md = (
        "Real content before.\n\n"
        "Log in to online servicesLog in\n"
        "## Log in to ATO online services\n"
        "Access secure services, view your details and lodge online.\n"
        "### [Individuals For individuals... ](https://example.com/individuals)\n"
        "### [Business  A secure system...  ](https://example.com/business)\n"
        "### [Foreign investor You or your representative... ](https://example.com/foreign)\n"
        "Real content after."
    )
    result = strip_residual_boilerplate(md)
    assert "Log in to ATO online services" not in result
    assert "Real content before." in result
    assert "Real content after." in result


def test_page_with_no_boilerplate_is_unchanged_aside_from_stripping_whitespace():
    md = "Just a normal paragraph of real content.\n\nAnd another one."
    result = strip_residual_boilerplate(md)
    assert result == md


def test_page_that_is_entirely_boilerplate_becomes_empty_or_near_empty():
    # Mirrors the real legaldatabase.md case: a page that's 100% site chrome
    # (login widget + nav) should end up empty or trivially short once
    # stripped, which crawl.py then treats as "skip this page."
    md = (
        "[Skip To Alex Virtual Assistant](https://example.com#alex)\n"
        "Log in to online servicesLog in\n"
        "## Log in to ATO online services\n"
        "Access secure services.\n"
        "### [Foreign investor ... ](https://example.com/foreign)\n"
        "### Tools\n  * Media centre"
    )
    result = strip_residual_boilerplate(md)
    assert len(result) < 20


def test_slugify_extracts_last_path_segment():
    assert slugify("https://www.ato.gov.au/individuals-and-families/deductions") == "deductions"


def test_slugify_handles_root_url():
    assert slugify("https://www.ato.gov.au/") == "index"


def test_slugify_replaces_invalid_characters():
    # Query strings aren't part of urlparse().path, so they're correctly
    # excluded from the slug entirely — only the underscore gets replaced.
    assert slugify("https://example.com/some_page?query=1") == "some-page"


def test_slugify_lowercases():
    assert slugify("https://example.com/Some-Page") == "some-page"


def test_slugify_never_returns_empty_string():
    # A URL whose last segment is entirely non-alphanumeric would otherwise
    # collapse to an empty slug — must fall back to something usable.
    result = slugify("https://example.com/???")
    assert result != ""
