#!/usr/bin/env python3
# Vendored from the `mdbook-figures` skill (v1.1.0, tests/test_figcheck.py); the checker it drives
# is scripts/check_figures.py beside it. Three changes from upstream, all noted in place:
#   * SCRIPT points at the vendored checker in this directory;
#   * W310 and W311 joined ALL_CODES and gained fixtures (upstream emits them untested);
#   * a scan test asserts every code the script passes to rep.add is listed here.
"""Negative tests for figcheck.py - one fixture per check, plus the regressions that motivated 1.1.0.

Run from the skill root:

    python3 -m unittest discover -s tests -v
    python3 tests/test_figcheck.py            # same thing, one file

Doctrine (from SKILL.md): a check that has never seen its defect is a hope. So almost every test
here *injects* one defect into a throwaway book and asserts the exact code comes back, with the
file it came from. The regressions section pins the false positives and false negatives found in
1.0.0 by probing - each one is a real fixture, not a description.

Every fixture is built in a temp directory by the `Book` helper: nothing is committed as a static
tree, so a fixture cannot silently rot into "the test still passes but tests something else".
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
SCRIPT = HERE / "check_figures.py"

# Every code the script may emit. `test_codes_are_documented` keeps this list, the docstring and
# the script in step; a new check that is not listed here fails that test on purpose.
# W310 and W311 were emitted by the script but *not* listed here upstream, so the reachability
# test could not prove them reachable and a later regression would have shipped unpinned. They are
# listed, given fixtures and added to the kitchen sink below.
ALL_CODES = {
    "E101", "E102", "E103", "E104", "E105", "E106",
    "E201", "E202", "E203", "E204",
    "W301", "W302", "W303", "W304", "W305", "W306", "W307", "W308", "W309",
    "W310", "W311",
}

CLEAN_FIGURE = """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 810 240" role="img" aria-label="a card and its label">
  <style>
    .fig-card { fill: var(--quote-bg, #13293d); stroke: var(--quote-border, #274357); stroke-width: 1; }
    .fig-t    { fill: var(--fg, #dce6f0); font-size: 12.5px; }
    .fig-a-pos { fill: #3f8f6f; }
  </style>
  <rect class="fig-card" x="40" y="48" width="230" height="120" rx="10"/>
  <text class="fig-t" x="60" y="80">A label</text>
</svg>
"""

CLEAN_PAGE = """# chapter

<figure class="book-figure">{{#include figures/01-a.svg}}</figure>

*Figure 1.1 - the one thing to take away.*
"""


def svg(body: str, viewbox: str = "0 0 810 240", extra: str = 'role="img" aria-label="test"') -> str:
    """A figure with the given body, indented so no blank lines sneak in."""
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="{viewbox}" {extra}>\n'
        + "".join(f"  {line}\n" for line in body.strip().splitlines())
        + "</svg>\n"
    )


class Book:
    """A throwaway mdBook tree; every path is created on demand under a temp root."""

    def __init__(self, root: Path) -> None:
        self.root = root

    def write(self, rel: str, text: str) -> Path:
        path = self.root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
        return path

    def figure(self, name: str, text: str) -> None:
        self.write(f"src/figures/{name}", text)

    def page(self, name: str, text: str) -> None:
        self.write(f"src/{name}", text)

    def book_toml(self, text: str) -> None:
        self.write("book.toml", text)

    def run(self, *args: str) -> subprocess.CompletedProcess:
        return subprocess.run(
            [sys.executable, str(SCRIPT), "--root", str(self.root), *args],
            capture_output=True, text=True,
        )

    def report(self, *args: str) -> dict:
        proc = self.run("--json", *args)
        if not proc.stdout.strip():
            raise AssertionError(f"no JSON on stdout (rc={proc.returncode}): {proc.stderr}")
        return json.loads(proc.stdout)

    def codes(self, *args: str) -> set[str]:
        rep = self.report(*args)
        return {f["code"] for f in rep["errors"] + rep["warnings"]}

    def findings(self, code: str, *args: str) -> list[dict]:
        rep = self.report(*args)
        return [f for f in rep["errors"] + rep["warnings"] if f["code"] == code]


class FigcheckTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory(prefix="figcheck-test-")
        self.addCleanup(self._tmp.cleanup)
        self.book = Book(Path(self._tmp.name))

    # -- helpers ---------------------------------------------------------------
    def assertCode(self, code: str, book: Book | None = None) -> list[dict]:
        book = book or self.book
        hits = book.findings(code)
        self.assertTrue(hits, f"expected {code}, got {sorted(book.codes())}")
        return hits

    def assertNoCode(self, code: str, book: Book | None = None) -> None:
        book = book or self.book
        self.assertNotIn(code, book.codes())


class CleanBook(FigcheckTestCase):
    def test_clean_book_reports_nothing(self) -> None:
        self.book.figure("01-a.svg", CLEAN_FIGURE)
        self.book.page("ch1.md", CLEAN_PAGE)
        self.book.book_toml(
            '[book]\ntitle = "t"\nsrc = "src"\n\n'
            '[output.html]\nadditional-css = ["css/figures.css"]\n'
        )
        self.book.write("css/figures.css", ".book-figure { margin: 1.8em 0; }\n")
        rep = self.book.report()
        self.assertEqual(rep["errors"], [])
        self.assertEqual(rep["warnings"], [])
        self.assertEqual(rep["summary"], {"figures": 1, "pages": 1, "errors": 0, "warnings": 0})


class SvgChecks(FigcheckTestCase):
    def test_e101_blank_line_inside_svg(self) -> None:
        self.book.figure("01-a.svg", CLEAN_FIGURE.replace("  <rect", "\n  <rect"))
        self.assertCode("E101")

    def test_e102_missing_viewbox(self) -> None:
        self.book.figure("01-a.svg", '<svg xmlns="http://www.w3.org/2000/svg" role="img" aria-label="x">\n  <rect x="1" y="1" width="2" height="2"/>\n</svg>\n')
        self.assertCode("E102")

    def test_e103_remote_reference(self) -> None:
        self.book.figure("01-a.svg", svg('<image href="https://example.com/a.png" x="0" y="0" width="10" height="10"/>'))
        self.assertCode("E103")

    def test_e103_allows_the_xml_namespace(self) -> None:
        self.book.figure("01-a.svg", CLEAN_FIGURE)
        self.assertNoCode("E103")

    def test_e104_hardcoded_colour(self) -> None:
        self.book.figure("01-a.svg", svg('<rect class="fig-card" x="40" y="48" width="10" height="10" fill="#ff0000"/>'))
        self.assertCode("E104")

    def test_e104_allows_var_with_fallback(self) -> None:
        self.book.figure("01-a.svg", svg('<rect class="fig-card" x="40" y="48" width="10" height="10" fill="var(--quote-bg, #13293d)"/>'))
        self.assertNoCode("E104")

    def test_e104_allows_accent_rule_body(self) -> None:
        self.book.figure("01-a.svg", svg('<style>.fig-a-pos { fill: #3f8f6f; }</style>\n<rect class="fig-a-pos" x="40" y="48" width="10" height="10"/>'))
        self.assertNoCode("E104")

    def test_e105_latex_command(self) -> None:
        self.book.figure("01-a.svg", svg('<text class="fig-t" x="40" y="48">\\alpha</text>'))
        self.assertCode("E105")

    def test_e105_math_in_dollars(self) -> None:
        self.book.figure("01-a.svg", svg('<text class="fig-t" x="40" y="48">$x^2$</text>'))
        self.assertCode("E105")

    def test_e106_character_outside_allowlist(self) -> None:
        self.book.figure("01-a.svg", svg('<text class="fig-t" x="40" y="48">cost \u20ac5</text>'))
        hits = self.assertCode("E106")
        self.assertIn("U+20AC", hits[0]["message"])

    def test_w301_no_style_block(self) -> None:
        self.book.figure("01-a.svg", svg('<rect class="fig-card" x="40" y="48" width="10" height="10"/>'))
        self.assertCode("W301")

    def test_w302_font_family_in_style(self) -> None:
        self.book.figure("01-a.svg", svg('<style>.fig-t { font-family: Inter; }</style>\n<text class="fig-t" x="40" y="48">t</text>'))
        self.assertCode("W302")

    def test_w303_missing_role(self) -> None:
        self.book.figure("01-a.svg", svg('<rect class="fig-card" x="40" y="48" width="10" height="10"/>', extra=''))
        self.assertCode("W303")

    def test_w304_element_outside_viewbox(self) -> None:
        self.book.figure("01-a.svg", svg('<rect class="fig-card" x="9000" y="9000" width="10" height="10"/>'))
        hits = self.assertCode("W304")
        self.assertIn("outside the viewBox", hits[0]["message"])

    def test_w307_viewbox_wider_than_page(self) -> None:
        self.book.figure("01-a.svg", svg('<rect class="fig-card" x="40" y="48" width="10" height="10"/>', viewbox="0 0 900 240"))
        self.assertCode("W307")

    def test_w307_boundary_at_820_is_clean(self) -> None:
        self.book.figure("01-a.svg", svg('<rect class="fig-card" x="40" y="48" width="10" height="10"/>', viewbox="0 0 820 240"))
        self.assertNoCode("W307")

    def test_w310_dark_only_accent_is_reported(self) -> None:
        # W310 reads the dark card from the figure's own var(--quote-bg, ...) fallback, so the
        # fixture has to carry it. #123456 is a dark-only mid-tone: it reads on the light card and
        # disappears on the dark one.
        self.book.figure("01-a.svg", svg(
            '<style>.fig-card { fill: var(--quote-bg, #13293d); }\n.fig-a-bad { fill: #123456; }</style>\n'
            '<rect class="fig-card" x="40" y="48" width="10" height="10"/>'))
        self.assertCode("W310")

    def test_w310_accent_that_reads_on_both_cards_is_clean(self) -> None:
        self.book.figure("01-a.svg", svg(
            '<style>.fig-card { fill: var(--quote-bg, #13293d); }\n.fig-a-pos { fill: #3f8f6f; }</style>\n'
            '<rect class="fig-card" x="40" y="48" width="10" height="10"/>'))
        self.assertNoCode("W310")

    def test_w311_label_running_past_the_viewbox(self) -> None:
        # Shrink the viewBox instead of writing a 200-character label, so the estimate cannot be
        # argued with. W304 judges the anchor point only, and the anchor is inside the box.
        self.book.figure("01-a.svg", svg(
            '<text class="fig-t" x="10" y="40">a label far too wide for this narrow figure</text>',
            viewbox="0 0 150 120"))
        self.assertCode("W311")

    def test_w311_centred_label_that_fits_is_clean(self) -> None:
        # A middle-anchored label is the case an estimate gets wrong in both directions; pin the
        # clean side so the check stays an estimate rather than a blanket failure.
        self.book.figure("01-a.svg", svg(
            '<text class="fig-t" x="100" y="40" text-anchor="middle">fits</text>',
            viewbox="0 0 200 120"))
        self.assertNoCode("W311")


class PageChecks(FigcheckTestCase):
    def test_e201_include_without_figure_wrapper(self) -> None:
        self.book.figure("01-a.svg", CLEAN_FIGURE)
        self.book.page("ch1.md", "# c\n\n{{#include figures/01-a.svg}}\n\n*Figure 1.1 - x*\n")
        self.assertCode("E201")

    def test_e202_missing_include_target(self) -> None:
        self.book.figure("01-a.svg", CLEAN_FIGURE)
        self.book.page("ch1.md", "# c\n\n<figure>{{#include figures/99-nope.svg}}</figure>\n\n*Figure 1.1 - x*\n")
        self.assertCode("E202")

    def test_e203_caption_missing(self) -> None:
        self.book.figure("01-a.svg", CLEAN_FIGURE)
        self.book.page("ch1.md", "# c\n\n<figure>{{#include figures/01-a.svg}}</figure>\n\nJust prose, no caption.\n")
        self.assertCode("E203")

    def test_e204_numbering_out_of_order(self) -> None:
        self.book.figure("01-a.svg", CLEAN_FIGURE)
        self.book.figure("02-b.svg", CLEAN_FIGURE)
        self.book.page("ch1.md",
                       "# c\n\n<figure>{{#include figures/02-b.svg}}</figure>\n\n*Figure 1.2 - second*\n\n"
                       "<figure>{{#include figures/01-a.svg}}</figure>\n\n*Figure 1.1 - first*\n")
        hits = self.assertCode("E204")
        self.assertEqual(len(hits), 2, "both an out-of-order caption and its follower are reported")
        self.assertIn("found 1.2", hits[0]["message"])

    def test_w305_caption_disagrees_with_file_prefix(self) -> None:
        self.book.figure("03-c.svg", CLEAN_FIGURE)
        self.book.page("ch1.md", "# c\n\n<figure>{{#include figures/03-c.svg}}</figure>\n\n*Figure 1.1 - x*\n")
        self.assertCode("W305")

    def test_w306_markdown_image(self) -> None:
        self.book.figure("01-a.svg", CLEAN_FIGURE)
        self.book.page("ch1.md", "# c\n\n![diagram](figures/01-a.svg)\n")
        self.assertCode("W306")

    def test_w306_html_img(self) -> None:
        self.book.figure("01-a.svg", CLEAN_FIGURE)
        self.book.page("ch1.md", '# c\n\n<img src="figures/01-a.svg" alt="d">\n')
        self.assertCode("W306")

    def test_w308_orphan_figure(self) -> None:
        self.book.figure("01-a.svg", CLEAN_FIGURE)
        self.book.figure("02-b.svg", CLEAN_FIGURE)
        self.book.page("ch1.md", CLEAN_PAGE)
        self.assertCode("W308")

    def test_w309_wrapper_css_not_wired(self) -> None:
        self.book.figure("01-a.svg", CLEAN_FIGURE)
        self.book.page("ch1.md", CLEAN_PAGE)
        self.book.write("css/figures.css", ".book-figure { margin: 0; }\n")
        self.book.book_toml('[book]\nsrc = "src"\n\n[output.html]\nadditional-css = ["css/other.css"]\n')
        self.assertCode("W309")

    def test_w309_silent_when_the_css_does_not_exist(self) -> None:
        self.book.figure("01-a.svg", CLEAN_FIGURE)
        self.book.page("ch1.md", CLEAN_PAGE)
        self.book.book_toml('[book]\nsrc = "src"\n\n[output.html]\nadditional-css = ["css/other.css"]\n')
        self.assertNoCode("W309")


class BookTomlDiscovery(FigcheckTestCase):
    def test_src_from_book_toml_is_honoured(self) -> None:
        self.book.book_toml('[book]\ntitle = "t"\nsrc = "docs"\n\n[output.html]\nadditional-css = ["css/figures.css"]\n')
        self.book.write("docs/figures/01-a.svg", CLEAN_FIGURE)
        self.book.write("docs/ch1.md", CLEAN_PAGE)
        self.book.write("css/figures.css", ".book-figure { margin: 0; }\n")
        rep = self.book.report()
        self.assertEqual(rep["summary"]["figures"], 1)
        self.assertEqual(rep["summary"]["pages"], 1)
        self.assertEqual(rep["errors"], [])


class Regressions(FigcheckTestCase):
    """Defects found in 1.0.0 by probing; each one is pinned so it cannot come back."""

    def test_multiline_figure_wrapper_is_accepted(self) -> None:
        # CommonMark type-6 HTML blocks run to the next blank line, so the wrapper may span lines.
        self.book.figure("01-a.svg", CLEAN_FIGURE)
        self.book.page("ch1.md",
                       "# c\n\n<figure class=\"book-figure\">\n{{#include figures/01-a.svg}}\n</figure>\n\n"
                       "*Figure 1.1 - x*\n")
        self.assertNoCode("E201")
        self.assertNoCode("E203")

    def test_include_with_line_range_is_not_a_missing_file(self) -> None:
        self.book.figure("01-a.svg", CLEAN_FIGURE)
        self.book.page("ch1.md", "# c\n\n<figure>{{#include figures/01-a.svg:1:5}}</figure>\n\n*Figure 1.1 - x*\n")
        self.assertNoCode("E202")

    def test_include_with_anchor_is_not_a_missing_file(self) -> None:
        self.book.figure("01-a.svg", CLEAN_FIGURE)
        self.book.page("ch1.md", "# c\n\n<figure>{{#include figures/01-a.svg:card}}</figure>\n\n*Figure 1.1 - x*\n")
        self.assertNoCode("E202")

    def test_url_fragment_id_is_not_a_colour(self) -> None:
        # `url(#fade)` is a gradient reference; `#fade` merely looks like hex.
        self.book.figure("01-a.svg", svg(
            '<defs><linearGradient id="fade"><stop offset="0" stop-color="var(--fg, #fff)"/></linearGradient></defs>\n'
            '<rect class="fig-card" x="40" y="48" width="10" height="10" fill="url(#fade)"/>'))
        self.assertNoCode("E104")

    def test_href_fragment_id_is_not_a_colour(self) -> None:
        self.book.figure("01-a.svg", svg('<use href="#ace" x="40" y="48"/>'))
        self.assertNoCode("E104")

    def test_accent_class_does_not_excuse_a_hardcoded_colour_on_the_same_element(self) -> None:
        # 1.0.0 exempted the whole *line* once it saw `fig-a-`, which hid the real defect.
        self.book.figure("01-a.svg", svg(
            '<style>.fig-a-pos { fill: #3f8f6f; }</style>\n'
            '<rect class="fig-a-pos" x="40" y="48" width="10" height="10" fill="#ff0000"/>'))
        hits = self.assertCode("E104")
        self.assertIn("#ff0000", hits[0]["message"])

    def test_currency_is_not_latex(self) -> None:
        self.book.figure("01-a.svg", svg('<text class="fig-t" x="40" y="48">cost $5 vs $7</text>'))
        self.assertNoCode("E105")

    def test_cross_chapter_numbering_on_one_page_is_clean(self) -> None:
        self.book.figure("03-c.svg", CLEAN_FIGURE)
        self.book.figure("04-d.svg", CLEAN_FIGURE)
        self.book.page("ch3.md",
                       "# c\n\n<figure>{{#include figures/03-c.svg}}</figure>\n\n*Figure 3.1 - a*\n\n"
                       "<figure>{{#include figures/04-d.svg}}</figure>\n\n*Figure 4.1 - b*\n")
        self.assertNoCode("E204")
        self.assertNoCode("W305")

    def test_single_quoted_role_is_accepted(self) -> None:
        self.book.figure("01-a.svg", CLEAN_FIGURE.replace('role="img" aria-label="a card and its label"',
                                                          "role='img' aria-label='a card and its label'"))
        self.assertNoCode("W303")

    def test_role_on_an_inner_element_does_not_count(self) -> None:
        self.book.figure("01-a.svg",
                         '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 810 240">\n'
                         '  <g role="img" aria-label="nope"><rect x="1" y="1" width="2" height="2"/></g>\n'
                         "</svg>\n")
        self.assertCode("W303")

    def test_font_family_in_a_comment_is_not_a_declaration(self) -> None:
        self.book.figure("01-a.svg", svg(
            '<style>.fig-t { fill: var(--fg, #dce6f0); }</style>\n'
            "<!-- never set font-family here -->\n"
            '<text class="fig-t" x="40" y="48">t</text>'))
        self.assertNoCode("W302")

    def test_polygon_outside_viewbox_is_caught(self) -> None:
        self.book.figure("01-a.svg", svg('<polygon points="8000,8000 9000,9000 8500,9500"/>'))
        self.assertCode("W304")

    def test_absolute_path_outside_viewbox_is_caught(self) -> None:
        self.book.figure("01-a.svg", svg('<path d="M0 0 L 9000 9000"/>'))
        self.assertCode("W304")

    def test_relative_path_is_skipped_not_guessed(self) -> None:
        # A naive number sweep over `l 9000 9000` would be a false positive; the check abstains.
        self.book.figure("01-a.svg", svg('<path d="m10 10 l 5 5 h 10 v 10"/>'))
        self.assertNoCode("W304")

    def test_markdown_image_with_title_is_caught(self) -> None:
        self.book.figure("01-a.svg", CLEAN_FIGURE)
        self.book.page("ch1.md", '# c\n\n![diagram](figures/01-a.svg "a title")\n')
        self.assertCode("W306")

    def test_relative_path_computes_a_real_box(self) -> None:
        # Guard the guard: the abstention must not silence *absolute* paths that are in range.
        self.book.figure("01-a.svg", svg('<path d="M40 48 L 200 160"/>'))
        self.assertNoCode("W304")


class CliContract(FigcheckTestCase):
    def _cli(self, *args: str) -> subprocess.CompletedProcess:
        return subprocess.run([sys.executable, str(SCRIPT), *args], capture_output=True, text=True)

    def test_version(self) -> None:
        proc = self._cli("--version")
        self.assertEqual(proc.returncode, 0)
        self.assertRegex(proc.stdout.strip(), r"^figcheck \d+\.\d+\.\d+$")

    def test_show_allow_chars_exits_zero_and_prints_a_pasteable_option(self) -> None:
        proc = self._cli("--show-allow-chars")
        self.assertEqual(proc.returncode, 0)
        self.assertIn("U+2014", proc.stdout)
        self.assertIn("--allow-chars \"", proc.stdout)

    def test_allow_chars_empty_means_ascii_only(self) -> None:
        self.book.figure("01-a.svg", svg('<text class="fig-t" x="40" y="48">\u2014 dash</text>'))
        self.assertNoCode("E106")  # em dash is allowed by default
        self.assertIn("E106", self.book.codes("--allow-chars", ""))

    def test_strict_turns_warnings_into_a_failure(self) -> None:
        self.book.figure("01-a.svg", svg('<rect class="fig-card" x="40" y="48" width="10" height="10"/>'))
        lenient = self.book.run()
        strict = self.book.run("--strict")
        self.assertEqual(lenient.returncode, 0)
        self.assertEqual(strict.returncode, 1)

    def test_json_summary_shape(self) -> None:
        self.book.figure("01-a.svg", CLEAN_FIGURE)
        rep = self.book.report()
        self.assertEqual(set(rep), {"summary", "errors", "warnings"})
        self.assertEqual(set(rep["summary"]), {"figures", "pages", "errors", "warnings"})

    def test_max_width_is_configurable(self) -> None:
        self.book.figure("01-a.svg", svg('<rect class="fig-card" x="40" y="48" width="10" height="10"/>', viewbox="0 0 900 240"))
        self.assertIn("W307", self.book.codes())              # 900 > the 820 default
        self.assertNotIn("W307", self.book.codes("--max-width", "1200"))

    def test_positional_figures_dir_runs_svg_checks_only(self) -> None:
        self.book.figure("01-a.svg", CLEAN_FIGURE)
        proc = self.book.run(str(self.book.root / "src" / "figures"))
        self.assertEqual(proc.returncode, 0)
        self.assertIn("(orphan check skipped: no pages)", proc.stdout)

    def test_missing_root_is_a_usage_error(self) -> None:
        proc = self._cli("--root", str(self.book.root / "nowhere"))
        self.assertEqual(proc.returncode, 1)
        self.assertIn("no figures or pages found", proc.stderr)


class Coverage(FigcheckTestCase):
    def test_every_code_is_documented_in_the_docstring(self) -> None:
        head = SCRIPT.read_text(encoding="utf-8").split('"""')[1]
        missing = sorted(code for code in ALL_CODES if code not in head)
        self.assertEqual(missing, [], f"codes missing from the docstring: {missing}")

    def test_no_undocumented_code_is_emitted(self) -> None:
        # The script must not invent a code that ALL_CODES (and the docstring) do not list.
        emitted = set()
        book = self._kitchen_sink()
        for finding in book.report()["errors"] + book.report()["warnings"]:
            emitted.add(finding["code"])
        self.assertTrue(emitted <= ALL_CODES, f"undocumented codes: {sorted(emitted - ALL_CODES)}")

    def test_every_code_the_script_emits_is_listed(self) -> None:
        # The kitchen sink only proves that *listed* codes are reachable, so a check added to the
        # script and nowhere else passes the whole suite with its rule unpinned. Scan the call sites
        # and require every code the script passes to rep.add to be in ALL_CODES.
        emitted = set(re.findall(
            r'rep\.add\(\s*"(?:error|warning)",\s*"([EW]\d{3})"',
            SCRIPT.read_text(encoding="utf-8")))
        self.assertTrue(emitted, "the scan found no rep.add call sites - the pattern is stale")
        self.assertEqual(sorted(emitted - ALL_CODES), [])

    def test_every_code_has_at_least_one_fixture(self) -> None:
        book = self._kitchen_sink()
        emitted = book.codes()
        self.assertEqual(sorted(ALL_CODES - emitted), [],
                         "every code must be reachable; these were not produced")

    def _kitchen_sink(self) -> Book:
        """One book that trips every code at least once. Kept in one place so a new check forces a
        new fixture line here rather than a silently skipped rule."""
        b = Book(Path(tempfile.mkdtemp(prefix="figcheck-sink-")))
        self.addCleanup(shutil.rmtree, b.root, ignore_errors=True)
        b.book_toml('[book]\ntitle = "sink"\nsrc = "src"\n\n[output.html]\nadditional-css = ["css/other.css"]\n')
        b.write("css/figures.css", ".book-figure { margin: 0; }\n")
        # E101 E103 E104 E105 E106 W301 W302 W303 W304
        b.figure("01-a.svg",
                 '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 810 240">\n'
                 "\n"
                 '  <image href="https://example.com/a.png" x="0" y="0" width="9" height="9"/>\n'
                 '  <rect x="40" y="48" width="10" height="10" fill="#ff0000"/>\n'
                 '  <text x="40" y="80">\\alpha</text>\n'
                 '  <text x="40" y="100">cost \u20ac5</text>\n'
                 '  <text x="40" y="120" style="font-family: Inter">t</text>\n'
                 '  <rect x="9000" y="9000" width="10" height="10"/>\n'
                 "</svg>\n")
        b.figure("02-b.svg", '<svg xmlns="http://www.w3.org/2000/svg" role="img" aria-label="x">\n  <rect x="1" y="1" width="2" height="2"/>\n</svg>\n')  # E102
        b.figure("03-c.svg", svg('<rect class="fig-card" x="40" y="48" width="10" height="10"/>', viewbox="0 0 900 240"))  # W307, W308
        b.figure("04-d.svg", CLEAN_FIGURE)  # W308
        b.figure("05-e.svg", svg(  # W310
            '<style>.fig-card { fill: var(--quote-bg, #13293d); }\n.fig-a-bad { fill: #123456; }</style>\n'
            '<rect class="fig-card" x="40" y="48" width="10" height="10"/>'))
        b.figure("06-f.svg", svg(  # W311
            '<text class="fig-t" x="10" y="40">a label far too wide for this narrow figure</text>',
            viewbox="0 0 150 120"))
        b.page("unwrap.md", "# c\n\n{{#include figures/04-d.svg}}\n\n*Figure 4.1 - x*\n")  # E201
        b.page("missing.md", "# c\n\n<figure>{{#include figures/99-nope.svg}}</figure>\n\n*Figure 1.1 - x*\n")  # E202
        b.page("nocap.md", "# c\n\n<figure>{{#include figures/04-d.svg}}</figure>\n\nProse only.\n")  # E203
        b.page("order.md",
               "# c\n\n<figure>{{#include figures/03-c.svg}}</figure>\n\n*Figure 3.2 - second*\n\n"
               "<figure>{{#include figures/04-d.svg}}</figure>\n\n*Figure 3.1 - first*\n")  # E204
        b.page("prefix.md", "# c\n\n<figure>{{#include figures/03-c.svg}}</figure>\n\n*Figure 1.1 - x*\n")  # W305
        b.page("img.md", "# c\n\n![d](figures/04-d.svg)\n")  # W306
        return b


if __name__ == "__main__":
    unittest.main(verbosity=2)
