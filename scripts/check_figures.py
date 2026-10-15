#!/usr/bin/env python3
# Vendored from the `mdbook-figures` skill (v1.1.0, scripts/figcheck.py). Only this header is
# added: every rule, message and CLI flag below is upstream's, so a fix can be taken from there
# without a merge. The checker's own fixtures live beside it in scripts/check_figures_tests.py,
# and `check-docs` runs both (`--root . --strict`, then the tests).
#
# Why the repo carries a copy instead of calling the skill: CI must run the version the repo
# pins, and a self-contained repository is the only copy a runner can execute.
"""figcheck - lint an mdBook repo's figures, and the pages that include them.

Every rule here was broken at least once in a real book, and `mdbook build` stayed green through
all of them - that is the entire reason this script exists. What each check costs when it is
missing:

  E101 blank line inside an SVG          CommonMark ends the HTML block -> mdBook warns, page breaks
  E102 viewBox missing                   the figure cannot scale with the column
  E103 remote reference                  the offline/print route breaks silently
  E104 literal colour                    low-contrast text on the other theme
  E105 LaTeX left in the text            an SVG cannot see MathJax: the reader sees the backslash
  E106 non-ASCII outside the allowlist   the repo's character gate fails the build
  E201 include without <figure>          the wrapper is what makes the SVG one HTML block
  E202 include whose file is missing     a hole in the page, and the build does not care
  E203 caption missing or malformed      the figure has no stated takeaway
  E204 caption numbering out of order    Figure 3.2 above Figure 3.1 in the same chapter
  W301 no <style> block                  the figure cannot follow the theme at all
  W302 font-family declared              overrides the page font instead of inheriting it
  W303 no role/aria-label                invisible to a reader who cannot see it
  W304 element outside the viewBox       silently invisible content; no tool warns about this
  W305 caption vs file prefix            Figure 3.1 in a file named 02-... : easy to mis-number
  W306 <img> to a local SVG              loses the theme (and breaks on GitHub)
  W307 viewBox wider than the page       every label scales down with it
  W308 orphan figure                     a file no page includes
  W309 wrapper CSS not wired             .book-figure / caption styles never load
  W310 accent contrast                   a mid-tone that reads on the dark card, faint on the light
  W311 text past the viewBox edge        the width is estimated, not left to the anchor point alone

Known blind spots, stated so nobody reads silence as proof:

  * W304 judges geometry only for shapes whose coordinates are absolute and self-contained
    (`rect`, `circle`, `ellipse`, `line`, `polygon`, `polyline`, `text`, and `path` built from
    M/L/C/S/Q/T only). A `path` using relative commands, or any `g`/`use`/`image`, is skipped:
    its bounding box cannot be derived without a real renderer.
  * W311 estimates a `text` element's rendered width from a per-character table and the class's own
    `font-size`/`letter-spacing`, so a long label running past the right edge is caught - which no
    other check does, because an SVG has no text metrics without a renderer. It is an estimate: a
    font much wider than the table, an element carrying `transform` or `textLength`, or a size set
    inline rather than in CSS, can still slip through or fire wrongly. W304 keeps judging the anchor
    point separately, so a label anchored outside the box is caught even if the width estimate is off.
  * E204 numbering is checked within a page; a chapter split across pages is not compared.
  * W309 fires only when the CSS file it names actually exists in the repo, so a book that folds
    the wrapper styles into its own stylesheet is not nagged.

Usage:
    python3 figcheck.py                          # --root . : <book src>/figures/*.svg + <book src>/**/*.md
    python3 figcheck.py --root ~/git/mine/book
    python3 figcheck.py src/figures             # a figures dir alone: SVG checks only
    python3 figcheck.py --json                   # machine-readable, to self-correct in a loop
    python3 figcheck.py --strict                 # warnings fail too; use this in CI
    python3 figcheck.py --show-allow-chars       # print the effective character allowlist
    python3 figcheck.py --version

When `book.toml` is present its `book.src` is honoured (a book with `src = "docs"` is checked in
`docs/`, not `src/`), and W309 checks that the wrapper CSS is listed in
`[output.html] additional-css`.

Exit status: 0 when nothing must be fixed, 1 when something must.

The only characters this file itself uses are ASCII, on purpose: a repo with a character gate will
scan this script too if it is vendored into the repo.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

try:  # Python 3.11+; older interpreters simply lose book.toml discovery and W309
    import tomllib
except ModuleNotFoundError:  # pragma: no cover
    tomllib = None  # type: ignore[assignment]

VERSION = "1.1.0"

# The typographic and notation characters a figure is allowed to carry, spelled as escapes so this
# file stays ASCII. Default = the set one book settled on: em dash, en dash, ellipsis, arrows, box
# drawing, the status marks, approximation, and the Greek notation letters a figure needs because an
# SVG cannot typeset LaTeX (gamma lambda delta pi theta beta tau). Override per repo:
#   --allow-chars ""            pure ASCII only
#   --allow-chars "\u2014\u03bb"  just an em dash and lambda
DEFAULT_ALLOW_CHARS = (
    "\u2014\u2013\u2026\u2192\u2194\u25ba\u25b2\u2500\u2502\u250c\u2510\u2514\u2518"
    "\u2705\u26a0\ufe0f\u274c\u2248\u03b3\u03bb\u03b4\u03c0\u03b8\u03b2\u03c4"
)

SVG_INCLUDE = re.compile(r"\{\{#include\s+([^}]+?)\s*\}\}")
FIGURE_OPEN = re.compile(r"<figure\b", re.I)
CAPTION = re.compile(r"^\*\s*(?:Figure|Fig\.)\s+(\d+)\.(\d+)\b", re.I)
MD_SVG_IMG = re.compile(r"!\[[^\]]*\]\(\s*([^)\s]+\.svg)(?:\s+(?:\"[^\"]*\"|'[^']*'))?\s*\)", re.I)
HTML_SVG_IMG = re.compile(r"<img\b[^>]*\bsrc\s*=\s*(?:\"([^\"]+\.svg)\"|'([^']+\.svg)')", re.I)
LITERAL_COLOUR = re.compile(r"#[0-9a-fA-F]{3,8}\b|\brgba?\(|\bhsla?\(")
VAR_FALLBACK = re.compile(r"var\(\s*--[A-Za-z0-9-]+\s*,[^)]*\)")
# A LaTeX command, or the escaped delimiters/subscripts an SVG cannot typeset.
LATEX_CMD = re.compile(r"\\[a-zA-Z]{2,}|\\\(|\\\)|\^\{|_\{")
LATEX_MATH_DOLLAR = re.compile(r"\$([^$\n]{1,120})\$")
# `$...$` is treated as maths only when the content cannot be prose: a LaTeX/notation character, or
# no whitespace at all plus a letter. "cost $5 vs $7" is currency, "$A(s,a)$" is maths.
MATH_CHARS = re.compile(r"[\\^_{}]")
ELEMENT = re.compile(r"<(rect|circle|ellipse|line|text|polygon|polyline|path)\b([^>]*)>", re.I)
ATTR = re.compile(r"([A-Za-z][A-Za-z0-9:-]*)\s*=\s*\"([^\"]*)\"")
STYLE_BLOCK = re.compile(r"<style[^>]*>(.*?)</style>", re.S | re.I)
CSS_RULE = re.compile(r"([^{}]+)\{([^{}]*)\}")
CSS_COMMENT = re.compile(r"/\*.*?\*/", re.S)
HTML_COMMENT = re.compile(r"<!--.*?-->", re.S)
# A fragment reference is an id lookup, not a colour, even when the id is spelled in hex (`#fade`).
FRAGMENT_REF = re.compile(r"(?:xlink:href|href)\s*=\s*[\"']#[^\"']*[\"']|url\(\s*#[^)]*\)", re.I)
PATH_COMMANDS = re.compile(r"[MmLlHhVvCcSsQqTtAaZz]")
ABSOLUTE_PAIR_COMMANDS = set("MLCSTQ")
# W310 compares every semantic accent against the two card fills a figure is drawn on. The dark one
# is read from the figure itself (the fallback in `var(--quote-bg, #...)`); the light one is mdBook
# 0.5's light-theme --quote-bg. A figure with a dark-only palette looks perfect in the source and
# washes out for every reader on the light theme, which is why this is arithmetic and not an eye.
LIGHT_CARD_FILL = "#f2f7f9"
ACCENT_RULE_TEMPLATE = r"\.([A-Za-z0-9_-]*{prefix}[A-Za-z0-9_-]*)\s*\{{([^{{}}]*)\}}"
HEX_COLOUR = re.compile(r"#[0-9a-fA-F]{6}\b")
TEXT_ELEMENT = re.compile(r"<text\b([^>]*)>(.*?)</text>", re.S | re.I)
# W311 estimates a label's rendered width, because the source has no text metrics. The table is in em
# per character and is calibrated against a real book: every figure that renders correctly passes,
# and a caption that clipped at the right edge of a figure every other check called clean does not.
TEXT_WIDTH_EM = {"space": 0.28, "narrow": 0.30, "upper_or_digit": 0.65, "lower": 0.47}
TEXT_NARROW = set("iljtfr.,;:!|'`()[]{}/\\-")
# An estimate gets a tolerance, so that a label ending within a few units of the edge is not nagged.
TEXT_EDGE_TOLERANCE = 4.0
TEXT_DEFAULT_SIZE = 12.0


def _relative_luminance(colour: str) -> float:
    parts = [int(colour[i : i + 2], 16) / 255 for i in (1, 3, 5)]
    linear = [(v / 12.92 if v <= 0.03928 else ((v + 0.055) / 1.055) ** 2.4) for v in parts]
    return 0.2126 * linear[0] + 0.7152 * linear[1] + 0.0722 * linear[2]


def _contrast_ratio(a: str, b: str) -> float:
    la, lb = _relative_luminance(a), _relative_luminance(b)
    hi, lo = max(la, lb), min(la, lb)
    return (hi + 0.05) / (lo + 0.05)


class Config:
    def __init__(self, args: argparse.Namespace) -> None:
        self.max_width = args.max_width
        self.accent_prefix = args.accent_prefix
        self.allow_chars = set(args.allow_chars)
        self.figure_css = args.figure_css


class Report:
    def __init__(self) -> None:
        self.findings: list[dict] = []

    def add(self, level: str, code: str, path: Path | str, line: int | None, message: str) -> None:
        self.findings.append(
            {"level": level, "code": code, "file": str(path), "line": line, "message": message}
        )

    def errors(self) -> list[dict]:
        return [f for f in self.findings if f["level"] == "error"]

    def warnings(self) -> list[dict]:
        return [f for f in self.findings if f["level"] == "warning"]


def _looks_like_math(body: str) -> bool:
    if MATH_CHARS.search(body):
        return True
    return not re.search(r"\s", body) and bool(re.search(r"[A-Za-z]", body))


def _masked_text(text: str, cfg: Config) -> str:
    """Blank out what the colour/LaTeX scan must not look at, keeping every offset stable.

    Masked: HTML comments, CSS comments, `var(--x, #fallback)` fallbacks, fragment references
    (`url(#fade)`, `href="#ace"`), and the body of every CSS rule whose selector carries an accent
    class (`fig-a-*`) - the only place a literal colour is allowed. Masking the rule *body* rather
    than the whole line is what keeps a hardcoded colour beside a `fig-a-` class visible.
    """
    chars = list(text)
    spans: list[tuple[int, int]] = []
    spans += [(m.start(), m.end()) for m in HTML_COMMENT.finditer(text)]
    spans += [(m.start(), m.end()) for m in VAR_FALLBACK.finditer(text)]
    spans += [(m.start(), m.end()) for m in FRAGMENT_REF.finditer(text)]
    for block in STYLE_BLOCK.finditer(text):
        body_off, body = block.start(1), block.group(1)
        spans += [(body_off + m.start(), body_off + m.end()) for m in CSS_COMMENT.finditer(body)]
        for rule in CSS_RULE.finditer(body):
            if cfg.accent_prefix in rule.group(1):
                spans.append((body_off + rule.start(2), body_off + rule.end(2)))
    for start, end in spans:
        for i in range(start, min(end, len(chars))):
            if chars[i] != "\n":
                chars[i] = " "
    return "".join(chars)


def _font_metrics(text: str) -> dict[str, tuple[float, float, str]]:
    """Read each CSS class's font-size, letter-spacing and text-anchor out of the figure's <style>.

    W311 needs them, and a figure that sets its label sizes in CSS - which is how these figures stay
    theme-aware - has nowhere else to say so. An inline `style=`/`font-size=` attribute is not read;
    that is one of the check's stated blind spots.
    """
    metrics: dict[str, tuple[float, float, str]] = {}
    for block in STYLE_BLOCK.finditer(text):
        for rule in CSS_RULE.finditer(block.group(1)):
            selectors, body = rule.group(1), rule.group(2)
            size = re.search(r"font-size\s*:\s*([\d.]+)px", body)
            if not size:
                continue
            spacing = re.search(r"letter-spacing\s*:\s*([\d.]+)(px|rem|em)", body)
            anchor = re.search(r"text-anchor\s*:\s*([a-z]+)", body)
            px = float(size.group(1))
            gap = 0.0
            if spacing:
                amount = float(spacing.group(1))
                gap = amount if spacing.group(2) == "px" else amount * px
            for one in selectors.split(","):
                name = one.strip().lstrip(".")
                if name and " " not in name:
                    metrics[name] = (px, gap, anchor.group(1) if anchor else "start")
    return metrics


def _text_width(body: str, size: float, gap: float) -> float:
    """Estimate a label's rendered width in user units, from the per-character table."""
    total = 0.0
    for char in body:
        if char == " ":
            total += TEXT_WIDTH_EM["space"]
        elif char.isdigit() or char.isupper():
            total += TEXT_WIDTH_EM["upper_or_digit"]
        elif char in TEXT_NARROW:
            total += TEXT_WIDTH_EM["narrow"]
        else:
            total += TEXT_WIDTH_EM["lower"]
    return total * size + gap * max(0, len(body) - 1)


def check_svg(path: Path, cfg: Config, rep: Report) -> None:
    text = path.read_text(encoding="utf-8", errors="replace")
    lines = text.splitlines()

    for n, line in enumerate(lines, 1):
        if not line.strip():
            rep.add("error", "E101", path, n,
                    "blank line inside the SVG ends the CommonMark HTML block early - delete it")

    box = re.search(r'viewBox\s*=\s*"([^"]+)"', text)
    if not box:
        rep.add("error", "E102", path, 1, "no viewBox: the figure cannot scale with the page column")
        vx = vy = vw = vh = None
    else:
        nums = [float(x) for x in re.findall(r"-?\d+(?:\.\d+)?", box.group(1))[:4]]
        if len(nums) != 4:
            rep.add("error", "E102", path, 1, f"viewBox needs four numbers, got: {box.group(1)}")
            vx = vy = vw = vh = None
        else:
            vx, vy, vw, vh = nums
            if vw > cfg.max_width:
                rep.add("warning", "W307", path, 1,
                        f"viewBox is {vw:.0f} wide (over {cfg.max_width:g}): the page scales it "
                        "down, and the smallest labels go with it")

    for n, line in enumerate(lines, 1):
        for url in re.findall(r"https?://[^\s\"'()<>]+", line):
            if url.startswith("http://www.w3.org/") or url.startswith("https://www.w3.org/"):
                continue  # the XML namespace, not a fetch
            rep.add("error", "E103", path, n, f"remote reference {url}: a figure must render offline")

    masked = _masked_text(text, cfg)
    for m in LITERAL_COLOUR.finditer(masked):
        rep.add("error", "E104", path, masked[: m.start()].count("\n") + 1,
                f"hardcoded colour {m.group(0)!r}: use var(--theme-variable, #fallback), or put it "
                f"in a rule whose selector carries {cfg.accent_prefix}* for a semantic accent")

    # W310: an accent is a mid-tone, and a mid-tone has to clear 3:1 on *both* card fills. A dark-only
    # accent is invisible in review - it looks right in whichever theme the author is reading in.
    # Scanned on the raw text with comments blanked, not on `masked`: masking blanks the body of every
    # accent rule (that is how E104 tolerates the accents) and the var() fallback, i.e. exactly the two
    # things this check needs to read.
    uncommented = HTML_COMMENT.sub(lambda m: " " * len(m.group(0)), text)
    uncommented = CSS_COMMENT.sub(lambda m: " " * len(m.group(0)), uncommented)
    fallback = re.search(r"var\(\s*--quote-bg\s*,\s*(#[0-9a-fA-F]{6})\s*\)", uncommented)
    dark_card = fallback.group(1) if fallback else None
    accent_rule = re.compile(ACCENT_RULE_TEMPLATE.format(prefix=re.escape(cfg.accent_prefix)))
    for rule in accent_rule.finditer(uncommented):
        cls, body = rule.group(1), rule.group(2)
        for colour in HEX_COLOUR.findall(body):
            for theme, card in (("dark", dark_card), ("light", LIGHT_CARD_FILL)):
                if not card:
                    continue
                ratio = _contrast_ratio(colour, card)
                if ratio < 3.0:
                    rep.add("warning", "W310", path, uncommented[: rule.start()].count("\n") + 1,
                            f"accent .{cls} {colour} is {ratio:.2f}:1 against the {theme} card "
                            f"({card}): pick a mid-tone that clears 3:1 on both themes")

    m = LATEX_CMD.search(masked)
    if not m:
        for dm in LATEX_MATH_DOLLAR.finditer(masked):
            if _looks_like_math(dm.group(1)):
                m = dm
                break
    if m:
        rep.add("error", "E105", path, masked[: m.start()].count("\n") + 1,
                f"LaTeX ({m.group(0)!r}) inside a figure: the SVG cannot see MathJax, write it as text")

    stray = sorted({c for c in text if ord(c) > 127 and c not in cfg.allow_chars})
    if stray:
        shown = ", ".join(f"{c!r} U+{ord(c):04X}" for c in stray)
        rep.add("error", "E106", path, text[: text.find(stray[0])].count("\n") + 1,
                f"character(s) outside the allowed set: {shown} - add to the repo allowlist or "
                "replace with ASCII")

    if "<style" not in text:
        rep.add("warning", "W301", path, 1,
                "no <style> block: the figure cannot follow the theme at all")
    # `masked` has comments blanked, so a note that merely *mentions* font-family does not fire.
    if "font-family" in masked:
        rep.add("warning", "W302", path, masked[: masked.find("font-family")].count("\n") + 1,
                "font-family in a figure overrides the page font instead of inheriting it")
    root = re.search(r"<svg\b[^>]*>", text, re.I | re.S)
    root_tag = root.group(0) if root else ""
    has_role = re.search(r"""(?<![-\w])role\s*=\s*("img"|'img')""", root_tag)
    has_label = re.search(r"""(?<![-\w])aria-label\s*=\s*("|')""", root_tag)
    if not (has_role and has_label):
        rep.add("warning", "W303", path, 1,
                'give the root <svg> role="img" and a one-sentence aria-label')

    if None not in (vx, vy, vw, vh):  # the four are set together, or none of them are
        assert vx is not None and vy is not None and vw is not None and vh is not None
        for n, line in enumerate(lines, 1):
            for tag, attrs in ELEMENT.findall(line):
                if "transform=" in attrs:
                    continue  # a transform moves the element; naive bounds would lie
                a = {k.lower(): v for k, v in ATTR.findall(attrs)}
                try:
                    bounds = _element_box(tag.lower(), a)
                    if bounds is None:
                        continue
                    x0, y0, x1, y1 = bounds
                except (KeyError, ValueError, TypeError):
                    continue  # not a box this check can judge (a missing attribute, a transform)
                if x0 < vx - 1 or y0 < vy - 1 or x1 > vx + vw + 1 or y1 > vy + vh + 1:
                    rep.add("warning", "W304", path, n,
                            f"<{tag}> spans ({x0:g},{y0:g})-({x1:g},{y1:g}), outside the viewBox "
                            f"({vx:g},{vy:g})-({vx + vw:g},{vy + vh:g}): invisible content")

        # W311: a label is invisible past the edge exactly like a shape, but the source carries no text
        # metrics, so the width has to be estimated. W304 above judges only the anchor point, which is
        # how a caption clipped at the right edge of a figure - every check green, the page looked
        # fine at a glance - survived review. Measured against the class's own font-size, which these
        # figures set in CSS. An estimate, so it warns rather than fails the build on its own.
        metrics = _font_metrics(text)
        for n, line in enumerate(lines, 1):
            for element in TEXT_ELEMENT.finditer(line):
                if "transform" in element.group(1) or "textLength" in element.group(1):
                    continue  # moved or stretched: the estimate would lie
                body = re.sub(r"<[^>]+>", "", element.group(2)).strip()
                if not body:
                    continue
                attrs = {k.lower(): v for k, v in ATTR.findall(element.group(1))}
                try:
                    x, y = float(attrs["x"]), float(attrs["y"])
                except (KeyError, ValueError):
                    continue
                size, gap, anchor = metrics.get(
                    attrs.get("class", "").strip(), (TEXT_DEFAULT_SIZE, 0.0, "start"))
                anchor = attrs.get("text-anchor", anchor)
                width = _text_width(body, size, gap)
                if anchor == "middle":
                    x0, x1 = x - width / 2, x + width / 2
                elif anchor == "end":
                    x0, x1 = x - width, x
                else:
                    x0, x1 = x, x + width
                y0, y1 = y - size * 0.8, y + size * 0.25
                inside = (x0 >= vx - TEXT_EDGE_TOLERANCE and y0 >= vy - TEXT_EDGE_TOLERANCE
                          and x1 <= vx + vw + TEXT_EDGE_TOLERANCE
                          and y1 <= vy + vh + TEXT_EDGE_TOLERANCE)
                if not inside:
                    rep.add("warning", "W311", path, n,
                            f"<text> {body[:40]!r} needs about {width:.0f} units from x={x:g} "
                            f"(font-size {size:g}), i.e. ({x0:.0f},{y0:.0f})-({x1:.0f},{y1:.0f}), "
                            f"past the viewBox ({vx:g},{vy:g})-({vx + vw:g},{vy + vh:g}): the label is "
                            f"clipped or runs off the page")


def _element_box(tag: str, a: dict) -> tuple[float, float, float, float] | None:
    f = lambda k: float(a[k])  # noqa: E731
    if tag == "rect":
        x, y, w, h = f("x"), f("y"), f("width"), f("height")
        return x, y, x + w, y + h
    if tag == "circle":
        x, y, r = f("cx"), f("cy"), f("r")
        return x - r, y - r, x + r, y + r
    if tag == "ellipse":
        x, y, rx, ry = f("cx"), f("cy"), f("rx"), f("ry")
        return x - rx, y - ry, x + rx, y + ry
    if tag == "line":
        x1, y1, x2, y2 = f("x1"), f("y1"), f("x2"), f("y2")
        return min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)
    if tag == "text":
        # A text node's horizontal extent depends on the font and the anchor, so the anchor itself is
        # used as a proxy: this catches a label anchored past the right edge and, more usefully, a
        # caption row sitting below the bottom of the viewBox - both silently invisible.
        x, y = f("x"), f("y")
        return x, y, x, y
    if tag in ("polygon", "polyline"):
        nums = [float(n) for n in re.findall(r"-?\d+(?:\.\d+)?", a["points"])]
        if len(nums) < 2:
            return None
        xs, ys = nums[0::2], nums[1::2]
        return min(xs), min(ys), max(xs), max(ys)
    if tag == "path":
        # Absolute, pair-shaped commands only. A relative command (or H/V/A/Z, which do not come in
        # x,y pairs) makes a naive number sweep lie, so the element is skipped rather than guessed.
        d = a.get("d", "")
        if set(PATH_COMMANDS.findall(d)) - ABSOLUTE_PAIR_COMMANDS:
            return None
        nums = [float(n) for n in re.findall(r"-?\d+(?:\.\d+)?", d)]
        if len(nums) < 2 or len(nums) % 2:
            return None
        xs, ys = nums[0::2], nums[1::2]
        return min(xs), min(ys), max(xs), max(ys)
    return None


def _block_bounds(lines: list[str], idx: int) -> tuple[int, int]:
    """The CommonMark HTML block around the include line: back and forward to the blank lines.

    `<figure>` is a CommonMark block tag, so the wrapper may open on the include line or on any line
    above it inside the same block; a blank line ends the block either way.
    """
    start = idx
    while start > 0 and lines[start - 1].strip():
        start -= 1
    end = idx
    while end + 1 < len(lines) and lines[end + 1].strip():
        end += 1
    return start, end


def check_pages(pages: list[Path], figures: list[Path], rep: Report) -> None:
    included: set[str] = set()
    for page in pages:
        lines = page.read_text(encoding="utf-8", errors="replace").splitlines()
        numbers: list[tuple[int, int, int]] = []
        for n, line in enumerate(lines, 1):
            hit = SVG_INCLUDE.search(line)
            if hit:
                # mdBook allows `file:3`, `file:3:7` and `file:anchor`; the modifier is not a path.
                target = hit.group(1).strip().partition(":")[0] or hit.group(1).strip()
                resolved = (page.parent / target).resolve()
                if not resolved.exists():
                    rep.add("error", "E202", page, n, f"include target does not exist: {target}")
                else:
                    included.add(str(resolved))
                start, end = _block_bounds(lines, n - 1)
                if not any(FIGURE_OPEN.search(lines[j]) for j in range(start, end + 1)):
                    rep.add("error", "E201", page, n,
                            "the include is not wrapped in <figure>: <svg> is not a CommonMark block "
                            "tag, so the raw SVG is not one HTML block and mdBook will warn")
                caption = None
                for follow in lines[end + 1:]:
                    if follow.strip():
                        caption = follow
                        break
                cm = CAPTION.match(caption.strip()) if caption else None
                if not cm:
                    rep.add("error", "E203", page, n,
                            "no caption after the figure: the next non-blank line after the figure "
                            "block must start *Figure N.M - ...* with the takeaway")
                else:
                    chapter, index = int(cm.group(1)), int(cm.group(2))
                    numbers.append((n, chapter, index))
                    prefix = re.match(r"(\d+)", resolved.stem)
                    if prefix and int(prefix.group(1)) != chapter:
                        rep.add("warning", "W305", page, n,
                                f"caption says Figure {chapter}.{index} but the file is "
                                f"{resolved.name} - the prefix and the caption should agree")
            for im in MD_SVG_IMG.finditer(line):
                if im.group(1).startswith(("http://", "https://")):
                    continue
                rep.add("warning", "W306", page, n,
                        f"{im.group(1)} is referenced as an image: CSS custom properties do not "
                        "cross the image boundary, so this figure loses the theme - include it "
                        "inline instead")
            for im in HTML_SVG_IMG.finditer(line):
                ref = im.group(1) or im.group(2) or ""
                if ref.startswith(("http://", "https://")):
                    continue
                rep.add("warning", "W306", page, n,
                        f"{ref} is referenced as an image: CSS custom properties do not cross the "
                        "image boundary, so this figure loses the theme - include it inline instead")

        # Numbering is per chapter: a page may legitimately carry the tail of one chapter and the
        # head of the next, and each chapter restarts at 1.
        expected: dict[int, int] = {}
        for n, chapter, index in numbers:
            want = expected.get(chapter, 1)
            if index != want:
                rep.add("error", "E204", page, n,
                        f"caption numbering in Figure {chapter}.x: expected {chapter}.{want} at this "
                        f"position, found {chapter}.{index}")
            expected[chapter] = index + 1

    for fig in figures:
        if str(fig.resolve()) not in included:
            rep.add("warning", "W308", fig, 1, "nothing includes this figure")


def _find_css(root: Path, name: str) -> Path | None:
    """Locate the wrapper stylesheet anywhere sensible, without walking build output."""
    skip = {".git", "node_modules", "book", "target", "dist", "build"}
    for path in sorted(root.rglob(name)):
        if not skip.intersection(path.parts):
            return path
    return None


def check_css_wiring(book: dict, pages: list[Path], cfg: Config, root: Path, rep: Report) -> None:
    """W309: the wrapper CSS exists in the repo but is not in `additional-css`.

    Only fires when the named stylesheet is on disk, so a book that folds the wrapper styles into
    its own stylesheet is left alone; and only when some page actually uses the figure idiom.
    """
    if not book or not pages:
        return
    uses_figures = any(
        SVG_INCLUDE.search(page.read_text(encoding="utf-8", errors="replace")) for page in pages
    )
    if not uses_figures:
        return
    existing = _find_css(root, cfg.figure_css)
    if existing is None:
        return
    wired = [Path(entry).name for entry in book["additional_css"]]
    if cfg.figure_css not in wired:
        rep.add("warning", "W309", book["path"], 1,
                f"{existing.relative_to(root)} exists but {cfg.figure_css!r} is not listed in "
                "[output.html] additional-css: the .book-figure wrapper and caption styles never "
                "load (rename it with --figure-css if that is deliberate)")


def load_book(root: Path) -> dict | None:
    """Read `book.toml` for `book.src` and `[output.html] additional-css` (Python 3.11+)."""
    path = root / "book.toml"
    if tomllib is None or not path.is_file():
        return None
    try:
        data = tomllib.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:  # a broken book.toml is reported by mdBook, not here
        print(f"note: cannot parse {path}: {exc}", file=sys.stderr)
        return None
    book = data.get("book") if isinstance(data.get("book"), dict) else {}
    html = data.get("output", {}).get("html", {}) if isinstance(data.get("output"), dict) else {}
    return {
        "path": path,
        "src": str(book.get("src", "src")),
        "additional_css": list(html.get("additional-css", []) or []),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="lint figures and figure includes in an mdBook repo")
    ap.add_argument("paths", nargs="*", type=Path, help="SVG files, page .md files, or directories")
    ap.add_argument("--root", type=Path, default=Path("."), help="repo root (default: .)")
    ap.add_argument("--figures", type=Path, help="figures dir (default: <book src>/figures)")
    ap.add_argument("--pages", type=Path, help="pages dir (default: <book src>)")
    ap.add_argument("--max-width", type=float, default=820,
                    help="viewBox width to warn above (default 820, the documented ceiling)")
    ap.add_argument("--accent-prefix", default="fig-a-",
                    help="class prefix whose CSS rule may carry a literal colour")
    ap.add_argument("--figure-css", default="figures.css",
                    help="wrapper stylesheet expected in additional-css (default figures.css)")
    ap.add_argument("--allow-chars", default=DEFAULT_ALLOW_CHARS,
                    help="non-ASCII characters a figure may carry (default: the typographic+notation set)")
    ap.add_argument("--show-allow-chars", action="store_true",
                    help="print the effective allowlist as U+ codes and a copy-paste option, then exit")
    ap.add_argument("--json", action="store_true", help="machine-readable output")
    ap.add_argument("--strict", action="store_true", help="exit 1 on warnings too")
    ap.add_argument("--quiet", action="store_true", help="summary line only")
    ap.add_argument("--version", action="version", version=f"figcheck {VERSION}")
    args = ap.parse_args()

    cfg = Config(args)

    if args.show_allow_chars:
        chars = sorted(cfg.allow_chars)
        for c in chars:
            print(f"U+{ord(c):04X}  {c!r}")
        print(f"{len(chars)} allowed character(s)")
        print("--allow-chars " + '"' + "".join(f"\\u{ord(c):04x}" for c in chars) + '"')
        return 0

    rep = Report()

    figs: list[Path] = []
    pages: list[Path] = []
    if args.paths:
        for p in args.paths:
            if p.is_file() and p.suffix.lower() == ".svg":
                figs.append(p)
            elif p.is_file() and p.suffix.lower() == ".md":
                pages.append(p)
            elif p.is_dir():
                figs.extend(sorted(p.glob("*.svg")))
                pages.extend(sorted(p.rglob("*.md")))
            else:
                print(f"skip: {p} is neither an SVG, a .md page, nor a directory", file=sys.stderr)
        book = None
    else:
        book = load_book(args.root)
        src = Path(book["src"]) if book else Path("src")
        figures_dir = args.figures or args.root / src / "figures"
        pages_dir = args.pages or args.root / src
        figs = sorted(figures_dir.glob("*.svg")) if figures_dir.is_dir() else []
        pages = sorted(pages_dir.rglob("*.md")) if pages_dir.is_dir() else []
        if not figs and not pages:
            where = f"{figures_dir} and {pages_dir}" if book else f"{args.root.resolve()}"
            print(f"no figures or pages found under {where}", file=sys.stderr)
            return 1

    for fig in figs:
        check_svg(fig, cfg, rep)
    if figs and pages:
        check_pages(pages, figs, rep)
    if book:
        check_css_wiring(book, pages, cfg, args.root, rep)

    errors, warnings = rep.errors(), rep.warnings()
    if args.json:
        print(json.dumps({
            "summary": {"figures": len(figs), "pages": len(pages),
                        "errors": len(errors), "warnings": len(warnings)},
            "errors": errors,
            "warnings": warnings,
        }, indent=2))
    else:
        if not args.quiet:
            for f in rep.findings:
                where = f"{f['file']}:{f['line']}" if f["line"] else f["file"]
                print(f"{where}: {'error' if f['level'] == 'error' else 'warning'} "
                      f"[{f['code']}] {f['message']}")
        notes = []
        if book:
            notes.append(f"book.toml src={book['src']}")
        if not pages:
            notes.append("orphan check skipped: no pages")
        suffix = f" ({'; '.join(notes)})" if notes else ""
        print(f"{len(figs)} figure(s), {len(pages)} page(s): "
              f"{len(errors)} error(s), {len(warnings)} warning(s){suffix}")

    return 1 if errors or (args.strict and warnings) else 0


if __name__ == "__main__":
    sys.exit(main())
