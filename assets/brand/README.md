# minfer brand

**Four ops, one step.** The mark is the compute graph folded to an icon: four operator
nodes on the axes, four edges, one filled hub — the step that reads them. The tile is the
mark's container: an amber node, with the graph in ink.

## The mark

- 32-unit grid; four `7`-unit nodes with a `2.9` outline, centred at `(16,5)`, `(27,16)`,
  `(16,27)`, `(5,16)`; four `1.9` spokes; a `2.8` hub. The ink box `0.05 – 31.95` is the
  whole grid.
- Tile: radius `22.5%` (`7.2`), amber `#E0A45C`, glyph in ink `#17171A` at `66%` of the
  tile. The favicon uses a slightly bolder glyph (`70%`, `3.4` outline) so the node
  holes survive a 16 px tab.

## The wordmark

`minfer` set in **Inter SemiBold**, lowercase, tracking `-0.018 em`. One change: the i
tittle is replaced by a `r = 150/2048 em` circle at the glyph's own centre, so the
wordmark carries the same node as the mark. The paths are outlined and ship as SVG; no
font file is distributed. Inter is licensed under the SIL Open Font License 1.1.

Lockup geometry: tile at `1.3×` the wordmark size, gap `0.42×`, tile lifted `1.5 units`
against the descender line.

## Colour

| Token | Hex | Use |
|---|---|---|
| Ink | `#17171A` | glyph on the tile, wordmark on paper |
| Paper | `#FBF9F4` | wordmark on dark, tile border on dark |
| Amber | `#E0A45C` | the tile, the hub on dark |
| Copper | `#A85A28` | the hub on paper (amber is too light there) |

## Files

| File | Use |
|---|---|
| `minfer-icon.svg` / `-512.png` / `-256.png` / `-128.png` | app icon, avatar, README |
| `minfer-favicon.svg` / `-32.png` | browser tab; also served by `minfer viz` |
| `minfer-logo-dark.svg` | lockup for dark backgrounds |
| `minfer-logo-light.svg` | lockup for light backgrounds |
| `minfer-mark.svg` | the mark alone, `currentColor` (inline UI) |
| `minfer-mark-dark.svg` / `-light.svg` | the mark alone, two-tone |
| `minfer-wordmark-dark.svg` / `-light.svg` | type only |
| `preview.html` | the brand sheet — open it in a browser |
| `preview.png` | the same sheet as an image |
| `minfer-social.png` | GitHub social preview (`1200×630`) |

## Rules

- Never rotate, skew, re-draw or re-colour the mark. The tile stays amber with an ink
  glyph; the free mark takes an amber hub on dark and a copper hub on paper.
- Clear space: at least one node diameter on every side; for the lockup, the tile's
  corner radius.
- Minimum sizes: icon `16 px`, free mark `20 px`, lockup `120 px` wide. Below those the
  node holes stop reading and the hub plus four spokes still carry the mark.
- The wordmark is never re-typeset — use the SVG, which already carries the circular
  tittle.
