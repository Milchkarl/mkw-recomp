# /// script
# requires-python = ">=3.14"
# dependencies = ["fonttools>=4.60", "uharfbuzz>=0.50", "polib>=1.2"]
# ///
"""The Hindi syllables as glyphs: i18n/syllables/hi.tsv and runtime/ui/fonts/NotoSansDevanagari-Syllables-*.ttf.

Why: neither the game nor ImGui shapes text (devanagari.py). So every syllable our Hindi texts use (i18n/ui/hi.po,
i18n/game/base/hi.tsv, i18n/game/1.8.0/hi.tsv and the language's own name) is shaped here once with HarfBuzz on Noto
Sans Devanagari (SIL OFL 1.1, runtime/ui/fonts/LICENSE-NotoSansDevanagari.txt; Google Fonts' variable font at a
pinned commit), its glyphs drawn at their shaped places as one glyph at a private-use code point, and the table
written. The menu embeds the Medium font (beside its Roboto Medium); the setup copies the ExtraBold glyphs into the
game's fonts (game_fonts_hi.py, beside Rodin EB). Both carry H too, for the scale. Rerun after changing a Hindi text:
the build and the setup stop on a syllable the table lacks.

Usage:
  uv run tools/i18n/devanagari_font.py         rewrite the table and both fonts (downloads the pinned font into build/fonts/)
"""

import hashlib
import io
import sys
import urllib.request
from pathlib import Path

import polib
import uharfbuzz as hb
from fontTools.fontBuilder import FontBuilder
from fontTools.pens.recordingPen import DecomposingRecordingPen
from fontTools.pens.transformPen import TransformPen
from fontTools.pens.ttGlyphPen import TTGlyphPen
from fontTools.ttLib import TTFont
from fontTools.varLib import instancer

sys.path.insert(0, str(Path(__file__).resolve().parent))
import devanagari  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
COMMIT = (
    "2bc5d431b584d89e646214bf6cfa75494e76ceea"  # google/fonts, the last change of ofl/notosansdevanagari (09/09/2026)
)
URL = f"https://raw.githubusercontent.com/google/fonts/{COMMIT}/ofl/notosansdevanagari/NotoSansDevanagari%5Bwdth,wght%5D.ttf"
SHA256 = "14ec4af41f27482216d1c2229f417ff9b1425e1babb014e57d1d40d03229853e"
WEIGHTS = {"Medium": 500, "ExtraBold": 800}
NAME = "हिन्दी"  # the language's name in the menus' language lists (runtime/gfx/nvn/host/ui_text_cat.h)


def source() -> Path:
    path = ROOT / "build" / "fonts" / "NotoSansDevanagari-VF.ttf"
    if not path.is_file():
        path.parent.mkdir(parents=True, exist_ok=True)
        with urllib.request.urlopen(URL, timeout=60) as r:
            path.write_bytes(r.read())
    if hashlib.sha256(path.read_bytes()).hexdigest() != SHA256:
        sys.exit(f"{path}: not the pinned Noto Sans Devanagari (sha256 {SHA256})")
    return path


def texts() -> list[str]:
    out = [NAME]
    for tsv in (ROOT / "i18n/game/base/hi.tsv", ROOT / "i18n/game/1.8.0/hi.tsv"):
        for line in tsv.read_text(encoding="utf-8").splitlines():
            if "\t" in line and not line.startswith("#"):
                out.append(line.split("\t", 1)[1])
    out += [e.msgstr for e in polib.pofile(str(ROOT / "i18n/ui/hi.po")) if e.msgstr]
    return out


def shape(font: hb.Font, run: str) -> list[tuple[str, list[tuple[int, int, int]], int]]:
    """A Devanagari run as HarfBuzz's clusters: (its text, its glyphs (id, x, y) from its start, its advance)."""
    buf = hb.Buffer()
    buf.add_codepoints([ord(c) for c in run])
    buf.guess_segment_properties()
    hb.shape(font, buf, {})
    starts = sorted({info.cluster for info in buf.glyph_infos}) + [len(run)]
    out = []
    for start, end in zip(starts, starts[1:]):
        glyphs, pen = [], 0
        for info, pos in zip(buf.glyph_infos, buf.glyph_positions):
            if info.cluster == start:
                glyphs.append((info.codepoint, pen + pos.x_offset, pos.y_offset))
                pen += pos.x_advance
        out.append((run[start:end], glyphs, pen))
    return out


def instance(src: Path, weight: int) -> tuple[TTFont, hb.Font]:
    font = instancer.instantiateVariableFont(TTFont(src), {"wght": weight, "wdth": 100})
    buf = io.BytesIO()
    font.save(buf)
    return TTFont(io.BytesIO(buf.getvalue())), hb.Font(hb.Face(buf.getvalue()))


def draw(glyphs, name: str, pen) -> None:
    """A glyph into pen with its components drawn out (the new font has none of the glyphs they name)."""
    rec = DecomposingRecordingPen(glyphs)
    glyphs[name].draw(rec)
    rec.replay(pen)


def write_font(font: TTFont, clusters: dict[str, tuple[list, int]], codes: dict[str, int], out: Path) -> None:
    names = font.getGlyphOrder()
    glyphs = font.getGlyphSet()
    order, outlines, metrics, cmap = [".notdef", "H"], {}, {}, {ord("H"): "H"}
    for name, src in ((".notdef", ".notdef"), ("H", font.getBestCmap()[ord("H")])):
        pen = TTGlyphPen(None)
        draw(glyphs, src, pen)
        outlines[name] = pen.glyph()
        metrics[name] = font["hmtx"][src]
    for syllable, cp in codes.items():
        name = f"uni{cp:04X}"
        parts, advance = clusters[syllable]
        pen = TTGlyphPen(None)
        for gid, x, y in parts:
            draw(glyphs, names[gid], TransformPen(pen, (1, 0, 0, 1, x, y)))
        order.append(name)
        outlines[name] = pen.glyph()
        cmap[cp] = name
        metrics[name] = (advance, 0)
    fb = FontBuilder(font["head"].unitsPerEm, isTTF=True)
    fb.setupGlyphOrder(order)
    fb.setupCharacterMap(cmap)
    fb.setupGlyf(outlines)
    for name, (adv, _) in metrics.items():  # left side bearings from the outlines just built
        metrics[name] = (adv, getattr(outlines[name], "xMin", 0) if outlines[name].numberOfContours else 0)
    fb.setupHorizontalMetrics(metrics)
    fb.setupHorizontalHeader(ascent=font["hhea"].ascent, descent=font["hhea"].descent)
    family = f"Noto Sans Devanagari Syllables {out.stem.rsplit('-', 1)[1]}"
    fb.setupNameTable({"familyName": family, "styleName": "Regular",
                       "copyright": font["name"].getDebugName(0), "licenseDescription": font["name"].getDebugName(13)})  # fmt: skip
    os2 = font["OS/2"]
    fb.setupOS2(sTypoAscender=os2.sTypoAscender, sTypoDescender=os2.sTypoDescender, usWinAscent=os2.usWinAscent,
                usWinDescent=os2.usWinDescent, sCapHeight=os2.sCapHeight, sxHeight=os2.sxHeight, usWeightClass=os2.usWeightClass)  # fmt: skip
    fb.setupPost()
    fb.save(out)
    print(f"{out.relative_to(ROOT)}: {out.stat().st_size} bytes, {len(order)} glyphs")


def main() -> int:
    src = source()
    fonts = {name: instance(src, weight) for name, weight in WEIGHTS.items()}
    cuts: dict[str, list[str]] = {}
    shaped: dict[str, dict[str, tuple[list, int]]] = {name: {} for name in WEIGHTS}
    for text in texts():
        for deva, run in devanagari.runs(text):
            if not deva or run in cuts:
                continue
            for name, (_, hbfont) in fonts.items():
                clusters = shape(hbfont, run)
                cuts.setdefault(run, [c[0] for c in clusters])
                if [c[0] for c in clusters] != cuts[run]:
                    sys.exit(f"{run!r}: the weights cut it differently")
                for syllable, parts, advance in clusters:
                    shaped[name].setdefault(syllable, (parts, advance))
    syllables = sorted({s for cut in cuts.values() for s in cut})
    if len(syllables) > devanagari.LAST - devanagari.FIRST + 1:
        sys.exit(f"{len(syllables)} syllables: more than U+{devanagari.FIRST:04X}-{devanagari.LAST:04X} holds")
    codes = {s: devanagari.FIRST + i for i, s in enumerate(syllables)}
    lines = [
        "# Each Hindi syllable our texts use and its private-use code point (tools/i18n/devanagari_font.py writes it)"
    ]
    devanagari.TABLE.write_text(
        "\n".join(lines + [f"{s}\t{cp:04X}" for s, cp in codes.items()]) + "\n", encoding="utf-8"
    )
    devanagari.table.cache_clear()
    wrong = [run for run, cut in cuts.items() if devanagari.syllables(run) != cut]
    if wrong:
        sys.exit(f"{len(wrong)} runs cut otherwise by the longest-first table, e.g. {wrong[:3]}")
    print(f"{devanagari.TABLE.relative_to(ROOT)}: {len(syllables)} syllables from {len(cuts)} words")
    for name, (font, _) in fonts.items():
        write_font(font, shaped[name], codes, ROOT / "runtime/ui/fonts" / f"NotoSansDevanagari-Syllables-{name}.ttf")
    return 0


if __name__ == "__main__":
    sys.exit(main())
