"""Arabic letters for the game's fonts (game_fonts.py), taken from Noto Sans Arabic ExtraBold.

The game's fonts have no Arabic at all, so unlike the Turkish, Vietnamese and Ukrainian letters these cannot be made
from their own pieces: each joined form arabic.py writes (Arabic Presentation Forms-B) and the Arabic punctuation are
copied from runtime/ui/fonts/NotoSansArabic-ExtraBold.ttf (SIL OFL 1.1, tools/i18n/arabic_font.py), scaled so that
Noto's Latin cap height (714 of 1000) matches the game font's H.
"""

from functools import cache
from pathlib import Path

from glyph_shapes import Shape, box, outline

FONT = Path(__file__).resolve().parents[2] / "runtime" / "ui" / "fonts" / "NotoSansArabic-ExtraBold.ttf"
NOTO_CAP = 714  # Noto Sans's cap height, which its Arabic is drawn to match


@cache
def noto():
    from fontTools.ttLib import TTFont

    font = TTFont(FONT)
    return font.getGlyphSet(), font.getBestCmap(), font["head"].unitsPerEm


def arabic(glyphs, cmap: dict[int, str], ch: str) -> tuple[str, list[tuple[Shape, tuple]], float] | None:
    """ch from Noto, scaled to the game font: (the glyph its font dict comes from, its parts, its advance width)."""
    src_glyphs, src_cmap, src_upm = noto()
    if ord(ch) not in src_cmap or ord("H") not in cmap or ord("a") not in cmap:
        return None
    name = src_cmap[ord(ch)]
    shape = outline(src_glyphs, name)
    cap = box(outline(glyphs, cmap[ord("H")]))[3]
    s = cap / (NOTO_CAP * src_upm / 1000)
    width = round(src_glyphs[name].width * s)
    return cmap[ord("a")], [(shape, (s, 0, 0, s, 0, 0))] if shape else [], width
