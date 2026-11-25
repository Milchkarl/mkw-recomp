"""Hindi syllables for the game's fonts (game_fonts.py), from runtime/ui/fonts/NotoSansDevanagari-Syllables-ExtraBold.ttf.

The game's fonts have no Devanagari, and the game draws one glyph per character without shaping: so each syllable our
texts use is one private-use character (devanagari.py), drawn already shaped in that font (devanagari_font.py, Noto
Sans Devanagari, SIL OFL 1.1). Its glyph is copied here, scaled so that the font's H matches the game font's H (Noto
draws its Latin to sit with its Devanagari).
"""

from functools import cache
from pathlib import Path

from glyph_shapes import Shape, box, outline

FONT = Path(__file__).resolve().parents[2] / "runtime" / "ui" / "fonts" / "NotoSansDevanagari-Syllables-ExtraBold.ttf"


@cache
def noto():
    from fontTools.ttLib import TTFont

    font = TTFont(FONT)
    glyphs, cmap = font.getGlyphSet(), font.getBestCmap()
    return glyphs, cmap, box(outline(glyphs, cmap[ord("H")]))[3]


def hindi(glyphs, cmap: dict[int, str], ch: str) -> tuple[str, list[tuple[Shape, tuple]], float] | None:
    """A syllable's character from the syllable font, scaled to the game font: (the glyph its font dict comes from,
    its parts, its advance width)."""
    if not 0xF100 <= ord(ch) <= 0xF7FF or ord("H") not in cmap or ord("a") not in cmap:
        return None
    src_glyphs, src_cmap, src_cap = noto()
    if ord(ch) not in src_cmap:
        return None
    name = src_cmap[ord(ch)]
    shape = outline(src_glyphs, name)
    s = box(outline(glyphs, cmap[ord("H")]))[3] / src_cap
    return cmap[ord("a")], [(shape, (s, 0, 0, s, 0, 0))] if shape else [], round(src_glyphs[name].width * s)
