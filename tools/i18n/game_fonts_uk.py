"""Ukrainian letters for the game's fonts (game_fonts.py), drawn from each font's own letters.

The game's European fonts have Russian's Cyrillic but not Ukrainian's ґ є і ї (and capitals) nor the apostrophe ʼ
(measured 25/11/2026: all but nintendoP_RodinNTLG-EB lack them). і ї are the same shapes as Latin i ï, ʼ is ’: the
font's own glyph serves. є is э mirrored (the same arc and bar, open the other way), and ґ is г with an upturn: the
stem of ı (I for Ґ) shortened, rising from the right end of its bar.
"""

from glyph_shapes import IDENTITY, Shape, box, outline

SAME = {"і": "i", "І": "I", "ї": "ï", "Ї": "Ï", "ʼ": "’"}  # the Latin or punctuation glyph that is the same letter
MIRRORED = {"є": "э", "Є": "Э"}
UPTURN = {"ґ": ("г", "ı"), "Ґ": ("Г", "I")}  # letter -> (base, the stem the upturn is cut from)


def ukrainian(glyphs, cmap: dict[int, str], ch: str) -> tuple[str, list[tuple[Shape, tuple]], float] | None:
    """ch drawn from the font's letters: (the glyph its metrics come from, its parts, its advance width)."""
    if ch in MIRRORED and ord(MIRRORED[ch]) in cmap:
        name = cmap[ord(MIRRORED[ch])]
        shape = outline(glyphs, name)
        b = box(shape)
        return name, [(shape, (-1, 0, 0, 1, b[0] + b[2], 0))], glyphs[name].width  # flipped within its own box
    if ch in UPTURN and all(ord(c) in cmap for c in UPTURN[ch]):
        name, stem_name = (cmap[ord(c)] for c in UPTURN[ch])
        shape, stem = outline(glyphs, name), outline(glyphs, stem_name)
        b, s = box(shape), box(stem)
        w = s[2] - s[0]  # the stem's width is the bar's thickness too
        rise = (0.32 if ch.islower() else 0.22) * (b[3] - b[1])  # above the bar, as Ukrainian type draws it
        sy = (rise + w) / (s[3] - s[1])  # from inside the bar (one stroke deep) to the top of the upturn
        m = (1, 0, 0, sy, b[2] - s[2], b[3] + rise - s[3] * sy)
        return name, [(shape, IDENTITY), (stem, m)], glyphs[name].width
    return None
