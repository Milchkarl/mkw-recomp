"""Vietnamese letters for the game's fonts (game_fonts.py), drawn from each font's own letters and marks."""

import unicodedata

from glyph_shapes import IDENTITY, Shape, box, fit, outline

# Vietnamese (#873): its letters with marks, which the game's fonts lack (all but those of Latin-1), drawn from the base
# letter and the font's own marks: the grave, acute, circumflex and tilde of à á â ã (À Á Â Ã for capitals), the
# breve, and the dot of i for the dot below. The hook above and the horn have no glyph in these fonts: the hook is the
# top of the font's "?" and the horn its "’", scaled. A tone mark over the circumflex goes beside it, at its right, as
# Vietnamese type sets ấ ầ ẩ, so the letter grows no taller; over the breve it stacks. Đ is the font's Ð (the same
# letter), đ the d with the font's hyphen as its bar.
GRAVE, ACUTE, CIRCUMFLEX, TILDE, BREVE, HOOK, HORN, DOT_BELOW = "̀", "́", "̂", "̃", "̆", "̉", "̛", "̣"  # fmt: skip
ABOVE_MARKS = (GRAVE, ACUTE, CIRCUMFLEX, TILDE, BREVE, HOOK)
FROM_LETTER = {GRAVE: "àÀ", ACUTE: "áÁ", CIRCUMFLEX: "âÂ", TILDE: "ãÃ"}  # (lower, upper), over a and A


class Marks:
    """A font's mark shapes for lower or upper case, each with the gap the font leaves between a letter and it."""

    def __init__(self, glyphs, cmap: dict[int, str], upper: bool):
        self.shapes: dict[str, Shape] = {}
        a = "A" if upper else "a"
        if ord(a) not in cmap:
            return
        top = box(outline(glyphs, cmap[ord(a)]))[3]
        for mark, letters in FROM_LETTER.items():
            if ord(letters[upper]) in cmap:
                shape = [c for c in outline(glyphs, cmap[ord(letters[upper])]) if c[0][1] > top - 5]
                if shape:
                    self.shapes[mark] = shape
        if not self.shapes.get(ACUTE):
            return
        self.gap = box(self.shapes[ACUTE])[1] - top
        height = box(self.shapes[ACUTE])[3] - box(self.shapes[ACUTE])[1]
        for cp in (0x306, 0x2D8):  # the combining breve, else the spacing one
            if cp in cmap:
                self.shapes[BREVE] = outline(glyphs, cmap[cp])
                break
        if ord("?") in cmap:  # the hook: "?" without its dot, as tall as the acute
            q = outline(glyphs, cmap[ord("?")])
            hook = [c for c in q if c[0][1] > min(b[1] for b, _ in q) + 1]
            if hook:
                self.shapes[HOOK] = hook
                self.hook_scale = height / (box(hook)[3] - box(hook)[1])
        # the horn: the closing quote, the apostrophe or the comma, whichever is one stroke (NinDev's quote has two)
        horns = [outline(glyphs, cmap[cp]) for cp in (0x2019, 0x27, 0x2C) if cp in cmap]
        if horns:
            self.shapes[HORN] = min(horns, key=len)
        if ord("i") in cmap:  # the dot below: the dot of i (its highest contour)
            dots = outline(glyphs, cmap[ord("i")])
            self.shapes[DOT_BELOW] = [max(dots, key=lambda c: c[0][1])]

    def scale(self, mark: str) -> float:
        return self.hook_scale if mark == HOOK else 1.0


def vietnamese(glyphs, cmap: dict[int, str], ch: str) -> tuple[str, list[tuple[Shape, tuple]], float] | None:
    """ch drawn from its base letter and marks: (the glyph its metrics come from, its parts, its advance width)."""
    nfd = unicodedata.normalize("NFD", ch)
    base, marks = nfd[0], nfd[1:]
    if ch == "đ":
        base, marks = "d", ""
        if ord(base) not in cmap:
            return None
    elif not marks or any(m not in ABOVE_MARKS + (HORN, DOT_BELOW) for m in marks) or ord(base) not in cmap:
        return None
    upper = base.isupper()
    font_marks = Marks(glyphs, cmap, upper)
    if not hasattr(font_marks, "gap"):
        return None
    base_box = box(outline(glyphs, cmap[ord(base)]))
    # start from the longest precomposed part the font has (â for ấ), else the base (ı for an i with a mark above)
    start, rest, above = base, marks, []
    for k in range(len(marks), 0, -1):
        pre = unicodedata.normalize("NFC", base + marks[:k])
        if len(pre) == 1 and ord(pre) in cmap:
            start, rest = pre, marks[k:]
            over = [c for c in outline(glyphs, cmap[ord(pre)]) if c[0][1] > base_box[3] - 5]
            above = [(m, box(over)) for m in marks[:k] if m in ABOVE_MARKS and over][-1:]
            break
    if start == "i" and any(m in ABOVE_MARKS for m in rest) and 0x131 in cmap:
        start = "ı"  # the marks sit where the dot was
        base_box = box(outline(glyphs, cmap[0x131]))
    name = cmap[ord(start)]
    width = glyphs[name].width
    parts = [(outline(glyphs, name), IDENTITY)]
    cx = (base_box[0] + base_box[2]) / 2
    right = base_box[2]
    if ch == "đ":  # the bar: the hyphen across the top of the stem, halfway between the x-height and the ascender
        if ord("-") not in cmap or ord("x") not in cmap:
            return None
        bar = outline(glyphs, cmap[ord("-")])
        hb, xh = box(bar), box(outline(glyphs, cmap[ord("x")]))[3]
        w = base_box[2] - base_box[0]
        sx, sy = 0.62 * w / (hb[2] - hb[0]), 0.8
        y = (xh + base_box[3]) / 2 - (hb[3] - hb[1]) * sy / 2
        m = (sx, 0, 0, sy, base_box[2] + 0.08 * w - hb[2] * sx, y - hb[1] * sy)
        return name, [*parts, (bar, m)], width
    for mark in rest:
        shape = font_marks.shapes.get(mark)
        if not shape:
            return None
        if mark == HORN:  # at the top right of o or u, rising past it
            acute, horn = box(font_marks.shapes[ACUTE]), box(shape)
            s = (acute[3] - acute[1]) / (horn[3] - horn[1])  # as tall as the acute
            w, h = base_box[2] - base_box[0], base_box[3] - base_box[1]
            m, b = fit(shape, s, x0=base_box[2] - 0.08 * w, y1=base_box[3] + 0.12 * h)
            right = max(right, b[2])
        elif mark == DOT_BELOW:  # under the baseline; on y beside its tail, at the right, as Vietnamese type sets ỵ
            w = base_box[2] - base_box[0]
            m, b = fit(shape, 1.0, cx=cx if base != "y" else base_box[0] + 0.8 * w, y1=-font_marks.gap)
        elif not above:
            m, b = fit(shape, font_marks.scale(mark), cx=cx, y0=base_box[3] + font_marks.gap)
        elif above[-1][0] == CIRCUMFLEX and mark in (ACUTE, GRAVE, HOOK):  # beside the circumflex, at its right
            prev = above[-1][1]
            s = 0.8 * font_marks.scale(mark)
            m, b = fit(shape, s, x0=prev[2] - 0.12 * (prev[2] - prev[0]), y0=prev[1] + 0.45 * (prev[3] - prev[1]))
            right = max(right, b[2])
        else:  # over the mark below it (the tone over a breve, the tilde over a circumflex)
            prev = above[-1][1]
            m, b = fit(shape, font_marks.scale(mark), cx=cx, y0=prev[3] + 0.35 * font_marks.gap)
        if mark in ABOVE_MARKS:
            above.append((mark, b))
        parts.append((shape, m))
    width = max(width, round(right + (width - base_box[2])))  # a horn or a side mark past the letter widens it
    return name, parts, width
