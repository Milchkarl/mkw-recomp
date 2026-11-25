"""Arabic text as the game can draw it: joined letters and each line in visual order.

Why: the game draws text one character at a time (nn::fontll::ScalableFontEngine: AcquireGlyphmap per code point,
GPOS kerning only; no shaping, no right-to-left). So each Arabic letter is replaced here by its joined form (the
Arabic Presentation Forms-B code points, whose glyphs game_fonts_ar.py adds to the fonts) and each line is written in
the order it is seen, left to right. The menu does the same: catalog.py stores the joined letters in reading order
and the ImGui text renderer reorders each line after wrapping it (runtime/ui/menu/ui_bidi.cpp, the same rules as
visual() here).

The joined forms come from Python's Unicode data (each U+FE70-FEFC is "<initial> 0628" and so on), so no table is
written by hand. The order follows the Unicode bidirectional algorithm reduced to what these texts have: a
right-to-left paragraph, Latin words and numbers kept left to right inside it, brackets mirrored.

Usage (a check, the build imports it):
  uv run tools/i18n/arabic.py 'نص عربي 150cc'      the text joined and in visual order
"""

import re
import sys
import unicodedata

FORM_NAMES = ("isolated", "final", "initial", "medial")
LAM, ALEFS = "ل", "آأإا"
TATWEEL = "ـ"


def _forms() -> tuple[dict[str, dict[str, str]], dict[str, dict[str, str]]]:
    """letter -> {form: presentation char}, and lam + alef pair -> {isolated|final: ligature}."""
    letters: dict[str, dict[str, str]] = {}
    pairs: dict[str, dict[str, str]] = {}
    for cp in range(0xFE70, 0xFEFD):
        dec = unicodedata.decomposition(chr(cp)).split()
        if not dec or dec[0].strip("<>") not in FORM_NAMES:
            continue
        form, base = dec[0].strip("<>"), "".join(chr(int(x, 16)) for x in dec[1:])
        (pairs if len(base) == 2 else letters).setdefault(base, {})[form] = chr(cp)
    return letters, pairs


LETTERS, LAM_ALEF = _forms()
LETTERS.setdefault(TATWEEL, {"isolated": TATWEEL, "final": TATWEEL, "initial": TATWEEL, "medial": TATWEEL})


def _joins_next(ch: str) -> bool:
    return "initial" in LETTERS.get(ch, {})


def _joins_prev(ch: str) -> bool:
    return "final" in LETTERS.get(ch, {})


def _transparent(ch: str) -> bool:
    return unicodedata.category(ch) == "Mn"


def shape(text: str) -> str:
    """Each Arabic letter as its joined form (logical order kept); lam + alef as their ligature."""
    out = []
    chars = list(text)
    i = 0
    while i < len(chars):
        ch = chars[i]
        if ch not in LETTERS:
            out.append(ch)
            i += 1
            continue
        prev = next((c for c in reversed(chars[:i]) if not _transparent(c)), "")
        after = [c for c in chars[i + 1 :] if not _transparent(c)]
        joined_before = _joins_next(prev) and _joins_prev(ch)
        if ch == LAM and after and after[0] in ALEFS:  # lam-alef: one glyph, joins only before it
            k = chars.index(after[0], i + 1)
            out.append(LAM_ALEF[ch + after[0]]["final" if joined_before else "isolated"])
            out += chars[i + 1 : k]  # marks between them stay
            i = k + 1
            continue
        joined_after = bool(after) and _joins_next(ch) and _joins_prev(after[0])
        form = {(False, False): "isolated", (True, False): "final", (False, True): "initial", (True, True): "medial"}
        forms = LETTERS[ch]
        out.append(forms.get(form[joined_before, joined_after]) or forms["isolated"])
        i += 1
    return "".join(out)


R, L, EN, N = "R", "L", "EN", "N"
MIRROR = str.maketrans("()[]<>«»", ")(][><»«")


def kind(ch: str) -> str:
    """The bidi class these texts need: R (Arabic), L (other letters), EN (digits, an inserted value), N (the rest)."""
    if ch == "￼" or ch.isdigit():
        return EN
    bidi = unicodedata.bidirectional(ch)
    if bidi in ("R", "AL"):
        return R
    if bidi == "L":
        return L
    if bidi == "AN":
        return EN
    return N


def levels(kinds: list[str]) -> list[int]:
    """1 for right-to-left, 2 for a left-to-right run (letters and numbers with what lies between them)."""
    lv = [1 if k == R else 2 if k in (L, EN) else 0 for k in kinds]
    for i, v in enumerate(lv):
        if v:
            continue
        before = next((x for x in reversed(lv[:i]) if x), 1)
        after = next((x for x in lv[i + 1 :] if x), 1)
        lv[i] = 2 if before == after == 2 else 1
    return lv


def reorder(items: list, kinds: list[str]) -> list:
    """items in visual order: each left-to-right run kept, the line reversed."""
    lv = levels(kinds)
    order = list(range(len(items)))
    i = 0
    while i < len(order):  # reverse each run at level 2, then the whole line
        if lv[order[i]] == 2:
            j = i
            while j < len(order) and lv[order[j]] == 2:
                j += 1
            order[i:j] = order[i:j][::-1]
            i = j
        else:
            i += 1
    order.reverse()
    return [items[k] for k in order]


def visual_line(text: str) -> str:
    """A line without tags in visual order, with brackets mirrored in its right-to-left parts."""
    kinds = [kind(c) for c in text]
    lv = levels(kinds)
    chars = [c.translate(MIRROR) if lvl == 1 else c for c, lvl in zip(text, lv)]
    return "".join(reorder(chars, kinds))


# game_texts.py's text: {0E ...} / {0F ...} tags and \n \t \\ escapes. Tags of group 0 type 1-3 (font, size, colour)
# and group 0x12D (a start and an end) apply to the text after them: they are carried per character and written back
# where that state changes in visual order. Any other tag inserts a value (a number, a name): it moves as one
# left-to-right character.
TOKEN = re.compile(r"\{0[EF] [0-9A-F ]+\}|\\.|.", re.S)


def _state_key(tag: str) -> str | None:
    parts = tag.strip("{}").split()
    if parts[1] == "0000" and parts[2] in ("0001", "0002", "0003"):
        return parts[2]
    if parts[1] == "012D":
        return "012D"
    return None


def visual(text: str) -> str:
    """game_texts.py's text with its Arabic joined and each line in visual order, its tags still applying to the
    same characters."""
    lines: list[list[tuple[str, dict[str, str]]]] = [[]]
    state: dict[str, str] = {}
    for tok in TOKEN.findall(shape_escaped(text)):
        if tok == "\\n":
            lines.append([])
        elif tok.startswith("{"):
            key = _state_key(tok)
            if key is None:
                lines[-1].append((tok, dict(state)))
            elif tok.startswith("{0F"):
                state.pop(key, None)
            else:
                state[key] = tok
        else:
            lines[-1].append((tok, dict(state)))
    end_state = state
    out, shown = [], {}
    for n, line in enumerate(lines):
        if n:
            out.append("\\n")
        kinds = [EN if t.startswith("{") else N if t.startswith("\\") else kind(t) for t, _ in line]
        lv = levels(kinds)
        items = [(t.translate(MIRROR) if lvl == 1 and len(t) == 1 else t, s) for (t, s), lvl in zip(line, lv)]
        for tok, st in reorder(items, kinds):
            out.append(_switch(shown, st))
            shown = st
            out.append(tok)
    out.append(_switch(shown, end_state))
    return "".join(out)


# What a state without its tag looks like: the font and colour reset (FFFF) and the size at 100 %, as the dump writes them
DEFAULT = {"0001": "{0E 0000 0001 0002 FFFF}", "0002": "{0E 0000 0002 0002 0064}", "0003": "{0E 0000 0003 0002 FFFF}"}


def _switch(shown: dict[str, str], want: dict[str, str]) -> str:
    """The tags that turn the state shown into the one wanted."""
    out = []
    for key in sorted(set(shown) | set(want)):
        have, need = shown.get(key, DEFAULT.get(key)), want.get(key, DEFAULT.get(key))
        if have == need:
            continue
        if key == "012D":
            out.append(need or "{0F 012D 0000}")
        else:
            out.append(need)
    return "".join(out)


def shape_escaped(text: str) -> str:
    """shape() over the text between its tags and escapes, which it leaves as they are."""
    parts = re.split(r"(\{0[EF] [0-9A-F ]+\}|\\.)", text)
    return "".join(p if i % 2 else shape(p) for i, p in enumerate(parts))


if __name__ == "__main__":
    print(visual(sys.argv[1]))
