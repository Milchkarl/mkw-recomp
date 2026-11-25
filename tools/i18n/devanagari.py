"""Hindi as the game and the menu can draw it: each syllable as one private-use character.

Why: the game draws text one character at a time (nn::fontll::ScalableFontEngine, no shaping) and so does ImGui, but
Devanagari reorders and joins letters within a syllable (कि is क then ि, drawn with ि first; क्ष is one shape). Unlike
Arabic (arabic.py) Unicode has no ready-joined forms for it, so devanagari_font.py shapes every syllable of our Hindi
texts with HarfBuzz once, draws each as one glyph at a private-use code point (U+F100-F7FF, which no font of the game's
Latin text nor the menu uses) and writes the table i18n/syllables/hi.tsv. Here each Devanagari run of a text is cut
into those syllables, longest first (devanagari_font.py checks this cut is HarfBuzz's own for every text), and each is
replaced by its code point. No HarfBuzz is needed here, so the setup and the build run it anywhere.

Usage (a check, catalog.py and game_texts.py import it):
  uv run tools/i18n/devanagari.py 'हिन्दी'      the text with its syllables' code points
"""

import sys
from functools import cache
from pathlib import Path

TABLE = Path(__file__).resolve().parents[2] / "i18n" / "syllables" / "hi.tsv"
FIRST, LAST = 0xF100, 0xF7FF


def is_devanagari(ch: str) -> bool:
    return "ऀ" <= ch <= "ॿ" or "꣠" <= ch <= "ꣿ"


def runs(text: str) -> list[tuple[bool, str]]:
    """text cut into its Devanagari runs and the rest, in order: (is Devanagari, part)."""
    out: list[tuple[bool, str]] = []
    for ch in text:
        deva = is_devanagari(ch)
        if out and out[-1][0] == deva:
            out[-1] = (deva, out[-1][1] + ch)
        else:
            out.append((deva, ch))
    return out


@cache
def table() -> dict[str, str]:
    """syllable -> its private-use character."""
    rows = {}
    for line in TABLE.read_text(encoding="utf-8").splitlines():
        if line and not line.startswith("#"):
            syllable, cp = line.split("\t")
            rows[syllable] = chr(int(cp, 16))
    return rows


def syllables(run: str) -> list[str]:
    """A Devanagari run cut into the table's syllables, longest first."""
    rows = table()
    longest = max(map(len, rows))
    out, i = [], 0
    while i < len(run):
        for n in range(min(longest, len(run) - i), 0, -1):
            if run[i : i + n] in rows:
                out.append(run[i : i + n])
                i += n
                break
        else:
            raise KeyError(
                f"{run!r}: no syllable for {run[i:]!r} in {TABLE.name}; run uv run tools/i18n/devanagari_font.py"
            )
    return out


def encode(text: str) -> str:
    """text with each Devanagari syllable as its private-use character."""
    rows = table()
    return "".join("".join(rows[s] for s in syllables(part)) if deva else part for deva, part in runs(text))


if __name__ == "__main__":
    print(" ".join(f"U+{ord(c):04X}" if ord(c) >= FIRST else c for c in encode(sys.argv[1])))
