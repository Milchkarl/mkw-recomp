# /// script
# requires-python = ">=3.14"
# dependencies = ["fonttools>=4.60"]
# ///
"""The two Arabic fonts in runtime/ui/fonts: Noto Sans Arabic cut to the letters arabic.py writes.

Why: neither the game's fonts nor Roboto have Arabic. The menu embeds NotoSansArabic-Medium.ttf (beside its Roboto
Medium) and the setup copies the glyphs of NotoSansArabic-ExtraBold.ttf into the game's fonts (game_fonts_ar.py,
beside Rodin EB). Both are static instances of Google Fonts' variable Noto Sans Arabic (SIL OFL 1.1,
runtime/ui/fonts/LICENSE-NotoSansArabic.txt) at a pinned commit, cut to the joined forms of Arabic Presentation
Forms-B and Arabic punctuation: no shaping tables, as arabic.py has already joined the letters.

Usage:
  uv run tools/i18n/arabic_font.py           rewrite both fonts (downloads the pinned font into build/fonts/)
"""

import hashlib
import sys
import urllib.request
from pathlib import Path

from fontTools import subset
from fontTools.ttLib import TTFont
from fontTools.varLib import instancer

ROOT = Path(__file__).resolve().parents[2]
COMMIT = "8b0a1d0f5983c89bc2b93f1b5fb55f9e252744b5"  # google/fonts, the last change of ofl/notosansarabic (25/11/2026)
URL = f"https://raw.githubusercontent.com/google/fonts/{COMMIT}/ofl/notosansarabic/NotoSansArabic%5Bwdth,wght%5D.ttf"
SHA256 = "63111b5b2e074dd48cc67692e0a2726d86ee94c1c37fe8598257b7b4e87e869e"
# the joined letters and lam-alef ligatures, the Arabic comma, semicolon, question mark and percent sign
UNICODES = [*range(0xFE70, 0xFEFD), 0x060C, 0x061B, 0x061F, 0x066A, 0x0640]
WEIGHTS = {"Medium": 500, "ExtraBold": 800}


def source() -> Path:
    path = ROOT / "build" / "fonts" / "NotoSansArabic-VF.ttf"
    if not path.is_file():
        path.parent.mkdir(parents=True, exist_ok=True)
        with urllib.request.urlopen(URL, timeout=60) as r:
            path.write_bytes(r.read())
    if hashlib.sha256(path.read_bytes()).hexdigest() != SHA256:
        sys.exit(f"{path}: not the pinned Noto Sans Arabic (sha256 {SHA256})")
    return path


def main() -> int:
    src = source()
    for name, weight in WEIGHTS.items():
        font = instancer.instantiateVariableFont(TTFont(src), {"wght": weight, "wdth": 100})
        opts = subset.Options()
        opts.layout_features = []
        opts.name_IDs = ["*"]
        opts.notdef_outline = True
        sub = subset.Subsetter(opts)
        sub.populate(unicodes=UNICODES)
        sub.subset(font)
        out = ROOT / "runtime" / "ui" / "fonts" / f"NotoSansArabic-{name}.ttf"
        font.save(out)
        print(f"{out.relative_to(ROOT)}: {out.stat().st_size} bytes, {len(font.getGlyphOrder())} glyphs")
    return 0


if __name__ == "__main__":
    sys.exit(main())
