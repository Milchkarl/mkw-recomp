# tools/i18n

| File | What | Usage |
|---|---|---|
| `game_texts.py` | The game's own texts of one `romfs/Mals` folder (SARC of MSBT) as label and text lines: the vocabulary a translation reuses | `uv run tools/i18n/game_texts.py EUde [-o DIR]` |
| `game_fonts.py` | The game's fonts with the letters a translation needs and they lack (Turkish ğ Ğ ş Ş İ, Vietnamese ă đ ơ ư and those with two marks, Ukrainian ґ є і ї), composed from their own letters and marks into the translation's RomFS overlay; `game_texts.py build` and the setup call it | `uv run tools/i18n/game_fonts.py missing\|build tr` |
| `game_fonts_vi.py`, `game_fonts_uk.py` | The Vietnamese and Ukrainian letters `game_fonts.py` composes: where each mark goes, which letter is another's shape | imported by `game_fonts.py` |
| `game_fonts_ar.py` | The Arabic letters `game_fonts.py` adds: the joined forms copied from Noto Sans Arabic ExtraBold, scaled to each font | imported by `game_fonts.py` |
| `arabic.py` | Arabic as the game and the menu can draw it (they draw one glyph per character, left to right): letters joined (Presentation Forms-B) and, for the game, each line in visual order with its tags | imported by `game_texts.py` and `catalog.py`; `uv run tools/i18n/arabic.py TEXT` |
| `game_fonts_hi.py` | The Hindi syllables `game_fonts.py` adds: each private-use character's glyph copied from `NotoSansDevanagari-Syllables-ExtraBold.ttf`, scaled to each font | imported by `game_fonts.py` |
| `devanagari.py` | Hindi as the game and the menu can draw it (one glyph per character, no shaping): each Devanagari syllable replaced by its private-use character from `i18n/syllables/hi.tsv` | imported by `game_texts.py` and `catalog.py`; `uv run tools/i18n/devanagari.py TEXT` |
| `devanagari_font.py` | Shapes every syllable of the Hindi texts with HarfBuzz on the pinned Noto Sans Devanagari and rewrites `i18n/syllables/hi.tsv` and `runtime/ui/fonts/NotoSansDevanagari-Syllables-{Medium,ExtraBold}.ttf`; rerun after changing a Hindi text | `uv run tools/i18n/devanagari_font.py` |
| `arabic_font.py` | Rewrites `runtime/ui/fonts/NotoSansArabic-{Medium,ExtraBold}.ttf` from the pinned Noto Sans Arabic | `uv run tools/i18n/arabic_font.py` |
| `glyph_shapes.py` | Glyph outlines as shapes to move, scale and add to a font, for `game_fonts.py` and its recipes | imported |
| `catalog.py` | The menu's text catalogs (`i18n/ui/mkw.pot`, `i18n/ui/<lang>.po`): merges the English template into every translation (changed English marked fuzzy), shows how far each language is, and generates the C arrays `build.py` compiles | `uv run tools/i18n/catalog.py update\|status\|gen DIR` |
| `fill.py` | Writes new or changed menu texts in every language from a JSON table `{msgctxt: {lang: text}}`: only empty or fuzzy entries (a fuzzy one loses its flag), never a checked translation | `uv run tools/i18n/fill.py TABLE.json` after `catalog.py update` |
| `fill_recent_strings.py` | One-shot fill of 19 menu, launcher and setup strings (Grand Prix, DSU, Discord, start screen, files gone/found, free-space error) left empty in 17 languages; same rules as `fill_mod_strings.py` (it reuses its `main`) | run once, then `catalog.py status` to confirm |
| `fill_mod_strings.py` | One-shot fill of the Mods page and last-run notice strings in all catalog languages (author-provided, translators refine): only empty entries, never a translator's words | run once per new string batch, then `catalog.py status` to confirm |
