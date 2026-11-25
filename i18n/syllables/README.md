# i18n/syllables

The game and the menu draw one glyph per character and shape nothing, so a script whose letters join and reorder
within a syllable is written with one private-use character per syllable. Each file lists the syllables our texts in
that language use and their code points; `tools/i18n/devanagari_font.py` writes it (with the fonts that draw them,
`runtime/ui/fonts/NotoSansDevanagari-Syllables-*.ttf`) and `tools/i18n/devanagari.py` reads it at build and setup.
After changing a Hindi text, rerun `uv run tools/i18n/devanagari_font.py`: the build stops on a syllable missing here.

| File | Contents |
|---|---|
| `hi.tsv` | Hindi (Devanagari): `syllable<TAB>code point`, U+F100 onwards |
