# i18n/game/base

One `<lang>.tsv` per language: every translated label of the game, the ones its updates added included (format and
use in [../README.md](../README.md)). The texts an update reworded keep here the translation of the first version.

| File | Language |
|---|---|
| `ca.tsv` | Catalan |
| `gl.tsv` | Galician |
| `eu.tsv` | Basque |
| `tr.tsv` | Turkish (over the British English) |
| `vi.tsv` | Vietnamese (over the British English) |
| `uk.tsv` | Ukrainian (over the British English, the game's Russian as the guide for names and lengths) |
| `id.tsv` | Indonesian (over the British English; names stay in English) |
| `ar.tsv` | Arabic (over the British English), in reading order: the build joins its letters and writes each line in visual order (`tools/i18n/arabic.py`) |
| `hi.tsv` | Hindi (over the British English), in Devanagari: the build writes each syllable as one character (`tools/i18n/devanagari.py`); after changing a text, `uv run tools/i18n/devanagari_font.py` |
