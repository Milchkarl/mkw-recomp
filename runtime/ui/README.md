# runtime/ui

Menu (F1) and performance overlay (F10) with Dear ImGui, drawn on the present path (`gfx_present.c`), and the start
screen, before the game. Every subfolder with headers is on the include path (`tools/build/build.py`): `#include` goes
by name.

| Folder | Contents |
|---|---|
| [menu/](menu/README.md) | Menu core, look, settings tabs and rows, control remapping |
| [launcher/](launcher/README.md) | Start screen and setup of the game files |
| [glyphs/](glyphs/README.md) | Button icons matching the controller, in the game and in the menu |
| `fonts/` | Roboto (Apache-2.0), Noto Sans Arabic cut to its joined letters (SIL OFL 1.1, `tools/i18n/arabic_font.py`) and the Hindi syllables drawn from Noto Sans Devanagari (SIL OFL 1.1, `tools/i18n/devanagari_font.py`), licenses next to them |
