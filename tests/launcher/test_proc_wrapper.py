"""Every child process of runtime/ goes through ui/launcher/launcher_proc.cpp.

Why: mkw.exe has no console (GUI subsystem), so SDL_CreateProcess there opens a console window for each Python script
and what it starts; launcher_proc.cpp starts them without one and keeps the real exit code. A direct call to
SDL_CreateProcess, SDL_WaitProcess or SDL_ReadProcess brings the windows back on Windows only.
"""

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
WRAPPER = ROOT / "runtime/ui/launcher/launcher_proc.cpp"
CALL = re.compile(r"\bSDL_(CreateProcess|CreateProcessWithProperties|WaitProcess|ReadProcess)\s*\(")


def test_process_calls_use_the_wrapper():
    runtime = ROOT / "runtime"
    hits = [
        f"{path.relative_to(ROOT)}:{n}"
        for path in sorted([*runtime.rglob("*.c"), *runtime.rglob("*.cpp")])
        if path != WRAPPER and "third_party" not in path.parts
        for n, line in enumerate(path.read_text(encoding="utf-8", errors="replace").splitlines(), 1)
        if CALL.search(line)
    ]
    assert not hits, "use proc_spawn/proc_wait/proc_read (ui/launcher/launcher_proc.h): " + ", ".join(hits)
