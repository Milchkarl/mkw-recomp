"""Windows plat_setjmp/plat_longjmp (runtime/platform/win32_jmp.c, which replaces ucrtbase's longjmp for a guest
thread's exit) built with llvm-mingw and run under wine by t_plat_jmp.c: registers, MXCSR, x87 control word and the
returned value survive a jump out of a deep call chain. Skipped without build/win/llvm-mingw or wine.

Usage: uv run pytest tests/runtime/test_plat_jmp.py
"""

import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
CC = ROOT / "build/win/llvm-mingw/bin/x86_64-w64-mingw32-clang"


@pytest.mark.skipif(not CC.exists() or not shutil.which("wine"), reason="needs build/win/llvm-mingw and wine")
def test_plat_jmp(tmp_path: Path) -> None:
    exe = tmp_path / "t_plat_jmp.exe"
    plat = ROOT / "runtime/platform"
    subprocess.run(
        [str(CC), "-O2", "-Wall", "-Werror", "-I", str(plat), str(ROOT / "tests/runtime/t_plat_jmp.c"),
         str(plat / "win32_jmp.c"), "-o", str(exe)],
        check=True,
    )
    env = {"WINEDEBUG": "-all", "WINEPREFIX": str(tmp_path / "wine"), "PATH": "/usr/bin:/bin"}
    r = subprocess.run(["wine", str(exe)], capture_output=True, text=True, timeout=120, env=env)
    assert r.returncode == 0, r.stdout + r.stderr
    assert r.stdout.strip().endswith("ok")
