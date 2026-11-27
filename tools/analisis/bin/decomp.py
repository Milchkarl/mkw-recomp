"""Decompile single functions of the game to pseudo-C with Ghidra, headless, with the names the binary DB knows.

Why: some questions are about what a function does with a value (does it wait for the GPU, which field does it read,
what does it decide with it), and reading a lifted or disassembled listing for that costs thousands of lines. Ghidra's
decompiler gives the same function in tens of lines. Only the functions asked for are decompiled: there is no
whole-program analysis (it would take an hour), so callees have no known signature and appear as f_<address>(...).

What the program gets: main and sdk at the addresses the generator and the runtime use (recomp/layout.py), with their
RELATIVE relocations applied (vtables and pointers in data point at code); labels for every function and PLT import
of build/bindb.sqlite (the NVN2 driver functions of sdk included, tools/gpu/nvn2_procs.py); and the game's NVN proc
pointers named: the loader that asks nvnDeviceGetProcAddress for each name stores the result through a GOT entry, so
each GOT entry becomes got_<nvnName> and the pointer it holds pfnc_<nvnName>. The GOT range is a read-only block, so a
call shows as (*pfnc_nvnQueueSubmitCommands)(...).

Needs Ghidra (pacman -S ghidra; GHIDRA_INSTALL_DIR, default /opt/ghidra) and the dev group (pyghidra). The Ghidra
project is kept in build/ghidra/ (one per build id; the first run builds it, ~1 min). Nothing of the game is written
outside --out.

Usage: uv run tools/analisis/bin/decomp.py ADDR... [--callers N] [--out DIR] [--db DB] [--exefs DIR]
  ADDR: 0x7000123456, main+0x21071d0, f_70021071d0 (a lifted function's name), a symbol or an NVN2 proc name.
  --callers N  also decompile the direct callers (BL/B) of each function, N levels up.
  Writes OUT/<name>.c per function (default build/decomp/), with its address, callers and callees in a header.
"""

import argparse
import os
import shutil
import sys
from pathlib import Path

from capstone import CS_ARCH_ARM64, CS_MODE_ARM, Cs
from capstone.arm64 import ARM64_OP_IMM, ARM64_OP_MEM, ARM64_OP_REG

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "recomp"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from nso import R_AARCH64_RELATIVE, Nso  # noqa: E402
from q import Bin  # noqa: E402

MODULES = ("main", "sdk")
LANG = "AARCH64:LE:64:v8A"


def relocated_image(path: Path, base: int) -> tuple[Nso, bytes]:
    nso = Nso(str(path))
    img = bytearray(nso.image)
    for off, kind, _sym, add in nso.relocs:
        if kind == R_AARCH64_RELATIVE and off + 8 <= len(img):
            img[off : off + 8] = (base + add).to_bytes(8, "little")
    return nso, bytes(img)


def nvn_slots(b: Bin, img: bytes, base: int) -> dict[int, str]:
    """GOT entry -> NVN name, from the loader that stores each nvnDeviceGetProcAddress result (main+0xcf5560 in
    1.0.0): adrp/add x1 = name, blr, adrp/ldr x8 = [GOT entry], str x0, [x8]."""
    row = b.db.execute(
        "SELECT r.func FROM str s JOIN ref r ON r.target = s.addr WHERE s.module='main' AND s.text=? LIMIT 1",
        ("nvnQueueSubmitCommands",),
    ).fetchone()
    if not row:
        return {}
    lo, hi, _ = b.func(row[0])
    md = Cs(CS_ARCH_ARM64, CS_MODE_ARM)
    md.detail = True
    regs: dict[int, int] = {}
    # x1 already holds the next name when the previous result is stored: the name of a store is the one x1 held
    # at the call (blr) before it.
    name, called, got, out = None, None, None, {}
    strs = dict(b.db.execute("SELECT addr, text FROM str WHERE module='main'"))
    for i in md.disasm(img[lo - base : hi - base], lo):
        ops = i.operands
        if i.mnemonic == "adrp":
            regs[ops[0].reg] = ops[1].imm
        elif i.mnemonic == "add" and len(ops) == 3 and ops[2].type == ARM64_OP_IMM and ops[1].reg in regs:
            regs[ops[0].reg] = regs[ops[1].reg] + ops[2].imm
            if i.reg_name(ops[0].reg) == "x1":
                name = strs.get(regs[ops[0].reg])
        elif i.mnemonic == "blr":
            called = name
        elif i.mnemonic == "ldr" and ops[1].type == ARM64_OP_MEM and ops[1].mem.base in regs:
            got = regs[ops[1].mem.base] + ops[1].mem.disp
            regs.pop(ops[0].reg, None)
        elif i.mnemonic == "str" and i.reg_name(ops[0].reg) == "x0" and called and got:
            out[got] = called
            called = got = None
        elif ops and ops[0].type == ARM64_OP_REG:
            regs.pop(ops[0].reg, None)
    return out


def build(gp, b: Bin, exefs: Path, name: str):
    """Create the program: memory blocks, labels, NVN pointer names."""
    from ghidra.program.database import ProgramDB
    from ghidra.program.model.lang import CompilerSpecID, LanguageID
    from ghidra.program.model.symbol import SourceType
    from ghidra.program.util import DefaultLanguageService
    from ghidra.util.task import TaskMonitor
    from java.io import FileInputStream

    mon = TaskMonitor.DUMMY
    lang = DefaultLanguageService.getLanguageService().getLanguage(LanguageID(LANG))
    p = ProgramDB(name, lang, lang.getCompilerSpecByID(CompilerSpecID("default")), gp)
    tx = p.startTransaction("load")
    space = p.getAddressFactory().getDefaultAddressSpace()
    mem = p.getMemory()
    st = p.getSymbolTable()
    bases = {n: base for n, base, _ in b.mods}
    slots: dict[int, str] = {}
    imgdir = ROOT / "build/ghidra"
    for mod in MODULES:
        base = bases[mod]
        nso, img = relocated_image(exefs / mod, base)
        path = imgdir / f"{mod}.img"
        path.write_bytes(img)
        stream = FileInputStream(str(path))
        fb = mem.createFileBytes(path.name, 0, len(img), stream, mon)
        stream.close()
        got_lo = got_hi = 0
        if mod == "main":
            slots = nvn_slots(b, img, base)
            if slots:
                got_lo = (min(slots) - base) & ~0xFFF
                got_hi = (max(slots) - base + 8 + 0xFFF) & ~0xFFF
        segs = [("text", *nso.text, "rx"), ("rodata", *nso.rodata, "r"), ("data", *nso.data, "rw")]
        for seg, off, size, perm in segs:
            parts = [(seg, off, size, perm)]
            if seg == "data" and got_lo and off <= got_lo < got_hi <= off + size:
                parts = [
                    (seg, off, got_lo - off, perm),
                    ("got", got_lo, got_hi - got_lo, "r"),
                    (seg + "2", got_hi, off + size - got_hi, perm),
                ]
            for part, o, n, pm in parts:
                if n <= 0:
                    continue
                blk = mem.createInitializedBlock(f"{mod}.{part}", space.getAddress(base + o), fb, o, n, False)
                blk.setPermissions("r" in pm, "w" in pm, "x" in pm)
        end = nso.data[0] + nso.data[1]
        if nso.bss_size:
            blk = mem.createUninitializedBlock(f"{mod}.bss", space.getAddress(base + end), nso.bss_size, False)
            blk.setPermissions(True, True, False)
    for a, n in b.db.execute("SELECT addr, name FROM func WHERE name IS NOT NULL AND name != ''"):
        st.createLabel(space.getAddress(a), label(n), SourceType.IMPORTED)
    for a, n in b.db.execute("SELECT addr, name FROM plt WHERE name IS NOT NULL AND name != ''"):
        st.createLabel(space.getAddress(a), label(n), SourceType.IMPORTED)
    main_img = (imgdir / "main.img").read_bytes()
    for got, n in slots.items():
        st.createLabel(space.getAddress(got), f"got_{n}", SourceType.IMPORTED)
        ptr = int.from_bytes(main_img[got - bases["main"] : got - bases["main"] + 8], "little")
        if ptr:
            st.createLabel(space.getAddress(ptr), f"pfnc_{n}", SourceType.IMPORTED)
    for mod in MODULES:
        (imgdir / f"{mod}.img").unlink()
    p.endTransaction(tx, True)
    print(f"built {name}: {len(slots)} NVN proc pointers named", file=sys.stderr)
    return p


def label(n: str) -> str:
    """A Ghidra label from a (possibly demangled) name: no spaces, under Ghidra's 2000-character limit."""
    return n.replace(" ", "_")[:1900]


def ensure_function(p, b: Bin, a: int):
    from ghidra.app.cmd.function import CreateFunctionCmd
    from ghidra.program.model.symbol import SourceType

    space = p.getAddressFactory().getDefaultAddressSpace()
    addr = space.getAddress(a)
    fm = p.getFunctionManager()
    f = fm.getFunctionAt(addr)
    if f is None:
        CreateFunctionCmd(addr).applyTo(p)
        f = fm.getFunctionAt(addr)
    if f is not None and f.getName().startswith("FUN_"):
        row = b.db.execute("SELECT name FROM func WHERE addr=?", (a,)).fetchone()
        f.setName(label(row[0]) if row and row[0] else f"f_{a:x}", SourceType.USER_DEFINED)
    return f


def resolve(b: Bin, text: str) -> int:
    for prefix in ("fb_", "f_"):
        if text.startswith(prefix):
            return int(text[len(prefix) :], 16)
    return b.addr(text)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("addrs", nargs="+")
    ap.add_argument("--callers", type=int, default=0)
    ap.add_argument("--out", type=Path, default=ROOT / "build/decomp")
    ap.add_argument("--db", type=Path, default=ROOT / "build/bindb.sqlite", help="1.8.0: build/bindb-180.sqlite")
    ap.add_argument("--exefs", type=Path, default=Path(os.environ.get("MKW_GAME", ROOT / "game")) / "exefs")
    a = ap.parse_args()

    b = Bin(a.db)
    funcs: dict[int, int] = {}  # function start -> distance from what was asked
    todo = [(b.func(resolve(b, t))[0], 0) for t in a.addrs]
    while todo:
        f, d = todo.pop()
        if f in funcs:
            continue
        funcs[f] = d
        if d < a.callers:
            todo += [(c, d + 1) for (c,) in b.db.execute("SELECT DISTINCT caller FROM call WHERE target=?", (f,))]

    os.environ.setdefault("GHIDRA_INSTALL_DIR", "/opt/ghidra")
    import pyghidra

    pyghidra.start()
    from ghidra.app.decompiler import DecompileOptions, DecompInterface
    from ghidra.base.project import GhidraProject
    from ghidra.util.task import TaskMonitor

    build_id = b.db.execute("SELECT build_id FROM module WHERE name='main'").fetchone()[0][:16]
    pdir = ROOT / "build/ghidra"
    pdir.mkdir(parents=True, exist_ok=True)
    pname = f"mkw-{build_id}"
    if (pdir / f"{pname}.gpr").exists():
        gp = GhidraProject.openProject(str(pdir), pname, True)
        p = gp.openProgram("/", "main", False)
    else:
        # A run that died while building leaves the project without its program: start again.
        shutil.rmtree(pdir / f"{pname}.rep", ignore_errors=True)
        gp = GhidraProject.createProject(str(pdir), pname, False)
        p = build(gp, b, a.exefs, "main")
        gp.saveAs(p, "/", "main", True)

    tx = p.startTransaction("functions")
    for f in funcs:
        ensure_function(p, b, f)
        for (t,) in b.db.execute("SELECT DISTINCT target FROM call WHERE caller=?", (f,)):
            ensure_function(p, b, t)
    p.endTransaction(tx, True)
    gp.save(p)

    di = DecompInterface()
    di.setOptions(DecompileOptions())
    di.openProgram(p)
    a.out.mkdir(parents=True, exist_ok=True)
    fm = p.getFunctionManager()
    space = p.getAddressFactory().getDefaultAddressSpace()
    for f, d in sorted(funcs.items(), key=lambda kv: (kv[1], kv[0])):
        fn = fm.getFunctionAt(space.getAddress(f))
        res = di.decompileFunction(fn, 180, TaskMonitor.DUMMY)
        if not res.decompileCompleted():
            print(f"{b.rel(f)}: {res.getErrorMessage()}", file=sys.stderr)
            continue
        callers = [b.name(c) for (c,) in b.db.execute("SELECT DISTINCT caller FROM call WHERE target=?", (f,))]
        head = [f"// {fn.getName()} at {b.rel(f)} ({f:#x}), {d} level(s) above what was asked"]
        head.append("// callers: " + (", ".join(callers) if callers else "none found by BL/B (indirect)"))
        path = a.out / f"{fn.getName()[:120]}.c"
        path.write_text("\n".join(head) + "\n" + res.getDecompiledFunction().getC())
        print(path)
    di.dispose()
    gp.close()


if __name__ == "__main__":
    main()
