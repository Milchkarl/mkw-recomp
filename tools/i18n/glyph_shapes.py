"""Glyph outlines as shapes to move, scale and add to a font (game_fonts.py and its per-language recipes)."""

from fontTools.pens.boundsPen import BoundsPen
from fontTools.pens.recordingPen import DecomposingRecordingPen

IDENTITY = (1, 0, 0, 1, 0, 0)
Box = tuple[float, float, float, float]
Shape = list[tuple[Box, list]]  # contours: (box of their points, pen operations)


def bounds(glyphs, name: str) -> tuple[float, float, float, float]:
    pen = BoundsPen(glyphs)
    glyphs[name].draw(pen)
    return pen.bounds


def contours(glyphs, name: str) -> list[tuple[float, float, float, float]]:
    """The control-point box of each contour of a glyph."""
    pen = DecomposingRecordingPen(glyphs)
    glyphs[name].draw(pen)
    out, pts = [], []
    for op, args in pen.value:
        pts += args
        if op in ("closePath", "endPath") and pts:
            xs, ys = [p[0] for p in pts], [p[1] for p in pts]
            out.append((min(xs), min(ys), max(xs), max(ys)))
            pts = []
    return out


def outline(glyphs, name: str) -> Shape:
    pen = DecomposingRecordingPen(glyphs)
    glyphs[name].draw(pen)
    out, ops, pts = [], [], []
    for op, args in pen.value:
        ops.append((op, args))
        pts += args
        if op in ("closePath", "endPath"):
            if pts:
                xs, ys = [p[0] for p in pts], [p[1] for p in pts]
                out.append(((min(xs), min(ys), max(xs), max(ys)), ops))
            ops, pts = [], []
    return out


def box(shape: Shape) -> Box:
    return (min(b[0] for b, _ in shape), min(b[1] for b, _ in shape), max(b[2] for b, _ in shape),
            max(b[3] for b, _ in shape))  # fmt: skip


def moved(b: Box, m: tuple) -> Box:
    return (b[0] * m[0] + m[4], b[1] * m[3] + m[5], b[2] * m[0] + m[4], b[3] * m[3] + m[5])


def fit(shape: Shape, scale: float, x0: float | None = None, y0: float | None = None, cx: float | None = None,
        y1: float | None = None) -> tuple[tuple, Box]:  # fmt: skip
    """The transform scaling shape and putting its left edge at x0 (or its centre at cx), its bottom at y0 (or its top
    at y1); and the box it then covers."""
    b = box(shape)
    dx = (x0 - b[0] * scale) if x0 is not None else (cx - (b[0] + b[2]) / 2 * scale)
    dy = (y0 - b[1] * scale) if y0 is not None else (y1 - b[3] * scale)
    m = (scale, 0, 0, scale, dx, dy)
    return m, moved(b, m)
