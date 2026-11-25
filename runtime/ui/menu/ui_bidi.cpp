// Right-to-left lines for ImGui, which draws every line left to right one glyph at a time. The Arabic catalog
// (i18n/ui/ar.po) is generated with its letters already joined (tools/i18n/arabic.py, Arabic Presentation Forms-B) and
// in reading order; ImGui wraps it as any text and, through the hook below in its RenderText (imgui_draw.cpp, marked
// mkw-recomp:), each wrapped line is drawn in visual order. The rules are arabic.py's visual_line: a right-to-left
// line, Latin words and numbers kept left to right inside it, what lies between two of them joining them, brackets
// mirrored in the right-to-left parts.
#include <stdint.h>
#include <string.h>

#include "imgui.h"
#include "imgui_internal.h"   // ImTextCharFromUtf8

extern int (*GImFontVisualLine)(const char *s, const char *e, char *out, int cap);   // imgui_draw.cpp

namespace {

enum Kind : uint8_t { N, R, L, EN };

bool is_rtl(uint32_t c) {
    return (c >= 0x0590 && c <= 0x08FF) || (c >= 0xFB1D && c <= 0xFDFF) || (c >= 0xFE70 && c <= 0xFEFF);
}

Kind kind(uint32_t c) {
    if ((c >= '0' && c <= '9') || (c >= 0x0660 && c <= 0x0669)) return EN;
    if (is_rtl(c)) return R;
    if ((c | 0x20) >= 'a' && (c | 0x20) <= 'z') return L;
    if (c >= 0xC0 && !(c >= 0x2000 && c <= 0x2BFF) && c != 0xD7 && c != 0xF7) return L;   // letters past Latin-1's signs
    return N;
}

uint32_t mirror(uint32_t c) {
    switch (c) {
    case '(': return ')'; case ')': return '(';
    case '[': return ']'; case ']': return '[';
    case '<': return '>'; case '>': return '<';
    case 0xAB: return 0xBB; case 0xBB: return 0xAB;
    default: return c;
    }
}

int put_utf8(char *out, int at, int cap, uint32_t c) {
    char buf[4];
    int n = c < 0x80 ? 1 : c < 0x800 ? 2 : c < 0x10000 ? 3 : 4;
    static const uint8_t lead[] = {0, 0, 0xC0, 0xE0, 0xF0};
    for (int k = n - 1; k > 0; k--, c >>= 6) buf[k] = (char)(0x80 | (c & 0x3F));
    buf[0] = (char)(lead[n] | c);
    if (at + n > cap) return -1;
    memcpy(out + at, buf, n);
    return at + n;
}

// [s, e) in visual order into out; -1 when the line has no right-to-left character (drawn as it is) or does not fit.
int visual_line(const char *s, const char *e, char *out, int cap) {
    enum { MAX = 512 };
    uint32_t cp[MAX];
    uint8_t lv[MAX];
    int n = 0, rtl = 0;
    for (const char *p = s; p < e && *p;) {
        unsigned int c;
        p += ImTextCharFromUtf8(&c, p, e);
        if (n == MAX) return -1;
        cp[n] = c;
        Kind k = kind(c);
        rtl |= k == R;
        lv[n++] = k == R ? 1 : k == N ? 0 : 2;
    }
    if (!rtl) return -1;
    for (int i = 0; i < n; i++) {   // a neutral between two left-to-right runs joins them; any other is right-to-left
        if (lv[i]) continue;
        int b = 1, a = 1;
        for (int j = i - 1; j >= 0; j--) if (lv[j]) { b = lv[j]; break; }
        for (int j = i + 1; j < n; j++) if (lv[j]) { a = lv[j]; break; }
        lv[i] = b == 2 && a == 2 ? 2 : 1;
    }
    for (int i = 0; i < n; i++)
        if (lv[i] == 1) cp[i] = mirror(cp[i]);
    int at = 0;
    for (int i = n - 1; i >= 0;) {   // right to left; each left-to-right run written forwards
        if (lv[i] == 2) {
            int j = i;
            while (j > 0 && lv[j - 1] == 2) j--;
            for (int k = j; k <= i && at >= 0; k++) at = put_utf8(out, at, cap, cp[k]);
            i = j - 1;
        } else {
            at = put_utf8(out, at, cap, cp[i--]);
        }
        if (at < 0) return -1;
    }
    return at;
}

}   // namespace

void ui_bidi_install(void) { GImFontVisualLine = visual_line; }
