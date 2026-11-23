// Windows plat_setjmp/plat_longjmp (runtime/platform/win32_jmp.c), run under wine by test_plat_jmp.py: a jump out
// of a deep call chain lands with the callee-saved integer and xmm registers, MXCSR and the x87 control word of the
// setjmp, a value of 0 arrives as 1, and the stack is usable after many jumps.
#include <stdio.h>
#include <xmmintrin.h>

#include "plat_jmp.h"

static PlatJmp jb;
static volatile int sink;

__attribute__((noinline)) static void deep(int n, int v) {
    volatile char pad[256];
    pad[n & 255] = (char)n;
    sink += pad[n & 255];
    if (n) {
        deep(n - 1, v);
        return;
    }
    _mm_setcsr((_mm_getcsr() & ~0x6000u) | 0x6000u);   // round toward zero, to be undone by the jump
    unsigned short cw = 0x0c7f;
    __asm__ volatile("fldcw %0" ::"m"(cw));
    // clobber every callee-saved register the compiler might keep the caller's values in
    __asm__ volatile("xor %%ebx,%%ebx; xor %%edi,%%edi; xor %%esi,%%esi; xor %%r12d,%%r12d; xor %%r13d,%%r13d;"
                     "xor %%r14d,%%r14d; xor %%r15d,%%r15d; xorps %%xmm6,%%xmm6; xorps %%xmm7,%%xmm7;"
                     "xorps %%xmm8,%%xmm8; xorps %%xmm9,%%xmm9; xorps %%xmm10,%%xmm10; xorps %%xmm11,%%xmm11;"
                     "xorps %%xmm12,%%xmm12; xorps %%xmm13,%%xmm13; xorps %%xmm14,%%xmm14; xorps %%xmm15,%%xmm15"
                     ::: "rbx", "rdi", "rsi", "r12", "r13", "r14", "r15", "xmm6", "xmm7", "xmm8", "xmm9", "xmm10",
                     "xmm11", "xmm12", "xmm13", "xmm14", "xmm15");
    plat_longjmp(jb, v);
}

__attribute__((noinline)) static int round_trip(int v, int depth) {
    register long a __asm__("rbx") = 0x1111, b __asm__("r12") = 0x2222, c __asm__("r15") = 0x3333;
    register double x __asm__("xmm6") = 1.5, y __asm__("xmm15") = -2.25;
    __asm__ volatile("" : "+r"(a), "+r"(b), "+r"(c), "+x"(x), "+x"(y));
    unsigned csr = _mm_getcsr();
    unsigned short cw0, cw1;
    __asm__ volatile("fnstcw %0" : "=m"(cw0));
    int r = plat_setjmp(jb);
    if (!r) deep(depth, v);
    __asm__ volatile("" : "+r"(a), "+r"(b), "+r"(c), "+x"(x), "+x"(y));
    __asm__ volatile("fnstcw %0" : "=m"(cw1));
    if (a != 0x1111 || b != 0x2222 || c != 0x3333) return printf("gpr lost: %lx %lx %lx\n", a, b, c), -1;
    if (x != 1.5 || y != -2.25) return printf("xmm lost: %g %g\n", x, y), -1;
    if (_mm_getcsr() != csr) return printf("mxcsr %#x, was %#x\n", _mm_getcsr(), csr), -1;
    if (cw1 != cw0) return printf("fpcw %#x, was %#x\n", cw1, cw0), -1;
    return r;
}

int main(void) {
    if (round_trip(7, 40) != 7) return puts("value 7 not returned"), 1;
    if (round_trip(0, 3) != 1) return puts("value 0 not returned as 1"), 1;
    for (int i = 0; i < 100000; i++)
        if (round_trip(i | 1, i & 15) != (i | 1)) return printf("round %d failed\n", i), 1;
    puts("ok");
    return 0;
}
