// Non-local jumps of the runtime (a guest thread's ExitThread, a guest longjmp back into a live setjmp owner).
//
// Windows: our own setjmp/longjmp, not ucrtbase's. ucrtbase's longjmp, even with the non-SEH NULL frame, ended the
// process with 0xC0000409 (fast fail 0xD, FAST_FAIL_INVALID_SET_OF_CONTEXT, raised in ucrtbase.dll per #702) on some
// Windows 11 machines at the first guest thread exit after "fs mount save" (#751, #762 on 0.1.17: the log stops right
// after the first "a guest thread exits"). Ours restores the callee-saved registers and jumps, and pops the CET
// shadow stack back to the setjmp's caller when it is on (as glibc's longjmp does). Linux keeps glibc's.
#pragma once

#ifdef _WIN32
#include <stdint.h>
// rbx rbp rdi rsi r12-r15 rsp rip (0-72), mxcsr+fpcw (80), ssp (88), xmm6-xmm15 (96-255)
typedef struct PlatJmpBuf {
    uint64_t gpr[10];
    uint32_t mxcsr, fpcw;
    uint64_t ssp;
    uint64_t xmm[20];
} PlatJmpBuf;
typedef PlatJmpBuf PlatJmp[1];
__attribute__((returns_twice)) int plat_setjmp(PlatJmpBuf *b);
__attribute__((noreturn)) void plat_longjmp(PlatJmpBuf *b, int v);
#else
#include <setjmp.h>
typedef jmp_buf PlatJmp;
#define plat_setjmp(b) setjmp(b)
#define plat_longjmp(b, v) longjmp(b, v)
#endif
