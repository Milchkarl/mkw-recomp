// The optional third-party libraries downloaded when the settings pick them (libs_fetch.h): tools/prepare/libs.py fetch
// in the package's Python, its event lines ("progress", "ready", "done", "error") read by a thread of ours. Players
// had to find these DLLs and copy them beside mkw.exe by hand (#788, #803); none ships in the package, as NVIDIA's
// licence forbids distributing DLSS's files on their own, so each is downloaded from its official source and checked
// against the sha256 pinned in libs.tsv.
#include <SDL3/SDL.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "ui/launcher/libs_fetch.h"
#include "ui/launcher/launcher_setup.h"

extern "C" {
#include "platform/platform.h"
void rt_log(const char *fmt, ...) __attribute__((format(printf, 1, 2)));
}

static SDL_Mutex *g_lock;
static LibsState g_state;
static SDL_Process *g_proc;
static char g_pending[64];   // asked for while a fetch ran: started when it ends

static void on_line(char *line) {
    rt_log("libs: %s", line);
    long long done, total;
    if (sscanf(line, "progress %lld %lld", &done, &total) == 2) g_state.done = done, g_state.total = total;
    else if (!strncmp(line, "ready ", 6)) g_state.fetched++;
    else if (!strncmp(line, "error ", 6)) {
        const char *msg = strchr(line + 6, ' ');   // "error fetch <message>"
        snprintf(g_state.error, sizeof g_state.error, "%s", msg ? msg + 1 : line + 6);
    }
}

static void start_locked(const char *features);

static int SDLCALL reader(void *) {
    SDL_IOStream *out = SDL_GetProcessOutput(g_proc);
    char line[512];
    size_t n = 0;
    char c;
    for (;;) {
        if (SDL_ReadIO(out, &c, 1) != 1) {
            if (SDL_GetIOStatus(out) != SDL_IO_STATUS_NOT_READY) break;
            SDL_Delay(20);
            continue;
        }
        if (c == '\r') continue;
        if (c != '\n') {
            if (n + 1 < sizeof line) line[n++] = c;
            continue;
        }
        line[n] = 0, n = 0;
        SDL_LockMutex(g_lock);
        on_line(line);
        SDL_UnlockMutex(g_lock);
    }
    int code = -1;
    SDL_WaitProcess(g_proc, true, &code);
    SDL_DestroyProcess(g_proc);
    SDL_LockMutex(g_lock);
    g_proc = nullptr;
    g_state.running = 0;
    g_state.finished = code == 0 ? 1 : -1;
    if (code && !g_state.error[0]) snprintf(g_state.error, sizeof g_state.error, "exit %d", code);
    if (g_pending[0]) {
        char next[sizeof g_pending];
        snprintf(next, sizeof next, "%s", g_pending);
        g_pending[0] = 0;
        start_locked(next);
    }
    SDL_UnlockMutex(g_lock);
    return 0;
}

static void fail_locked(const char *why) {
    g_state.running = 0, g_state.finished = -1;
    snprintf(g_state.error, sizeof g_state.error, "%s", why);
    rt_log("libs: cannot download %s: %s", g_state.features, why);
}

static void start_locked(const char *features) {
    char dir[PLAT_PATH_MAX], python[1200], script[1200];
    snprintf(g_state.features, sizeof g_state.features, "%s", features);
    g_state.running = 1, g_state.finished = 0, g_state.done = g_state.total = 0, g_state.fetched = 0;
    g_state.error[0] = 0;
    char *cut = plat_exe_path(dir, sizeof dir) ? plat_last_sep(dir) : nullptr;
    if (!cut) return fail_locked("no program folder");
    *cut = 0;
    if (!setup_tool(dir, "prepare/libs.py", python, script, sizeof python)) return fail_locked("no setup tools");
    const char *args[] = {python, "-I", script, "fetch", "--features", features, "--dir", dir, nullptr};
    g_proc = SDL_CreateProcess(args, true);
    if (!g_proc) return fail_locked(SDL_GetError());
    SDL_DetachThread(SDL_CreateThread(reader, "libs-fetch", nullptr));
}

void libs_fetch_start(const char *features) {
    if (!g_lock) g_lock = SDL_CreateMutex();
    SDL_LockMutex(g_lock);
    if (g_state.running) snprintf(g_pending, sizeof g_pending, "%s", features);
    else start_locked(features);
    SDL_UnlockMutex(g_lock);
}

LibsState libs_state(void) {
    if (!g_lock) return g_state;
    SDL_LockMutex(g_lock);
    LibsState s = g_state;
    SDL_UnlockMutex(g_lock);
    return s;
}
