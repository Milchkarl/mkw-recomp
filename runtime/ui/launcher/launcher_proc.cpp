// Child processes without a console window on Windows (launcher_proc.h). SDL adds CREATE_NO_WINDOW only to
// background processes (SDL 3.4, src/process/windows/SDL_windowsprocess.c), which also send inherited output to NUL
// and report exit code 0. So a background process here gets our own stdout and stderr handed over explicitly, and
// keeps a handle of its own to read the real exit code (SDL holds the process open until SDL_DestroyProcess, so its
// id cannot be reused before OpenProcess).
#include "ui/launcher/launcher_proc.h"

#ifdef _WIN32
#include <windows.h>

static const char HANDLE_PROP[] = "mkw.process.handle";

static void SDLCALL close_handle(void *, void *value) { CloseHandle((HANDLE)value); }

// An SDL_IOStream that only carries a Windows handle: what SDL_PROCESS_STDIO_REDIRECT reads (it duplicates it).
static SDL_IOStream *handle_stream(DWORD which) {
    HANDLE h = GetStdHandle(which);
    if (!h || h == INVALID_HANDLE_VALUE) return nullptr;   // a GUI start: nothing to inherit
    SDL_IOStream *io = SDL_IOFromDynamicMem();
    if (io) SDL_SetPointerProperty(SDL_GetIOProperties(io), SDL_PROP_IOSTREAM_WINDOWS_HANDLE_POINTER, h);
    return io;
}

SDL_Process *proc_spawn(const char *const *args, bool pipe_stdio) {
    SDL_PropertiesID props = SDL_CreateProperties();
    SDL_SetPointerProperty(props, SDL_PROP_PROCESS_CREATE_ARGS_POINTER, (void *)args);
    SDL_SetBooleanProperty(props, SDL_PROP_PROCESS_CREATE_BACKGROUND_BOOLEAN, true);
    SDL_IOStream *out = pipe_stdio ? nullptr : handle_stream(STD_OUTPUT_HANDLE);
    SDL_IOStream *err = handle_stream(STD_ERROR_HANDLE);
    if (pipe_stdio) {
        SDL_SetNumberProperty(props, SDL_PROP_PROCESS_CREATE_STDIN_NUMBER, SDL_PROCESS_STDIO_APP);
        SDL_SetNumberProperty(props, SDL_PROP_PROCESS_CREATE_STDOUT_NUMBER, SDL_PROCESS_STDIO_APP);
    } else if (out) {
        SDL_SetNumberProperty(props, SDL_PROP_PROCESS_CREATE_STDOUT_NUMBER, SDL_PROCESS_STDIO_REDIRECT);
        SDL_SetPointerProperty(props, SDL_PROP_PROCESS_CREATE_STDOUT_POINTER, out);
    }
    if (err) {
        SDL_SetNumberProperty(props, SDL_PROP_PROCESS_CREATE_STDERR_NUMBER, SDL_PROCESS_STDIO_REDIRECT);
        SDL_SetPointerProperty(props, SDL_PROP_PROCESS_CREATE_STDERR_POINTER, err);
    }
    SDL_Process *p = SDL_CreateProcessWithProperties(props);
    SDL_DestroyProperties(props);
    if (out) SDL_CloseIO(out);
    if (err) SDL_CloseIO(err);
    if (!p) return nullptr;
    SDL_PropertiesID pp = SDL_GetProcessProperties(p);
    DWORD pid = (DWORD)SDL_GetNumberProperty(pp, SDL_PROP_PROCESS_PID_NUMBER, 0);
    HANDLE h = pid ? OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, FALSE, pid) : nullptr;
    if (h) SDL_SetPointerPropertyWithCleanup(pp, HANDLE_PROP, h, close_handle, nullptr);
    return p;
}

static void real_code(SDL_Process *p, int *code) {
    HANDLE h = (HANDLE)SDL_GetPointerProperty(SDL_GetProcessProperties(p), HANDLE_PROP, nullptr);
    DWORD c;
    *code = h && GetExitCodeProcess(h, &c) && c != STILL_ACTIVE ? (int)c : -1;
}

bool proc_wait(SDL_Process *p, bool block, int *code) {
    if (!SDL_WaitProcess(p, block, code)) return false;
    if (code) real_code(p, code);
    return true;
}

void *proc_read(SDL_Process *p, size_t *size, int *code) {
    void *data = SDL_ReadProcess(p, size, code);   // it waits for the end
    if (data && code) real_code(p, code);
    return data;
}

#else

SDL_Process *proc_spawn(const char *const *args, bool pipe_stdio) { return SDL_CreateProcess(args, pipe_stdio); }
bool proc_wait(SDL_Process *p, bool block, int *code) { return SDL_WaitProcess(p, block, code); }
void *proc_read(SDL_Process *p, size_t *size, int *code) { return SDL_ReadProcess(p, size, code); }

#endif
