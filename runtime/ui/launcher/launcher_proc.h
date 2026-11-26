// The launcher's child processes (ui/launcher/launcher_proc.cpp): SDL_CreateProcess, SDL_WaitProcess and
// SDL_ReadProcess with the same arguments, but on Windows without a console window. mkw.exe is a GUI program (no
// console), so each console program it starts (python.exe, and through it uv, git, clang...) would open its own;
// SDL only hides it for background processes, whose exit code it then reports as 0: these read the real one.
#pragma once

#include <SDL3/SDL.h>

SDL_Process *proc_spawn(const char *const *args, bool pipe_stdio);
bool proc_wait(SDL_Process *p, bool block, int *code);
void *proc_read(SDL_Process *p, size_t *size, int *code);
