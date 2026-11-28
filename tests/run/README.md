# tests/run

The game-run layer of [tools/runner](../../tools/runner/README.md), checked without the game.

| File | What it checks |
|---|---|
| `fake_mkw.py` | Not a test: a scripted `build/mkw` (`FAKE_MKW` JSON) that writes the log lines, phases, `frames.tsv` and control answers of the real one, and can die by FATAL, crash, hang or window close |
| `conftest.py` | The `runenv` fixture: a temporary `build/run`, no job runner, no perf, the fake as `MKW_RUN_BIN` |
| `test_run.py` | `run.py` end to end in seconds: each way a run ends gets its word in `NAME.json`, the A/B stop at its target, `live` with `ctl` and `stop`, `at` on phases, `watch` with STALLED, BLACK and FATAL, the `.sh` wrappers, a SIGTERM that still writes the end, the flight dump pruned after a clean run, and a shot without ffmpeg (GitHub's runners) |
| `test_overlap.py` | The frame-time model of `tools/run/overlap.py` on synthetic perf windows: a serial window gains its wait back, a paced one is left out, one bound by the presenting thread's own work gains nothing |
| `test_expect.py` | `tools/runner/expect.py`: `expect=` read from `MKW_RUN_WHY` (range, single number, comma and minus sign), the always-valid interval wider than the t interval and shrinking, the refusal of an A/B too coarse for its expectation, and hit, miss and open on synthetic pairs |
| `test_framesim.py` | `tools/run/framesim.py` on a synthetic serial pipeline: the run comes back as measured with nothing changed, and a free replay, a GPU twice as fast or no wait for the GPU shorten the frame by what the arithmetic says, also when the presenting thread only waits for another game thread's wake |
| `test_abstat.py` | The A/B verdict of `abstat.py` on synthetic per-frame series with a known answer: an A/A interval holds 0 about 95 % of the time, a known effect lands inside its interval, a linear trend over the race does not read as an effect, a run cut mid-line still reads, `--metric` judges only its column, and `NAME.json`'s verdict is strict JSON (null, never NaN) |
