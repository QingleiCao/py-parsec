# Changelog

All notable changes to py-parsec are listed here. The project follows
[semantic versioning](https://semver.org); until 1.0, minor versions may
change the API.

## 0.1.0 (unreleased)

First release, moved out of the PaRSEC repository (ICLDisco/parsec#834,
which replaced ICLDisco/parsec#765) into a standalone package built against
an installed PaRSEC.

- `Context` (`parsec_init` / `parsec_fini`), one per process; compute
  threads start when work is scheduled, `wait()` completes every taskpool.
- `Taskpool` for every DSL: DTD taskpools, PTG taskpools from library calls,
  and C-built taskpools through `Taskpool.from_address`, all in one context.
- `BlockCyclic` tiled matrices with tiles exposed as NumPy views, shared by
  DTD and PTG taskpools; `redistribute_new` (PTG) and `redistribute_dtd`.
- DTD task classes with `Data` / `Value` parameters, Python bodies (CPU) and
  C chores (any device), `insert_task`, batched `insert_tasks`, tiles given
  as `(matrix, m, n)` tuples, read-only views for `INPUT` tiles, exceptions
  of bodies raised by the next wait.
- Lifetimes: the context keeps taskpools until their work completes;
  finalizers never call PaRSEC, releases are deferred to the next runtime
  call from the thread that created the context, outside of calls that may
  run tasks. `insert_tasks` behaves exactly like a loop of `insert_task`,
  also when its iterable inserts, flushes or waits.
- Import fails with a clear message when the loaded `libparsec` is not the
  build the extension modules were compiled against, and the modules are
  rebuilt when that PaRSEC changes.

Changes from the version reviewed in ICLDisco/parsec#834:

- `redistribute_*` reject matrices other than float64 instead of corrupting
  them (the PaRSEC kernels are compiled for double).
- `Taskpool.wait()` on a PTG taskpool works on a single process: the
  communication engine is made ready when the context is created
  (ICLDisco/parsec#836 fixes this in PaRSEC).
- No blocking call from finalizers: a taskpool with work pending is kept by
  its context, and collected taskpools are released by the next runtime call.
- Inserting or flushing on an idle context starts it instead of hanging once
  the DTD window is full.
- `redistribute_dtd` first completes the other taskpools of the context,
  with the flushes and error reporting of `Context.wait()`.
- The variadic DTD calls are made with the exact number of arguments
  instead of 64 padded pairs; per-insert validation is cached; body
  arguments are built with one array allocation per tile; PaRSEC worker
  threads keep their Python thread state; `insert_tasks` converts tasks in
  batches and inserts them without the GIL. In the same conditions, the
  binding adds 0.7 to 1.2 us per insertion over C instead of 1.0 to 1.9 us,
  an empty Python body on 8 workers costs 9.7 us per task instead of 26.1 us
  (6.8 us with `insert_tasks`), and the GEMM example with 128 x 128 tiles
  runs at 142 GFlop/s instead of 56 on 16 cores.
- All exceptions raised by bodies are reported (the first is raised with a
  note counting the others).
- A second `Context` raises an error instead of crashing in `parsec_init`.
- Matrices are released like taskpools (never from a finalizer), and their
  storage outlives the context while tile arrays refer to it, so results can
  be read after a `with` block.
- Saved exceptions of task bodies no longer keep the body's locals, which
  are arrays over data copies that do not outlive the task.
- The GEMM example verifies correctly when the matrix size is not a multiple
  of the tile size.
