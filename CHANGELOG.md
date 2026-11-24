# Changelog

All notable changes to py-parsec are listed here. The project follows
[semantic versioning](https://semver.org); until 1.0, minor versions may
change the API.

## 0.1.0 (2026-11-24)

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
  binding adds 0.5 to 0.7 us per insertion over C instead of 1.0 to 1.6 us
  (0.2 to 0.3 us with `insert_tasks`), an empty Python body on 8 workers
  costs 8.1 us per task instead of 25.5 us (6.0 us with `insert_tasks`), and
  the GEMM example with 128 x 128 tiles and the same body runs at 143
  GFlop/s instead of 56 on 16 cores.
- All exceptions raised by bodies are reported (the first is raised with a
  note counting the others).
- A second `Context` raises an error instead of crashing in `parsec_init`.
- Matrices are released like taskpools (never from a finalizer), and their
  storage outlives the context while tile arrays refer to it, so results can
  be read after a `with` block.
- Saved exceptions of task bodies no longer keep the body's locals, which
  are arrays over data copies that do not outlive the task.
- The GEMM example verifies correctly when the matrix size is not a multiple
  of the tile size, and keeps its update column-major (`c += (b.T @ a.T).T`),
  which doubles the speed of its body with 128 x 128 tiles.
- Calls into the runtime from a thread other than the one that created the
  context, or from a task body, raise `RuntimeError` instead of crashing or
  hanging. A context whose thread has ended is still finalized at exit.
- An exception of a task body no longer makes `redistribute_dtd` skip the
  collective copy on that rank (the other ranks waited for good): it is
  raised after the copy.
- `add_chore` raises past the 24 task classes a DTD taskpool can hold
  (`dtd.MAX_TASK_CLASSES`), where PaRSEC aborted the process.
- `TiledMatrix.release()` raises while a taskpool whose work no wait has
  completed uses the matrix, instead of aborting or freeing tiles PaRSEC
  still holds.
- Empty matrices, `Param` subclasses, parameters and tiles created without
  their constructor are rejected instead of crashing in PaRSEC.
