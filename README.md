# py-parsec

[![CI](https://github.com/QingleiCao/py-parsec/actions/workflows/ci.yml/badge.svg)](https://github.com/QingleiCao/py-parsec/actions/workflows/ci.yml)

Python bindings for [PaRSEC](https://github.com/ICLDisco/parsec), the
task-based runtime system for distributed heterogeneous architectures.

The bindings follow the C API closely and keep its structure: one context
per process, one kind of taskpool for every DSL, and data collections that
are not tied to a DSL. Task bodies are Python callables (NumPy, SciPy, ...)
or C function pointers, and the same program runs on one process or under
`mpiexec`.

| Python | C |
| --- | --- |
| `py_parsec.Context(nb_cores, args)` | `parsec_init` / `parsec_fini` |
| `ctx.add_taskpool(tp)`, `ctx.start()`, `ctx.wait()` | `parsec_context_add_taskpool`, `parsec_context_start`, `parsec_context_wait` |
| `py_parsec.Taskpool`: `wait()`, `free()`, `from_address()` | a `parsec_taskpool_t` of any DSL |
| `py_parsec.BlockCyclic(ctx, m, n, mb, nb, dtype, P)` | `parsec_matrix_block_cyclic_t` |
| `py_parsec.dtd.Taskpool()` | `parsec_dtd_taskpool_new` |
| `tp.task_class(name, dtd.Data(...), dtd.Value(...))`, `tc.add_chore(dev, body)` | `parsec_dtd_create_task_class`, `parsec_dtd_task_class_add_chore` |
| `tp.insert_task(tc, (A, m, n), 3)`, `tp.insert_tasks(tc, iterable)` | `parsec_dtd_insert_task_with_task_class` |
| `tp.data_flush_all(A)` | `parsec_dtd_data_flush_all` |
| `py_parsec.matrix.redistribute_new(src, dst, ...)` | `parsec_redistribute_New` (a PTG taskpool) |

## Example

```python
import py_parsec
from py_parsec import dtd

with py_parsec.Context() as ctx:
    A = py_parsec.BlockCyclic(ctx, 1000, 1000, 100, 100, name="A")
    tp = dtd.Taskpool()
    ctx.add_taskpool(tp)

    scale = tp.task_class("scale", dtd.Data(dtd.INOUT | dtd.AFFINITY, A), dtd.Value("d"))

    def body(tile, alpha):      # tile: (100, 100) column-major view of the PaRSEC data
        tile *= alpha

    scale.add_chore(py_parsec.DEV_CPU, body)
    for m in range(A.mt):
        for n in range(A.nt):
            tp.insert_task(scale, (A, m, n), 2.0)
    tp.wait()                   # flushes A, then completes the tasks

    for m, n, tile in A.local_tiles():   # the tiles owned by this process
        ...
```

Run it with `python example.py` or `mpiexec -n 4 python example.py`. More in
[`examples/`](examples): a tiled GEMM (`dtd_gemm.py`) and DTD and PTG
taskpools working on the same data (`redistribute.py`).

## Requirements

- PaRSEC from its `master` branch, 2026-11-07 or later (4.1.0 development
  version, which includes ICLDisco/parsec#840), built as a shared library
  (`-DBUILD_SHARED_LIBS=ON`) with its development headers
  (`-DPARSEC_WITH_DEVEL_HEADERS=ON`, the default). MPI support is optional.
- Python 3.10 or later and NumPy.
- To build: a C compiler, `pkg-config`, Cython 3 (installed automatically by
  `pip`).

## Installation

Point `PARSEC_ROOT` at the PaRSEC installation prefix (the directory that
holds `lib/pkgconfig/parsec.pc` or `lib64/pkgconfig/parsec.pc`):

```bash
PARSEC_ROOT=/path/to/parsec/install pip install py-parsec
```

or, from a clone of this repository:

```bash
PARSEC_ROOT=/path/to/parsec/install pip install .
```

The build reads the PaRSEC flags from `pkg-config parsec` and links the
extension modules with `libparsec`, with a run path to it. `CC` selects the C
compiler. Without a pkg-config file, set `PARSEC_INCLUDE_DIRS` and
`PARSEC_LIBRARY_DIRS` instead (`os.pathsep`-separated lists).

The bindings read some PaRSEC structures directly, whose layout depends on
how PaRSEC was configured, so py-parsec must run with the PaRSEC build it
was compiled against: importing it with another `libparsec` (another commit,
or the same commit configured again) raises an `ImportError` that says so.
Reinstall py-parsec after rebuilding PaRSEC.

## Concepts

### The context

`py_parsec.Context(nb_cores=-1, args=None)` initializes PaRSEC; `nb_cores`
counts the calling thread, and `args` are PaRSEC options (for instance
`["--mca", "runtime_comm_short_limit", "0"]`). PaRSEC can be initialized once
per process, so create one context and keep it for the whole run. It is
finalized by `fini()`, at the end of a `with` block, or at interpreter exit;
finalizing completes and releases every taskpool and matrix. PaRSEC
initializes MPI if the application has not, and then finalizes it in
`fini()`: import `mpi4py.MPI` before creating the context to keep using MPI
after that.

The compute threads start when work is scheduled (`ctx.start()` starts them
explicitly, for instance to let PTG taskpools run while Python does something
else). `ctx.wait()` completes every taskpool of the context and leaves the
threads idle; `tp.wait()` completes one taskpool and leaves the others
running.

### Taskpools

Every DSL produces a `parsec_taskpool_t`, wrapped by `py_parsec.Taskpool`.
DTD taskpools (`py_parsec.dtd.Taskpool`), PTG taskpools returned by library
calls (`py_parsec.matrix.redistribute_new`), and taskpools built by C code
(`Taskpool.from_address(addr, keepalive=...)`, for instance a JDF compiled
into its own library) are added to the context and waited the same way.
`Context.address` and `TiledMatrix.address` hand the context and the
matrices to such C code.

A taskpool can be dropped at any time: the context keeps it until a wait has
completed its work, and it is released afterwards. `tp.free()` completes and
releases it right away. No PaRSEC function is ever called from a Python
finalizer: taskpools and matrices collected by Python are released by the
next call into the runtime from the thread that created the context.

Only the thread that created the context can drive the runtime (insert,
flush, wait, free, release a matrix): PaRSEC identifies its callers by an
execution stream that only that thread has. Calls from other threads, and
from task bodies, raise `RuntimeError`. If that thread has ended when the
interpreter exits, the exit handler takes over its role to finalize the
context.

### Matrices

`py_parsec.BlockCyclic(ctx, m, n, mb, nb, dtype=np.float64, P=None, ...)` is
a 2D block-cyclic tiled matrix (float32, float64, complex64, complex128,
int32, int8 or uint8) over a `P x Q` process grid. Its local tiles are
allocated and zeroed; `A.tile(m, n)` and `A.local_tiles()` give them as
`(mb, nb)` Fortran-ordered NumPy arrays that share the PaRSEC storage. The
same matrix can be used by DTD and PTG taskpools. A matrix and its storage
live as long as the matrix object or one of its tile arrays; `A.release()`
frees them earlier, once no taskpool that used the matrix has work that no
wait has completed (it raises `RuntimeError` otherwise). Once the context is finalized, the tile arrays still
hold their data (to read results after a `with` block), but the matrix can
no longer be used by tasks.

### DTD

A task class lists its parameters: `dtd.Data(mode, like=A)` (or
`shape=(mb, nb), dtype=...`) for a tile, where the mode is `dtd.INPUT`,
`dtd.OUTPUT` or `dtd.INOUT`, possibly with `dtd.AFFINITY` (run the task where
this tile lives; tasks that write data need it in distributed runs),
`dtd.DONT_TRACK`, `dtd.PUSHOUT` or `dtd.PULLIN`; and `dtd.Value(fmt)` for a
number passed by value, `fmt` being a `struct` code among `b B h H i I l L q
Q f d`.

- `tp.insert_task(tc, *args, priority=0, device=DEV_ALL)` inserts one task.
  A tile argument is `(A, m, n)`, `(A, m, n, flags)` (flags added for this
  insertion only, such as `dtd.PUSHOUT`) or `dtd.tile_of(A, m, n)`.
- `tp.insert_tasks(tc, tasks)` inserts one task per argument tuple of an
  iterable, in batches, without taking the GIL back between the tasks of a
  batch (see Performance). It is equivalent to a loop of `insert_task`: if
  the iterable inserts other tasks, flushes or waits, the tasks it produced
  before are inserted first, and if it raises, the tasks before the failing
  item are inserted.
- `tc.add_chore(py_parsec.DEV_CPU, body)` adds a Python body. It is called
  with one argument per parameter: NumPy arrays for tiles (read-only for
  `INPUT`), numbers for values. The arrays are only valid during the call.
  A body may return `py_parsec.HOOK_RETURN_AGAIN` to be rescheduled; any
  other result completes the task. A task whose body raises completes, and
  the next wait raises the exception (with a note counting the others).
- `tc.add_chore(device, address)` adds a C chore for any device type, given
  as an int, a PyCapsule or a ctypes function pointer with the DTD chore
  signature `parsec_hook_return_t f(parsec_execution_stream_t *, parsec_task_t *)`.
  Accelerator kernels stay in user code this way. `dtd.NOOP_CHORE` is a C
  chore that does nothing.
- Waits flush the collections the taskpool's tasks used since their last
  flush, as DTD requires; `tp.data_flush(tile)` and `tp.data_flush_all(A)`
  flush explicitly.

## Performance

- **One BLAS thread per task.** Each task is one core's work, but NumPy's
  BLAS (OpenBLAS, MKL) starts its own threads on every call by default, so
  every worker oversubscribes the cores PaRSEC already uses. Set
  `OPENBLAS_NUM_THREADS=1` (and `OMP_NUM_THREADS`, `MKL_NUM_THREADS`) before
  NumPy is imported, as `examples/dtd_gemm.py` does. In that example (4096,
  256 x 256 tiles, 16 cores), 16 BLAS threads per call instead of one cut the
  throughput by more than half on one rank, and more on 4 ranks of 4 cores.
- **Keep results column-major.** Tiles are Fortran-ordered, while NumPy
  returns products in C order, so `c += a @ b` adds two arrays of opposite
  layouts: a strided pass over the tile that halved the speed of the GEMM
  kernel with 128 x 128 tiles. `c += (b.T @ a.T).T` computes the same
  product in column-major order, at full speed.
- **Enough work per Python body.** A Python body runs with the GIL, which
  NumPy releases in its heavy calls. With one worker, a Python body costs
  about 0.2 us more than a C one; with many workers per process, the bodies
  and the inserting thread queue for the GIL (an empty body costs about
  8 us per task with 8 workers, 6 us with `insert_tasks`). Give each body a fraction of a millisecond
  of work or more, and use `insert_tasks` for many small tasks. A body that
  spends long in pure Python keeps the GIL for up to
  `sys.getswitchinterval()` (5 ms) before another thread gets it.
- **Ranks rather than threads for small tasks.** Each process has its own
  GIL, so several ranks per node with a few cores each scale better than
  one rank with many cores when tasks are short. Give each rank its own
  cores: PaRSEC binds its threads inside the CPU set of the process, so with
  Open MPI run `mpiexec -n 4 --map-by slot:PE=4 --bind-to core ...` for 4
  ranks of 4 cores, not `--bind-to none`, which puts the threads of every
  rank on the same cores (PaRSEC warns about it).
- **C bodies for fine-grained tasks.** A C chore runs without the GIL.

Measured on a node with two Xeon Gold 6140 (one process unless noted),
Python 3.11, NumPy 2.4 with OpenBLAS, PaRSEC master built in Release mode.
Per-task cost from `benchmarks/overhead.py` and its C counterpart
`benchmarks/c_overhead.c`, which insert independent tasks on 1 x 1 tiles,
so that the time is the runtime's and the binding's:

| per task, 1 core | PaRSEC from C | `insert_task` | `insert_tasks` |
| --- | --- | --- | --- |
| insert, 1 tile, C body | 1.57 us | 2.06 us | 1.75 us |
| insert, 3 tiles, C body | 5.38 us | 6.04 us | 5.68 us |
| insert, 1 tile and 2 values, C body | 1.60 us | 2.26 us | 1.93 us |
| Python body instead of a C one | | +0.22 us | |

PaRSEC executes the same instructions in both cases. Of the extra
instructions of an insertion from Python, about a quarter are the binding's
(1,100 per task with `insert_task`, 400 with `insert_tasks`) and the rest
are the caller's own loop. With 8 workers, an empty Python body costs 8.1 us
per task with `insert_task` and 6.0 us with `insert_tasks` (insertion and
completion).

Tiled GEMM, C += A B with 4096 x 4096 doubles (`examples/dtd_gemm.py`, NumPy
bodies, one BLAS thread, best of 3 runs), and its fraction of 16 times the
single-core speed of the same NumPy tile update (63.7 GFlop/s for 256 x 256
tiles, 56.6 for 128 x 128):

| tile | 1 rank x 16 cores | 4 ranks x 4 cores |
| --- | --- | --- |
| 256 (0.53 ms per task) | 653 GFlop/s, 64% | 558 GFlop/s, 55% |
| 128 (0.07 ms per task) | 154 GFlop/s, 17% | 399 GFlop/s, 44% |

## Limitations

- One PaRSEC context per process, initialized once (PaRSEC cannot be
  initialized again after `parsec_fini`).
- Python bodies run on CPU devices; accelerator chores are C functions.
- `parsec_context_test()` and `parsec_taskpool_test()` are not exposed: the
  first cannot report completion while the context is started, and the
  second returns the number of tasks the caller ran, which is 0 both when the
  taskpool completed and when no task was ready.
- `redistribute_new` / `redistribute_dtd` handle float64 matrices only, as
  PaRSEC compiles them for double precision.
- A DTD taskpool holds at most `dtd.MAX_TASK_CLASSES` (24) task classes
  with a chore, released ones included: create task classes once and reuse
  them for every insertion.

## Tests, examples and benchmarks

```bash
pip install pytest
python -m pytest tests
mpiexec -n 4 python -m pytest tests
```

Each rank runs every test and checks the tiles it owns;
`PY_PARSEC_TEST_CORES` sets the cores per process (2 by default).

```bash
mpiexec -n 4 python examples/dtd_gemm.py -M 2048 -b 256 --check
mpiexec -n 4 python examples/redistribute.py
python benchmarks/overhead.py --cores 1 8
```

## Development

Build in place against a PaRSEC installation and run the tests from the
source tree:

```bash
pip install cython numpy pytest setuptools
PARSEC_ROOT=/path/to/parsec/install python setup.py build_ext --inplace
PYTHONPATH=src python -m pytest tests
```

The extension modules are rebuilt automatically when the PaRSEC found by
`pkg-config` changes. Continuous integration builds PaRSEC `master` and runs
the tests on 1, 2 and 4 ranks, and the examples on 4 ranks
([`.github/workflows/ci.yml`](.github/workflows/ci.yml)). See
[RELEASING.md](RELEASING.md) to cut a release.

## License

BSD 3-Clause Open MPI variant, the license of PaRSEC; see [LICENSE](LICENSE).
