#!/usr/bin/env python3
# Copyright (c) 2026      The University of Tennessee and The University
#                         of Tennessee Research Foundation.  All rights
#                         reserved.
"""Per-task overhead of the bindings.

Each case inserts independent tasks (one tile each, so no task waits for
another) and reports the time per task, insertion and completion included:

* ``c-1tile`` / ``c-3tiles`` / ``c-values``: a C chore that does nothing, so
  the time is PaRSEC's plus the binding's insertion cost;
* ``py-body``: an empty Python body, which adds the body dispatch and the GIL.

Compare with the same tasks inserted from C (``tests/dsl/dtd`` in PaRSEC) to
separate the binding overhead from the runtime's. Run on one process::

    python benchmarks/overhead.py --cores 1 8 --tasks 200000
"""

import argparse
import os
import time

for _var in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import numpy as np

import py_parsec
from py_parsec import dtd


def _noop_chore():
    if hasattr(dtd, "NOOP_CHORE"):
        return dtd.NOOP_CHORE
    import ctypes
    lib = ctypes.CDLL(os.environ["PY_PARSEC_NOOP_LIB"])
    return ctypes.cast(lib.noop_chore, ctypes.c_void_p).value


# Tiles as (matrix, m, n) tuples where supported, dtd.tile_of() otherwise.
_TUPLES = hasattr(dtd, "NOOP_CHORE")


def _tile(A, m, n):
    return (A, m, n) if _TUPLES else dtd.tile_of(A, m, n)


def run_case(ctx, case, ntasks, batch):
    side = int(np.ceil(np.sqrt(ntasks)))
    A = py_parsec.BlockCyclic(ctx, side, side, 1, 1, name="A")
    B = py_parsec.BlockCyclic(ctx, side, side, 1, 1, name="B")
    C = py_parsec.BlockCyclic(ctx, side, side, 1, 1, name="C")
    tp = dtd.Taskpool()
    ctx.add_taskpool(tp)
    ctx.start()
    if case == "c-3tiles":
        tc = tp.task_class(case, dtd.Data(dtd.INPUT, A), dtd.Data(dtd.INPUT, B),
                           dtd.Data(dtd.INOUT | dtd.AFFINITY, C))
    elif case == "c-values":
        tc = tp.task_class(case, dtd.Data(dtd.INOUT | dtd.AFFINITY, A), dtd.Value("i"), dtd.Value("d"))
    else:
        tc = tp.task_class(case, dtd.Data(dtd.INOUT | dtd.AFFINITY, A))
    if case == "py-body":
        tc.add_chore(py_parsec.DEV_CPU, lambda tile: None)
    else:
        tc.add_chore(py_parsec.DEV_CPU, _noop_chore())

    def args(i):
        m, n = divmod(i, side)
        if case == "c-3tiles":
            return (_tile(A, m, n), _tile(B, m, n), _tile(C, m, n))
        if case == "c-values":
            return (_tile(A, m, n), i, 0.5)
        return (_tile(A, m, n),)

    start = time.perf_counter()
    if batch:
        tp.insert_tasks(tc, (args(i) for i in range(ntasks)))
    else:
        insert = tp.insert_task
        for i in range(ntasks):
            insert(tc, *args(i))
    inserted = time.perf_counter()
    tp.wait()
    done = time.perf_counter()
    tp.free()
    for X in (A, B, C):
        X.release()
    return (inserted - start) / ntasks * 1e6, (done - start) / ntasks * 1e6


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cores", type=int, nargs="+", default=[1, 8])
    parser.add_argument("--tasks", type=int, default=200_000)
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--cases", nargs="+", default=["c-1tile", "c-3tiles", "c-values", "py-body"])
    parser.add_argument("--batch", action="store_true", help="insert with Taskpool.insert_tasks")
    args = parser.parse_args()
    for cores in args.cores:
        with py_parsec.Context(nb_cores=cores) as ctx:
            for case in args.cases:
                best = min((run_case(ctx, case, args.tasks, args.batch) for _ in range(args.repeat)),
                           key=lambda r: r[1])
                if ctx.rank == 0:
                    print(f"cores={cores:3d} {case:9s} insert {best[0]:6.2f} us/task   "
                          f"insert+complete {best[1]:6.2f} us/task")


if __name__ == "__main__":
    main()
