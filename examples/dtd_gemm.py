#!/usr/bin/env python3
# Copyright (c) 2026      The University of Tennessee and The University
#                         of Tennessee Research Foundation.  All rights
#                         reserved.
"""Tiled C = C + A B with DTD tasks whose bodies are NumPy calls.

Run on one process or with an MPI launcher, for instance
``mpiexec -n 4 python dtd_gemm.py -M 2048 -b 256``. With ``--check`` every
rank rebuilds the matrices from the seed and verifies the tiles it owns.

Each task is one core's work, so the BLAS library under NumPy is kept to one
thread (unless the environment says otherwise): with its default, every
task would start its own BLAS threads on the cores PaRSEC already uses.
"""

import argparse
import os
import time

for _var in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "1")  # must happen before NumPy is imported

import numpy as np

import py_parsec
from py_parsec import dtd


def tile_values(seed, rows, cols, b):
    """A b x b tile: random values in its rows x cols part inside the matrix,
    zeros in the padding of the last tile row and column."""
    tile = np.zeros((b, b))
    tile[:rows, :cols] = np.random.default_rng(seed).random((rows, cols))
    return tile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-M", type=int, default=1024, help="rows of A and C")
    parser.add_argument("-N", type=int, default=None, help="columns of B and C (default M)")
    parser.add_argument("-K", type=int, default=None, help="columns of A, rows of B (default M)")
    parser.add_argument("-b", "--tile", type=int, default=128, help="tile size")
    parser.add_argument("-c", "--cores", type=int, default=-1, help="compute threads per process")
    parser.add_argument("-P", type=int, default=None, help="rows of the process grid")
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--seed", type=int, default=3872)
    parser.add_argument("--batch", action="store_true", help="insert with Taskpool.insert_tasks")
    parser.add_argument("--check", action="store_true", help="verify the result")
    args = parser.parse_args()
    M, b = args.M, args.tile
    N = args.N or M
    K = args.K or M

    with py_parsec.Context(nb_cores=args.cores) as ctx:
        A = py_parsec.BlockCyclic(ctx, M, K, b, b, P=args.P, name="A")
        B = py_parsec.BlockCyclic(ctx, K, N, b, b, P=args.P, name="B")
        C = py_parsec.BlockCyclic(ctx, M, N, b, b, P=args.P, name="C")
        tp = dtd.Taskpool()
        ctx.add_taskpool(tp)

        # Every tile gets its own random stream, so any rank can rebuild it.
        def seed_of(salt, X, m, n):
            return (args.seed * 4 + salt) * 1_000_003 + m * X.nt + n

        init = tp.task_class("init", dtd.Data(dtd.OUTPUT | dtd.AFFINITY, A),
                             dtd.Value("q"), dtd.Value("i"), dtd.Value("i"))

        def init_body(tile, seed, rows, cols):
            tile[:] = tile_values(seed, rows, cols, b)

        init.add_chore(py_parsec.DEV_CPU, init_body)
        for X, salt in ((A, 1), (B, 2), (C, 3)):
            for m in range(X.mt):
                for n in range(X.nt):
                    tp.insert_task(init, (X, m, n), seed_of(salt, X, m, n), *X.tile_shape(m, n))
        tp.wait()

        gemm = tp.task_class("gemm", dtd.Data(dtd.INPUT, A), dtd.Data(dtd.INPUT, B),
                             dtd.Data(dtd.INOUT | dtd.AFFINITY, C))

        def gemm_body(a, b, c):
            # Tiles are column-major and a @ b would be row-major, making the
            # update a strided pass over c; (b.T @ a.T).T is the same product,
            # column-major. NumPy releases the GIL inside the BLAS call.
            c += (b.T @ a.T).T

        gemm.add_chore(py_parsec.DEV_CPU, gemm_body)

        gflop = 2.0 * M * N * K / 1e9
        for run in range(args.runs):
            start = time.perf_counter()
            if args.batch:
                tp.insert_tasks(gemm, (((A, m, k), (B, k, n), (C, m, n, dtd.PUSHOUT if k == A.nt - 1 else 0))
                                       for m in range(C.mt) for n in range(C.nt) for k in range(A.nt)))
            else:
                for m in range(C.mt):
                    for n in range(C.nt):
                        for k in range(A.nt):
                            tp.insert_task(gemm, (A, m, k), (B, k, n),
                                           (C, m, n, dtd.PUSHOUT if k == A.nt - 1 else 0),
                                           priority=C.mt * C.nt * A.nt - k)
            tp.wait()
            elapsed = time.perf_counter() - start
            if ctx.rank == 0:
                print(f"run {run}: M={M} N={N} K={K} tile={b} nodes={ctx.nb_nodes} "
                      f"cores={ctx.nb_cores} time={elapsed:.3f}s gflops={gflop / elapsed:.2f}")

        if args.check:
            def full(X, salt):
                out = np.empty((X.mt * b, X.nt * b))
                for m in range(X.mt):
                    for n in range(X.nt):
                        out[m * b:(m + 1) * b, n * b:(n + 1) * b] = tile_values(
                            seed_of(salt, X, m, n), *X.tile_shape(m, n), b)
                return out[:X.m, :X.n]

            expected = full(C, 3) + args.runs * (full(A, 1) @ full(B, 2))
            error = 0.0
            for m, n, tile in C.local_tiles():
                rows, cols = C.tile_shape(m, n)
                block = expected[m * b:m * b + rows, n * b:n * b + cols]
                error = max(error, float(np.abs(tile[:rows, :cols] - block).max()))
            ok = error <= 1e-10 * K * args.runs
            print(f"rank {ctx.rank}: max error {error:.3e} {'PASSED' if ok else 'FAILED'}")
            if not ok:
                raise SystemExit(1)
        tp.free()


if __name__ == "__main__":
    main()
