# Copyright (c) 2026      The University of Tennessee and The University
#                         of Tennessee Research Foundation.  All rights
#                         reserved.
"""Behaviour across the whole life of a process, checked in a fresh Python
process: PaRSEC can be initialized only once per process."""

import os
import subprocess
import sys
import textwrap

import pytest

# A separate PaRSEC process cannot be started from inside an MPI job.
pytestmark = pytest.mark.skipif(
    any(v in os.environ for v in ("OMPI_COMM_WORLD_SIZE", "PMI_SIZE", "PMIX_RANK", "MPI_LOCALNRANKS")),
    reason="runs a separate PaRSEC process, not possible under an MPI launcher")


def run(code):
    env = dict(os.environ, OPENBLAS_NUM_THREADS="1")
    proc = subprocess.run([sys.executable, "-c", textwrap.dedent(code)], env=env,
                          capture_output=True, text=True, timeout=120)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    return proc.stdout


def test_tiles_outlive_the_context():
    out = run("""
        import numpy as np, py_parsec
        with py_parsec.Context(nb_cores=1) as ctx:
            A = py_parsec.BlockCyclic(ctx, 64, 64, 64, 64)
            tile = A.tile(0, 0)
            tile[:] = 2.0
        assert A.released
        try:
            A.tile(0, 0)
        except RuntimeError:
            pass
        else:
            raise AssertionError("tile() after fini")
        print(tile.sum())
        del A, tile
    """)
    assert out.split()[-1] == str(2.0 * 64 * 64)


def test_no_second_initialization():
    out = run("""
        import py_parsec
        with py_parsec.Context(nb_cores=1):
            pass
        try:
            py_parsec.Context(nb_cores=1)
        except RuntimeError as e:
            print("refused:", e)
    """)
    assert "refused: PaRSEC cannot be initialized again" in out


def test_finalized_at_exit_with_pending_work():
    out = run("""
        import py_parsec
        from py_parsec import dtd
        ctx = py_parsec.Context(nb_cores=2)
        A = py_parsec.BlockCyclic(ctx, 16, 16, 4, 4)
        tp = dtd.Taskpool()
        ctx.add_taskpool(tp)
        tc = tp.task_class("inc", dtd.Data(dtd.INOUT | dtd.AFFINITY, A))
        tc.add_chore(py_parsec.DEV_CPU, lambda a: a.__iadd__(1))
        for m in range(A.mt):
            for n in range(A.nt):
                tp.insert_task(tc, (A, m, n))
        print("exiting")
    """)
    assert "exiting" in out


def test_context_of_an_ended_thread_finalized_at_exit():
    """The context was created, and left active, by a thread that has ended:
    the exit handler runs on the main thread and takes over."""
    out = run("""
        import threading, py_parsec
        from py_parsec import dtd
        box = []
        def make():
            ctx = py_parsec.Context(nb_cores=2)
            A = py_parsec.BlockCyclic(ctx, 8, 8, 4, 4)
            tp = dtd.Taskpool()
            ctx.add_taskpool(tp)
            tc = tp.task_class("inc", dtd.Data(dtd.INOUT | dtd.AFFINITY, A))
            tc.add_chore(py_parsec.DEV_CPU, lambda a: a.__iadd__(1))
            tp.insert_task(tc, (A, 0, 0))
            tp.wait()
            box.append(A)
        thread = threading.Thread(target=make)
        thread.start()
        thread.join()
        print("value", box[0].tile(0, 0)[0, 0])
    """)
    assert "value 1.0" in out
