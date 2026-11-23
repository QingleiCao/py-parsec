# Copyright (c) 2026      The University of Tennessee and The University
#                         of Tennessee Research Foundation.  All rights
#                         reserved.
"""PTG and DTD taskpools on the same context and the same data."""

import gc

import numpy as np
import pytest

import py_parsec
from py_parsec import dtd
from py_parsec.matrix import redistribute_dtd, redistribute_new

from conftest import assert_matches, scatter

M, N = 50, 40


def global_matrix():
    return np.arange(M * N, dtype=np.float64).reshape(M, N)


def test_redistribute_ptg_context_wait(ctx):
    src = py_parsec.BlockCyclic(ctx, M, N, 8, 8, name="src")
    dst = py_parsec.BlockCyclic(ctx, M, N, 12, 6, P=1, name="dst")
    scatter(src, global_matrix())
    tp = redistribute_new(src, dst)
    ctx.add_taskpool(tp)
    ctx.wait()
    tp.free()
    assert_matches(dst, global_matrix())


def test_redistribute_ptg_taskpool_wait(ctx):
    """parsec_taskpool_wait() on a PTG taskpool, also on a single process."""
    src = py_parsec.BlockCyclic(ctx, M, N, 8, 8, name="src")
    dst = py_parsec.BlockCyclic(ctx, M, N, 12, 6, P=1, name="dst")
    scatter(src, global_matrix())
    for _ in range(2):
        tp = redistribute_new(src, dst)
        ctx.add_taskpool(tp)
        tp.wait()
        tp.free()
        assert_matches(dst, global_matrix())


def test_redistribute_ptg_submatrix(ctx):
    src = py_parsec.BlockCyclic(ctx, M, N, 8, 8, name="src")
    dst = py_parsec.BlockCyclic(ctx, 64, 64, 16, 16, name="dst")
    scatter(src, global_matrix())
    ctx.start()
    tp = redistribute_new(src, dst, 20, 15, disi_source=5, disj_source=3, disi_target=10, disj_target=7)
    ctx.add_taskpool(tp)
    ctx.wait()
    tp.free()
    expected = np.zeros((64, 64))
    expected[10:30, 7:22] = global_matrix()[5:25, 3:18]
    assert_matches(dst, expected)


def test_redistribute_arguments(ctx):
    src = py_parsec.BlockCyclic(ctx, M, N, 8, 8)
    single = py_parsec.BlockCyclic(ctx, M, N, 8, 8, dtype=np.float32)
    with pytest.raises(TypeError, match="float64"):
        redistribute_new(src, single)
    with pytest.raises(TypeError, match="float64"):
        redistribute_new(single, py_parsec.BlockCyclic(ctx, M, N, 8, 8, dtype=np.float32))
    with pytest.raises(TypeError, match="float64"):
        redistribute_dtd(ctx, single, single)
    with pytest.raises(ValueError, match="rejected"):
        redistribute_new(src, py_parsec.BlockCyclic(ctx, 10, 10, 8, 8))


def test_redistribute_dtd(ctx):
    src = py_parsec.BlockCyclic(ctx, M, N, 8, 8, name="src")
    dst = py_parsec.BlockCyclic(ctx, M, N, 12, 6, P=1, name="dst")
    scatter(src, global_matrix())
    redistribute_dtd(ctx, src, dst)
    assert not ctx.active
    assert_matches(dst, global_matrix())


def test_redistribute_dtd_completes_pending_work(ctx):
    """redistribute_dtd first completes the other taskpools of the context,
    flushing their collections, so their tasks and errors are not lost."""
    src = py_parsec.BlockCyclic(ctx, M, N, 8, 8, name="src")
    dst = py_parsec.BlockCyclic(ctx, M, N, 10, 10, P=1, name="dst")
    scatter(src, global_matrix())
    tp = dtd.Taskpool()
    ctx.add_taskpool(tp)
    twice = tp.task_class("twice", dtd.Data(dtd.INOUT | dtd.AFFINITY, src))
    twice.add_chore(py_parsec.DEV_CPU, lambda tile: tile.__imul__(2))
    for m in range(src.mt):
        for n in range(src.nt):
            tp.insert_task(twice, (src, m, n))
    redistribute_dtd(ctx, src, dst)     # no explicit flush or wait of tp
    assert_matches(dst, 2 * global_matrix())
    tp.free()


def test_dtd_then_ptg_then_dtd(ctx):
    """DTD writes src, a PTG taskpool copies it to dst, DTD scales dst."""
    src = py_parsec.BlockCyclic(ctx, M, N, 8, 8, name="src")
    dst = py_parsec.BlockCyclic(ctx, M, N, 10, 10, P=1, name="dst")

    fill = dtd.Taskpool()
    ctx.add_taskpool(fill)
    init = fill.task_class("init", dtd.Data(dtd.OUTPUT | dtd.AFFINITY, src), dtd.Value("i"), dtd.Value("i"))

    def init_body(tile, m, n):
        rows = np.arange(m * 8, m * 8 + 8)[:, None]
        cols = np.arange(n * 8, n * 8 + 8)[None, :]
        tile[:] = rows * N + cols

    init.add_chore(py_parsec.DEV_CPU, init_body)
    for m in range(src.mt):
        for n in range(src.nt):
            fill.insert_task(init, (src, m, n), m, n)
    ctx.wait()

    copy = redistribute_new(src, dst)
    ctx.add_taskpool(copy)
    ctx.wait()

    scale = dtd.Taskpool()
    ctx.add_taskpool(scale)
    twice = scale.task_class("twice", dtd.Data(dtd.INOUT | dtd.AFFINITY, dst))
    twice.add_chore(py_parsec.DEV_CPU, lambda tile: tile.__imul__(2))
    for m in range(dst.mt):
        for n in range(dst.nt):
            scale.insert_task(twice, (dst, m, n))
    ctx.wait()
    for t in (fill, copy, scale):
        t.free()
    assert_matches(dst, 2 * global_matrix())


def test_never_enqueued_taskpool(ctx):
    """A PTG taskpool that never ran can be freed or dropped."""
    src = py_parsec.BlockCyclic(ctx, M, N, 8, 8)
    dst = py_parsec.BlockCyclic(ctx, M, N, 8, 8)
    tp = redistribute_new(src, dst)
    tp.free()
    assert tp.released
    dropped = redistribute_new(src, dst)
    del dropped
    gc.collect()
    ctx.wait()


def test_redistribute_dtd_after_a_body_error(ctx, tp):
    """The copy is collective: an exception of a body, stored on one rank
    only, is raised after it rather than before (which left the other ranks
    waiting in the copy for good)."""
    src = py_parsec.BlockCyclic(ctx, M, N, 8, 8, name="src")
    dst = py_parsec.BlockCyclic(ctx, M, N, 12, 6, P=1, name="dst")
    scatter(src, global_matrix())
    tc = tp.task_class("boom", dtd.Data(dtd.INOUT | dtd.AFFINITY, src))

    def body(tile):
        raise ValueError("body error")

    tc.add_chore(py_parsec.DEV_CPU, body)
    tp.insert_task(tc, (src, 0, 0))
    if src.rank_of(0, 0) == ctx.rank:
        with pytest.raises(ValueError, match="body error"):
            redistribute_dtd(ctx, src, dst)
    else:
        redistribute_dtd(ctx, src, dst)
    assert_matches(dst, global_matrix())


def test_release_refused_while_a_ptg_taskpool_uses_the_matrix(ctx):
    src = py_parsec.BlockCyclic(ctx, M, N, 8, 8, name="src")
    dst = py_parsec.BlockCyclic(ctx, M, N, 12, 6, P=1, name="dst")
    tp = redistribute_new(src, dst)
    ctx.add_taskpool(tp)
    with pytest.raises(RuntimeError, match="no wait has completed"):
        dst.release()
    ctx.wait()
    dst.release()
    src.release()
