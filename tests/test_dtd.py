# Copyright (c) 2026      The University of Tennessee and The University
#                         of Tennessee Research Foundation.  All rights
#                         reserved.

import ctypes
import gc
import sys
import weakref

import numpy as np
import pytest

import py_parsec
from py_parsec import dtd

from conftest import Counter, assert_matches, scatter


def test_gemm(ctx, tp):
    """C += A @ B with Python bodies; tiles move between ranks."""
    M, N, K, b = 64, 48, 80, 16
    rng = np.random.default_rng(7)
    Ag, Bg, Cg = rng.standard_normal((M, K)), rng.standard_normal((K, N)), rng.standard_normal((M, N))
    A = py_parsec.BlockCyclic(ctx, M, K, b, b, name="A")
    B = py_parsec.BlockCyclic(ctx, K, N, b, b, name="B")
    C = py_parsec.BlockCyclic(ctx, M, N, b, b, name="C")
    for X, G in ((A, Ag), (B, Bg), (C, Cg)):
        scatter(X, G)

    gemm = tp.task_class("gemm", dtd.Data(dtd.INPUT, A), dtd.Data(dtd.INPUT, B),
                         dtd.Data(dtd.INOUT | dtd.AFFINITY, C))

    def body(a, b, c):
        c += a @ b

    gemm.add_chore(py_parsec.DEV_CPU, body)
    for m in range(C.mt):
        for n in range(C.nt):
            for k in range(A.nt):
                last = dtd.PUSHOUT if k == A.nt - 1 else 0
                tp.insert_task(gemm, (A, m, k), dtd.tile_of(B, k, n), (C, m, n, last))
    tp.wait()
    assert_matches(C, Cg + Ag @ Bg, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("dtype", [np.float32, np.float64, np.complex64, np.complex128, np.int32])
def test_transpose_tiles(ctx, tp, dtype):
    """B(m, n) = A(n, m).T moves tiles of every supported type across ranks."""
    n, b = 24, 8
    full = (np.arange(n * n).reshape(n, n) % 251).astype(dtype)
    if np.iscomplexobj(full):
        full = full + 1j * full.T
    A = py_parsec.BlockCyclic(ctx, n, n, b, b, dtype=dtype, name="A")
    B = py_parsec.BlockCyclic(ctx, n, n, b, b, dtype=dtype, name="B")
    scatter(A, full)

    tc = tp.task_class("transpose", dtd.Data(dtd.INPUT, A), dtd.Data(dtd.OUTPUT | dtd.AFFINITY, B))

    def body(a, out):
        out[:] = a.T

    tc.add_chore(py_parsec.DEV_CPU, body)
    for i in range(A.mt):
        for j in range(A.nt):
            tp.insert_task(tc, (A, j, i), (B, i, j))
    tp.data_flush_all(A)
    tp.data_flush_all(B)
    tp.wait()
    assert_matches(B, full.T)


def test_wait_flushes_used_collections(ctx, tp):
    """Without data_flush_all, wait() flushes the collections the tasks used
    (a distributed DTD wait would not terminate otherwise)."""
    n, b = 24, 8
    full = np.arange(n * n, dtype=np.float64).reshape(n, n)
    A = py_parsec.BlockCyclic(ctx, n, n, b, b, name="A")
    B = py_parsec.BlockCyclic(ctx, n, n, b, b, name="B")
    scatter(A, full)
    tc = tp.task_class("transpose", dtd.Data(dtd.INPUT, A), dtd.Data(dtd.OUTPUT | dtd.AFFINITY, B))
    tc.add_chore(py_parsec.DEV_CPU, lambda a, out: np.copyto(out, a.T))
    for i in range(A.mt):
        for j in range(A.nt):
            tp.insert_task(tc, (A, j, i), (B, i, j))
    tp.wait()
    assert_matches(B, full.T)
    copy = tp.task_class("copy", dtd.Data(dtd.INPUT, A), dtd.Data(dtd.OUTPUT | dtd.AFFINITY, B))
    copy.add_chore(py_parsec.DEV_CPU, lambda a, out: np.copyto(out, a))
    for i in range(A.mt):
        for j in range(A.nt):
            tp.insert_task(copy, (A, i, j), (B, i, j))
    ctx.wait()
    assert_matches(B, full)


def test_values(ctx, tp):
    formats = "bBhHiIlLqQfd"
    values = [-5, 250, -300, 60000, -70000, 4000000000, -2**40, 2**41, -2**60, 2**63 + 5, 1.5, -2.25]
    A = py_parsec.BlockCyclic(ctx, 4, 4, 4, 4)
    tc = tp.task_class("values", dtd.Data(dtd.INOUT | dtd.AFFINITY, A), *(dtd.Value(f) for f in formats))
    seen = []
    tc.add_chore(py_parsec.DEV_CPU, lambda tile, *args: seen.append(args))
    tp.insert_task(tc, (A, 0, 0), *values)
    with pytest.raises(OverflowError):
        tp.insert_task(tc, (A, 0, 0), 1000, *values[1:])
    tp.wait()
    assert seen == ([tuple(values)] if A.is_local(0, 0) else [])


def test_body_exception_raised_by_wait(ctx, tp):
    A = py_parsec.BlockCyclic(ctx, 16, 16, 8, 8)
    tc = tp.task_class("boom", dtd.Data(dtd.INOUT | dtd.AFFINITY, A))

    def body(tile):
        raise ZeroDivisionError("raised in a task body")

    tc.add_chore(py_parsec.DEV_CPU, body)
    for m in range(A.mt):
        for n in range(A.nt):
            tp.insert_task(tc, (A, m, n))
    if A.nb_local_tiles:
        with pytest.raises(ZeroDivisionError, match="raised in a task body") as info:
            tp.wait()
        if A.nb_local_tiles > 1 and sys.version_info >= (3, 11):
            assert f"{A.nb_local_tiles - 1} other exception(s)" in " ".join(info.value.__notes__)
    else:
        tp.wait()
    tp.wait()  # reported once


def test_context_wait_raises_body_exceptions(ctx, tp):
    A = py_parsec.BlockCyclic(ctx, 8, 8, 8, 8)
    tc = tp.task_class("boom", dtd.Data(dtd.INOUT | dtd.AFFINITY, A))
    tc.add_chore(py_parsec.DEV_CPU, lambda tile: 1 / 0)
    tp.insert_task(tc, (A, 0, 0))
    if A.is_local(0, 0):
        with pytest.raises(ZeroDivisionError):
            ctx.wait()
    else:
        ctx.wait()


def test_input_tiles_are_read_only(ctx, tp):
    A = py_parsec.BlockCyclic(ctx, 8, 8, 8, 8)
    B = py_parsec.BlockCyclic(ctx, 8, 8, 8, 8)
    flags = []
    tc = tp.task_class("ro", dtd.Data(dtd.INPUT, A), dtd.Data(dtd.INOUT | dtd.AFFINITY, B))

    def body(a, b):
        flags.append((a.flags.writeable, b.flags.writeable, a.flags.f_contiguous, a.shape))
        a[0, 0] = 1.0

    tc.add_chore(py_parsec.DEV_CPU, body)
    tp.insert_task(tc, (A, 0, 0), (B, 0, 0))
    if B.is_local(0, 0):
        with pytest.raises(ValueError, match="read-only"):
            tp.wait()
        assert flags == [(False, True, True, (8, 8))]
    else:
        tp.wait()


def test_c_function_chore(ctx, tp):
    """A chore given as a C function pointer (here a ctypes callback)."""
    A = py_parsec.BlockCyclic(ctx, 32, 32, 8, 8)
    count = Counter()

    @ctypes.CFUNCTYPE(ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p)
    def chore(es, task):
        count()
        return py_parsec.HOOK_RETURN_DONE

    tc = tp.task_class("c_chore", dtd.Data(dtd.INOUT | dtd.AFFINITY, A))
    tc.add_chore(py_parsec.DEV_CPU, chore)
    for m in range(A.mt):
        for n in range(A.nt):
            tp.insert_task(tc, (A, m, n))
    tp.wait()
    assert count.value == A.nb_local_tiles


def test_noop_chore(ctx, tp):
    A = py_parsec.BlockCyclic(ctx, 32, 32, 8, 8)
    tc = tp.task_class("noop", dtd.Data(dtd.INOUT | dtd.AFFINITY, A), dtd.Value("d"))
    tc.add_chore(py_parsec.DEV_CPU, dtd.NOOP_CHORE)
    for m in range(A.mt):
        for n in range(A.nt):
            tp.insert_task(tc, (A, m, n), 1.0)
    tp.wait()


def test_several_waits(ctx, tp):
    """A DTD taskpool can be fed again after a wait."""
    A = py_parsec.BlockCyclic(ctx, 16, 16, 8, 8)
    inc = tp.task_class("inc", dtd.Data(dtd.INOUT | dtd.AFFINITY, A))
    inc.add_chore(py_parsec.DEV_CPU, lambda t: t.__iadd__(1))
    for _ in range(3):
        for m in range(A.mt):
            for n in range(A.nt):
                tp.insert_task(inc, (A, m, n))
        tp.wait()
    assert_matches(A, np.full((16, 16), 3.0))


def test_insert_starts_an_idle_context(ctx, tp):
    """Inserting into an idle context starts it: more tasks than the DTD
    window would otherwise block the inserting thread for good."""
    ctx.wait()
    assert not ctx.active
    A = py_parsec.BlockCyclic(ctx, 128, 128, 1, 1)
    tc = tp.task_class("noop", dtd.Data(dtd.INOUT | dtd.AFFINITY, A))
    tc.add_chore(py_parsec.DEV_CPU, dtd.NOOP_CHORE)
    for m in range(A.mt):
        for n in range(A.nt):
            tp.insert_task(tc, (A, m, n))
    assert ctx.active
    tp.wait()


def test_insert_errors(ctx, tp):
    A = py_parsec.BlockCyclic(ctx, 16, 16, 8, 8)
    small = py_parsec.BlockCyclic(ctx, 16, 16, 4, 4)
    tc = tp.task_class("t", dtd.Data(dtd.INOUT | dtd.AFFINITY, A), dtd.Value("i"))
    with pytest.raises(ValueError, match="no chore"):
        tp.insert_task(tc, (A, 0, 0), 1)
    tc.add_chore(py_parsec.DEV_CPU, lambda t, i: None)
    with pytest.raises(ValueError, match="already has"):
        tc.add_chore(py_parsec.DEV_CPU, lambda t, i: None)
    with pytest.raises(ValueError, match="exactly one device type"):
        tc.add_chore(py_parsec.DEV_ALL, lambda t, i: None)
    if ctx.nb_devices(py_parsec.DEV_CUDA):
        with pytest.raises(TypeError, match="DEV_CPU"):
            tc.add_chore(py_parsec.DEV_CUDA, lambda t, i: None)
    else:
        with pytest.raises(ValueError, match="no device"):
            tc.add_chore(py_parsec.DEV_CUDA, 0x1234)
    with pytest.raises(TypeError, match="takes 2 arguments"):
        tp.insert_task(tc, (A, 0, 0))
    with pytest.raises(TypeError, match="must be a tile"):
        tp.insert_task(tc, 0, 1)
    with pytest.raises(TypeError, match="must be a tile"):
        tp.insert_task(tc, (A, 0), 1)
    with pytest.raises(TypeError, match="must be a tile"):
        tp.insert_task(tc, (0, 0, 0), 1)
    with pytest.raises(TypeError):
        tp.insert_task(tc, (A, 0, 0), "one")
    with pytest.raises(ValueError, match="expects 8x8"):
        tp.insert_task(tc, (small, 0, 0), 1)
    with pytest.raises(ValueError, match="no chore for device"):
        tp.insert_task(tc, (A, 0, 0), 1, device=py_parsec.DEV_CUDA)
    with pytest.raises(IndexError):
        tp.insert_task(tc, (A, 2, 0), 1)
    with pytest.raises(IndexError):
        dtd.tile_of(A, 5, 0)
    with pytest.raises(ValueError, match="insertion flags"):
        tp.insert_task(tc, (A, 0, 0, dtd.INPUT), 1)

    other = dtd.Taskpool()
    with pytest.raises(RuntimeError, match="not been added"):
        other.task_class("x", dtd.Data(dtd.INPUT, A))
    ctx.add_taskpool(other)
    with pytest.raises(ValueError, match="another taskpool"):
        other.insert_task(tc, (A, 0, 0), 1)
    other.free()


def test_param_errors(ctx):
    A = py_parsec.BlockCyclic(ctx, 8, 8, 4, 4)
    with pytest.raises(ValueError, match="INPUT, OUTPUT or INOUT"):
        dtd.Data(dtd.AFFINITY, A)
    with pytest.raises(TypeError):
        dtd.Data(dtd.INPUT)
    with pytest.raises(ValueError, match="format"):
        dtd.Value("x")
    with pytest.raises(TypeError, match="dtype"):
        dtd.Data(dtd.INPUT, shape=(4, 4), dtype=np.int64)
    d = dtd.Data(dtd.INPUT, shape=(4, 2), dtype=np.float32)
    assert (d.mb, d.nb, d.dtype) == (4, 2, np.float32)
    with pytest.raises(ValueError, match="flags"):
        dtd.tile_of(A, 0, 0, flags=dtd.INPUT)


def test_dropped_taskpool_is_completed_by_the_context(ctx):
    """A taskpool dropped with work pending, here in a reference cycle with its
    body, is completed by the next context wait and then released; dropping
    it never blocks."""
    A = py_parsec.BlockCyclic(ctx, 32, 32, 8, 8)
    count = Counter()
    tp = dtd.Taskpool()
    ctx.add_taskpool(tp)
    tc = tp.task_class("count", dtd.Data(dtd.INOUT | dtd.AFFINITY, A))

    def body(tile, tp=tp):      # the body refers to its taskpool
        count()

    tc.add_chore(py_parsec.DEV_CPU, body)
    for m in range(A.mt):
        for n in range(A.nt):
            tp.insert_task(tc, (A, m, n))
    ref = weakref.ref(tp)
    del tp, tc, body
    gc.collect()
    assert ref() is not None    # kept by the context until its work completes
    ctx.wait()
    assert count.value == A.nb_local_tiles
    gc.collect()
    assert ref() is None
    ctx.wait()                  # runs the deferred release


def test_garbage_collected_without_free(ctx):
    """Taskpools dropped without free() are released."""
    A = py_parsec.BlockCyclic(ctx, 16, 16, 8, 8)
    plain = dtd.Taskpool()
    ctx.add_taskpool(plain)
    plain.wait()
    del plain

    with_classes = dtd.Taskpool()
    ctx.add_taskpool(with_classes)
    tc = with_classes.task_class("c", dtd.Data(dtd.INOUT | dtd.AFFINITY, A))
    tc.add_chore(py_parsec.DEV_CPU, dtd.NOOP_CHORE)
    with_classes.insert_task(tc, (A, 0, 0))
    with_classes.wait()
    del with_classes
    with pytest.raises(RuntimeError, match="released"):
        tc.add_chore(py_parsec.DEV_CUDA, dtd.NOOP_CHORE)
    never_added = dtd.Taskpool()
    del never_added
    gc.collect()
    ctx.wait()


def test_write_without_affinity(ctx, tp):
    A = py_parsec.BlockCyclic(ctx, 16, 16, 8, 8)
    tc = tp.task_class("w", dtd.Data(dtd.INOUT, A))
    tc.add_chore(py_parsec.DEV_CPU, lambda t: None)
    if ctx.nb_nodes > 1:
        with pytest.raises(ValueError, match="AFFINITY"):
            tp.insert_task(tc, (A, 0, 0))
        tp.insert_task(tc, (A, 0, 0, dtd.AFFINITY))   # placed by this insertion
    else:
        tp.insert_task(tc, (A, 0, 0))
    tp.wait()


def test_body_return_codes(ctx, tp):
    A = py_parsec.BlockCyclic(ctx, 16, 16, 8, 8)
    again = tp.task_class("again", dtd.Data(dtd.INOUT | dtd.AFFINITY, A))
    calls = []

    def body(tile):
        calls.append(1)
        return py_parsec.HOOK_RETURN_AGAIN if len(calls) == 1 else None

    again.add_chore(py_parsec.DEV_CPU, body)
    tp.insert_task(again, (A, 0, 0))
    tp.wait()
    assert len(calls) == (2 if A.is_local(0, 0) else 0)

    bad = tp.task_class("bad", dtd.Data(dtd.INOUT | dtd.AFFINITY, A))
    bad.add_chore(py_parsec.DEV_CPU, lambda t: 7)
    tp.insert_task(bad, (A, 1, 1))
    if A.is_local(1, 1):
        with pytest.raises(ValueError, match="returned 7"):
            tp.wait()
    else:
        tp.wait()


def test_max_flows(ctx, tp):
    """A task class with MAX_FLOWS tiles is accepted by PaRSEC and runs."""
    assert 1 <= dtd.MAX_FLOWS <= dtd.MAX_PARAMS
    A = py_parsec.BlockCyclic(ctx, 8, 8 * dtd.MAX_FLOWS, 8, 8)
    params = [dtd.Data(dtd.INPUT, A)] * (dtd.MAX_FLOWS - 1) + [dtd.Data(dtd.INOUT | dtd.AFFINITY, A)]
    tc = tp.task_class("max", *params)
    tc.add_chore(py_parsec.DEV_CPU, dtd.NOOP_CHORE)
    tp.insert_task(tc, *[(A, 0, n) for n in range(dtd.MAX_FLOWS)])
    tp.wait()
    with pytest.raises(ValueError, match="Data parameters"):
        tp.task_class("too_many", *[dtd.Data(dtd.INPUT, A)] * (dtd.MAX_FLOWS + 1))


def test_many_values(ctx, tp):
    """Every parameter count up to MAX_PARAMS goes through the C helpers."""
    A = py_parsec.BlockCyclic(ctx, 8, 8, 8, 8)
    seen = []
    nvalues = dtd.MAX_PARAMS - 1
    tc = tp.task_class("many", dtd.Data(dtd.INOUT | dtd.AFFINITY, A), *[dtd.Value("q")] * nvalues)
    tc.add_chore(py_parsec.DEV_CPU, lambda tile, *values: seen.append(values))
    tp.insert_task(tc, (A, 0, 0), *range(nvalues))
    tp.wait()
    assert seen == ([tuple(range(nvalues))] if A.is_local(0, 0) else [])


def test_insert_tasks(ctx, tp):
    """The batch insertion gives the same result as insert_task."""
    n, b = 40, 4
    A = py_parsec.BlockCyclic(ctx, n, n, b, b)
    tc = tp.task_class("set", dtd.Data(dtd.INOUT | dtd.AFFINITY, A), dtd.Value("i"), dtd.Value("d"))

    def body(tile, m, scale):
        tile += m * scale

    tc.add_chore(py_parsec.DEV_CPU, body)
    tasks = [((A, m, k), m, 0.5) for m in range(A.mt) for k in range(A.nt)]
    assert tp.insert_tasks(tc, tasks * 30) == 30 * A.mt * A.nt   # several batches
    assert tp.insert_tasks(tc, iter([])) == 0
    tp.wait()
    expected = np.repeat(np.arange(A.mt) * 0.5 * 30, b)[:, None] * np.ones((1, n))
    assert_matches(A, expected)


def test_insert_tasks_error_keeps_earlier_tasks(ctx, tp):
    A = py_parsec.BlockCyclic(ctx, 8, 8, 4, 4)
    inc = tp.task_class("inc", dtd.Data(dtd.INOUT | dtd.AFFINITY, A))
    inc.add_chore(py_parsec.DEV_CPU, lambda t: t.__iadd__(1))
    with pytest.raises(IndexError):
        tp.insert_tasks(inc, [((A, 0, 0),), ((A, 1, 1),), ((A, 9, 9),), ((A, 0, 1),)])
    with pytest.raises(TypeError, match="takes 1 arguments"):
        tp.insert_tasks(inc, [((A, 0, 0),), ()])
    tp.wait()
    expected = np.zeros((8, 8))
    expected[:4, :4] = 2
    expected[4:, 4:] = 1
    assert_matches(A, expected)


def test_data_flush(ctx, tp):
    A = py_parsec.BlockCyclic(ctx, 8, 8, 4, 4)
    inc = tp.task_class("inc", dtd.Data(dtd.INOUT | dtd.AFFINITY, A))
    inc.add_chore(py_parsec.DEV_CPU, lambda t: t.__iadd__(1))
    tp.insert_task(inc, (A, 1, 0))
    tp.data_flush((A, 1, 0))
    tp.data_flush(dtd.tile_of(A, 0, 1))
    with pytest.raises(TypeError):
        tp.data_flush(A)
    tp.wait()
    expected = np.zeros((8, 8))
    expected[4:, :4] = 1
    assert_matches(A, expected)


def test_context_wait_with_idle_taskpools(ctx, tp):
    """A DTD taskpool that is alive but has nothing pending does not hold up
    a context wait, and can be fed again afterwards."""
    A = py_parsec.BlockCyclic(ctx, 16, 16, 8, 8)
    tc = tp.task_class("noop", dtd.Data(dtd.INOUT | dtd.AFFINITY, A))
    tc.add_chore(py_parsec.DEV_CPU, dtd.NOOP_CHORE)
    for m in range(A.mt):
        for n in range(A.nt):
            tp.insert_task(tc, (A, m, n))
    tp.wait()
    other = dtd.Taskpool()
    ctx.add_taskpool(other)
    ctx.wait()
    tp.insert_task(tc, (A, 0, 0))
    ctx.wait()
    other.free()


def test_insert_tasks_iterable_inserting_keeps_order(ctx, tp):
    """Tasks inserted by the iterable of insert_tasks() come after the tasks
    it produced before: the result is that of the sequential loop."""
    A = py_parsec.BlockCyclic(ctx, 4, 4, 4, 4)
    inc = tp.task_class("inc", dtd.Data(dtd.INOUT | dtd.AFFINITY, A))
    dbl = tp.task_class("dbl", dtd.Data(dtd.INOUT | dtd.AFFINITY, A))
    inc.add_chore(py_parsec.DEV_CPU, lambda a: a.__iadd__(1))
    dbl.add_chore(py_parsec.DEV_CPU, lambda a: a.__imul__(2))

    def tasks():
        for _ in range(4):
            yield ((A, 0, 0),)
            tp.insert_task(dbl, (A, 0, 0))

    assert tp.insert_tasks(inc, tasks()) == 4
    tp.wait()
    assert_matches(A, np.full((4, 4), 30.0))     # ((((0+1)*2+1)*2+1)*2+1)*2


def test_insert_tasks_iterable_waiting(ctx, tp):
    """A wait made by the iterable completes the tasks it produced before
    (a distributed run would not terminate if their tiles were dropped)."""
    last = ctx.nb_nodes - 1
    A = py_parsec.BlockCyclic(ctx, 4, 4, 4, 4, P=ctx.nb_nodes, name="A")
    B = py_parsec.BlockCyclic(ctx, 4 * ctx.nb_nodes, 4, 4, 4, P=ctx.nb_nodes, name="B")
    inc = tp.task_class("inc", dtd.Data(dtd.INPUT | dtd.AFFINITY, B), dtd.Data(dtd.INOUT, A))
    inc.add_chore(py_parsec.DEV_CPU, lambda b, a: a.__iadd__(1))

    def tasks():
        for i in range(6):
            yield ((B, last, 0), (A, 0, 0))
            if i == 2:
                tp.wait()

    assert tp.insert_tasks(inc, tasks()) == 6
    tp.wait()
    assert_matches(A, np.full((4, 4), 6.0))


def test_insert_tasks_nested_and_freed(ctx):
    A = py_parsec.BlockCyclic(ctx, 8, 8, 4, 4)
    tp = dtd.Taskpool()
    ctx.add_taskpool(tp)
    inc = tp.task_class("inc", dtd.Data(dtd.INOUT | dtd.AFFINITY, A))
    inc.add_chore(py_parsec.DEV_CPU, lambda a: a.__iadd__(1))

    def inner():
        yield ((A, 1, 1),)

    def outer():
        yield ((A, 0, 0),)
        tp.insert_tasks(inc, inner())
        yield ((A, 0, 0),)

    assert tp.insert_tasks(inc, outer()) == 2
    tp.wait()
    expected = np.zeros((8, 8))
    expected[:4, :4] = 2
    expected[4:, 4:] = 1
    assert_matches(A, expected)

    def freeing():
        yield ((A, 0, 0),)
        tp.free()
        yield ((A, 0, 0),)

    with pytest.raises(RuntimeError, match="released"):
        tp.insert_tasks(inc, freeing())
    expected[:4, :4] = 3     # the task before free() was inserted and completed
    assert_matches(A, expected)


def test_drop_taskpool_during_context_wait(ctx):
    """A taskpool dropped and a matrix created by a body while the context
    waits: the release is not run from the body's thread or during the wait."""
    idle = dtd.Taskpool()
    ctx.add_taskpool(idle)
    idle.wait()
    holder = [idle]
    del idle
    tp = dtd.Taskpool()
    ctx.add_taskpool(tp)
    A = py_parsec.BlockCyclic(ctx, 4, 4, 4, 4)

    def body(a):
        if holder:
            holder.clear()
            py_parsec.BlockCyclic(ctx, 4, 4, 4, 4)
        a += 1

    tc = tp.task_class("inc", dtd.Data(dtd.INOUT | dtd.AFFINITY, A))
    tc.add_chore(py_parsec.DEV_CPU, body)
    for _ in range(5):
        tp.insert_task(tc, (A, 0, 0))
    ctx.wait()
    assert_matches(A, np.full((4, 4), 5.0))
    tp.free()


def test_stored_exception_has_no_locals(ctx, tp):
    A = py_parsec.BlockCyclic(ctx, 4, 4, 4, 4)
    tc = tp.task_class("boom", dtd.Data(dtd.INOUT | dtd.AFFINITY, A))

    def body(tile):
        raise KeyError("boom")

    tc.add_chore(py_parsec.DEV_CPU, body)
    tp.insert_task(tc, (A, 0, 0))
    if A.is_local(0, 0):
        with pytest.raises(KeyError) as info:
            tp.wait()
        tb = info.value.__traceback__
        while tb.tb_next is not None:
            tb = tb.tb_next
        assert tb.tb_frame.f_code.co_name == "body"
        assert "tile" not in tb.tb_frame.f_locals
    else:
        tp.wait()
