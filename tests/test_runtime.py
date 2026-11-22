# Copyright (c) 2026      The University of Tennessee and The University
#                         of Tennessee Research Foundation.  All rights
#                         reserved.

import pytest

import py_parsec
from py_parsec import dtd


def test_version():
    major, minor, release = py_parsec.version()
    assert major >= 4
    info = py_parsec.build_info()
    assert info["version"].startswith(f"{major}.{minor}")
    assert "git_hash" in info
    assert py_parsec.__version__


def test_context(ctx):
    assert 0 <= ctx.rank < ctx.nb_nodes
    assert ctx.nb_cores >= 1
    assert ctx.address != 0
    assert ctx.nb_devices(py_parsec.DEV_CPU) >= 1
    assert not ctx.finalized
    assert "Context" in repr(ctx)


def test_one_context_per_process(ctx):
    with pytest.raises(RuntimeError, match="already has a PaRSEC context"):
        py_parsec.Context()


def test_start_and_wait(ctx):
    ctx.wait()
    ctx.start()
    assert ctx.active
    assert not ctx.start()  # already running
    ctx.wait()
    assert not ctx.active
    ctx.wait()              # nothing to do on an idle context
    assert not ctx.active


def test_taskpool_lifecycle(ctx):
    tp = dtd.Taskpool()
    assert tp.context is None
    with pytest.raises(RuntimeError, match="not been added"):
        tp.wait()
    ctx.add_taskpool(tp)
    assert tp.context is ctx
    with pytest.raises(RuntimeError, match="already been added"):
        ctx.add_taskpool(tp)
    ctx.wait()
    tp.wait()               # starts the idle context
    assert ctx.active
    tp.free()
    assert tp.released
    with pytest.raises(RuntimeError, match="released"):
        tp.wait()
    tp.free()               # idempotent


def test_context_wait_completes_dtd_taskpools(ctx):
    tp = dtd.Taskpool()
    ctx.add_taskpool(tp)
    ctx.wait()
    assert not ctx.active
    tp.free()


def test_free_completes_pending_work(ctx):
    tp = dtd.Taskpool()
    ctx.add_taskpool(tp)
    ctx.wait()
    tp.free()               # never waited since it was added: free() waits first
    assert tp.released


def test_taskpool_from_address_rejects_null(ctx):
    with pytest.raises(ValueError):
        py_parsec.Taskpool.from_address(0)
    with pytest.raises(TypeError):
        py_parsec.Taskpool.from_address("0x1234")
