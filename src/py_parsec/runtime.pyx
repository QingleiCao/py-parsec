# Copyright (c) 2026      The University of Tennessee and The University
#                         of Tennessee Research Foundation.  All rights
#                         reserved.
"""PaRSEC contexts and taskpools.

A :class:`Context` owns the PaRSEC runtime of the process (``parsec_init`` /
``parsec_fini``). Every domain specific language produces the same kind of
object, a ``parsec_taskpool_t``, so all of them are wrapped by
:class:`Taskpool` and scheduled on the context the same way, with
:meth:`Context.add_taskpool`. DTD taskpools (:class:`py_parsec.dtd.Taskpool`)
and the PTG taskpools returned by library calls are interchangeable there.

Lifetimes
---------
The context keeps every taskpool whose work has not been completed by a wait,
so a taskpool can be dropped at any time: its work still completes at the next
:meth:`Context.wait` (or :meth:`Context.fini`). Finalizers never call into
PaRSEC; a taskpool or matrix collected by Python is queued on its context and
released by the next call that enters the runtime from the thread that
created the context (``start``, ``wait``, ``add_taskpool``, creating a
taskpool or a matrix, ``fini``), outside of any call that may run tasks.

Threads
-------
Drive the runtime (insertions, flushes, waits, frees) from one thread at a
time, normally the one that created the context. Task bodies run on PaRSEC
threads and must not call these functions.
"""

import atexit
import os
import weakref

cimport cython
from libc.stdint cimport uint64_t, uintptr_t
from libc.stdlib cimport free, malloc
from cpython.pycapsule cimport PyCapsule_GetPointer, PyCapsule_IsValid
from cpython.pythread cimport PyThread_get_thread_ident

from py_parsec cimport cparsec as c

DEV_CPU = c.PARSEC_DEV_CPU
DEV_CUDA = c.PARSEC_DEV_CUDA
DEV_HIP = c.PARSEC_DEV_HIP
DEV_LEVEL_ZERO = c.PARSEC_DEV_LEVEL_ZERO
DEV_ALL = c.PARSEC_DEV_ALL

HOOK_RETURN_DONE = c.PARSEC_HOOK_RETURN_DONE
HOOK_RETURN_AGAIN = c.PARSEC_HOOK_RETURN_AGAIN
HOOK_RETURN_NEXT = c.PARSEC_HOOK_RETURN_NEXT
HOOK_RETURN_DISABLE = c.PARSEC_HOOK_RETURN_DISABLE
HOOK_RETURN_ASYNC = c.PARSEC_HOOK_RETURN_ASYNC
HOOK_RETURN_ERROR = c.PARSEC_HOOK_RETURN_ERROR


def version():
    """Return the (major, minor, release) version of the PaRSEC library."""
    cdef int major = 0, minor = 0, release = 0
    c.parsec_version(&major, &minor, &release)
    return (major, minor, release)


def build_info():
    """Describe the PaRSEC library in use (``parsec_version_ex``), as a dict."""
    cdef char buf[8192]
    buf[0] = 0
    c.parsec_version_ex(sizeof(buf), buf)
    info = {}
    for line in buf.decode(errors="replace").splitlines():
        key, sep, value = line.partition("=")
        if sep:
            info[key.strip()] = value.strip()
    return info


def _check_build():
    """Refuse to run with a PaRSEC other than the one the bindings were compiled
    against: they read PaRSEC structures whose layout depends on its build."""
    if os.environ.get("PY_PARSEC_SKIP_BUILD_CHECK"):
        return
    built = {"git_hash": c.pyparsec_built_hash().decode(),
             "compile_date": c.pyparsec_built_date().decode()}
    info = build_info()
    found = {key: info.get(key, "") for key in built}
    if any(built[key] and built[key] != found[key] for key in built):
        raise ImportError(
            "py_parsec was compiled against another PaRSEC build than the libparsec it loaded "
            f"(compiled against git {built['git_hash']}, configured {built['compile_date']}; "
            f"loaded git {found['git_hash']}, configured {found['compile_date']}). Reinstall py_parsec "
            "against this PaRSEC, or set PY_PARSEC_SKIP_BUILD_CHECK=1 to load it anyway.")


_check_build()


def address_of(obj):
    """Return the C pointer held by a PyCapsule or an int as an int."""
    if isinstance(obj, int):
        return obj
    if PyCapsule_IsValid(obj, NULL):
        return <uintptr_t>PyCapsule_GetPointer(obj, NULL)
    raise TypeError(f"expected an address (int) or an unnamed PyCapsule, got {type(obj).__name__}")


# The context of the process, from its creation to its fini(). PaRSEC cannot be
# initialized again in a process once finalized, so there is at most one.
cdef Context _live = None
cdef bint _initialized_once = False
cdef uint64_t _next_uid = 1


cdef Context live_context():
    return _live


def runtime_initialized():
    """True between the creation of the :class:`Context` and its :meth:`~Context.fini`."""
    return _live is not None


@atexit.register
def _fini_at_exit():
    # Runs before the Py_AtExit handlers, one of which finalizes MPI for mpi4py.
    if _live is not None:
        _live.fini()


cdef raise_errors(list errors):
    """Raise the first exception of ``errors``, noting how many others there were."""
    if not errors:
        return
    first = errors[0]
    if len(errors) > 1 and hasattr(first, "add_note"):
        others = ", ".join(repr(e) for e in errors[1:6])
        more = f", ... ({len(errors) - 6} more)" if len(errors) > 6 else ""
        first.add_note(f"{len(errors) - 1} other exception(s) were raised by task bodies: {others}{more}")
    raise first


# Neither Context, Taskpool nor TiledMatrix is ever the link that closes a
# reference cycle the garbage collector must break, so it never clears their
# attributes: taskpools and matrices need their context until they are released.
@cython.no_gc_clear
cdef class Context:
    """The PaRSEC runtime of the process.

    Parameters
    ----------
    nb_cores : int
        Number of compute threads, the calling thread included; a value <= 0
        uses every core available to the process.
    args : sequence of str, optional
        PaRSEC options, as accepted on the command line after ``--``
        (for instance ``["--mca", "runtime_comm_short_limit", "0"]``).

    PaRSEC can be initialized once per process: create one context, use it
    for the whole run, and finalize it with :meth:`fini` (or a ``with``
    block, or at interpreter exit). PaRSEC initializes MPI if the application
    did not and then finalizes it in :meth:`fini`; import ``mpi4py.MPI``
    first to keep using MPI after that.

    The compute threads start when work is scheduled (:meth:`start` starts
    them explicitly) and :meth:`wait` completes every taskpool of the context.
    """

    def __cinit__(self, int nb_cores=-1, args=None):
        global _live, _initialized_once
        cdef int argc
        cdef char **pargv
        cdef Py_ssize_t i

        self._ctx = NULL
        self._c_argv = NULL
        self._dtd_arenas = {}
        self._pending = set()
        self._taskpools = weakref.WeakSet()
        self._collections = weakref.WeakSet()
        self._deferred = []
        self._owner = PyThread_get_thread_ident()
        self._in_runtime = 0
        self._batch = None
        if _live is not None:
            raise RuntimeError("the process already has a PaRSEC context; PaRSEC supports one per process")
        if _initialized_once:
            raise RuntimeError("PaRSEC cannot be initialized again in a process once it has been finalized")
        if c.pyparsec_mpi_finalized():
            raise RuntimeError("MPI has been finalized: PaRSEC cannot be initialized")

        self._argv = [a.encode() if isinstance(a, str) else bytes(a) for a in (args or ())]
        argc = <int>len(self._argv)
        self._c_argv = <char **>malloc((argc + 1) * sizeof(char *))
        if self._c_argv == NULL:
            raise MemoryError()
        for i in range(argc):
            self._c_argv[i] = <char *>self._argv[i]
        self._c_argv[argc] = NULL
        pargv = self._c_argv

        with nogil:
            self._ctx = c.parsec_init(nb_cores, &argc, &pargv)
        if self._ctx == NULL:
            raise RuntimeError("parsec_init failed")
        _initialized_once = True
        _live = self

        self.rank = c.parsec_context_query(self._ctx, c.PARSEC_CONTEXT_QUERY_RANK)
        self.nb_nodes = c.parsec_context_query(self._ctx, c.PARSEC_CONTEXT_QUERY_NODES)
        self.nb_cores = c.parsec_context_query(self._ctx, c.PARSEC_CONTEXT_QUERY_CORES)
        if self.nb_nodes == 1:
            # In a single-process run the communication engine, which also
            # performs local copies, is only made ready by parsec_context_wait()
            # (fixed by ICLDisco/parsec#836): a parsec_taskpool_wait() on a PTG
            # taskpool issued earlier would crash. One empty start/wait makes it
            # ready for the whole run.
            with nogil:
                c.parsec_context_start(self._ctx)
                c.parsec_context_wait(self._ctx)

    def __dealloc__(self):
        # The process keeps its context until fini(), so the runtime is
        # always finalized when a context is collected.
        free(self._c_argv)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.fini()
        return False

    cdef parsec_context_t *ptr(self) except NULL:
        if self._ctx == NULL:
            raise RuntimeError("the PaRSEC context has been finalized")
        return self._ctx

    cdef bint is_active(self):
        return self._ctx != NULL and c.pyparsec_context_is_active(self._ctx)

    cdef int ensure_started(self) except -1:
        cdef c.parsec_context_t *ctx = self.ptr()
        if not c.pyparsec_context_is_active(ctx):
            c.parsec_context_start(ctx)
        return 0

    cdef int drain(self, bint force=False) except -1:
        """Run the releases queued by finalizers: only from the thread that
        created the context and outside of any PaRSEC call that may run tasks
        (a context wait, for instance, has already visited every taskpool)."""
        cdef Deferred release
        if not force and (self._in_runtime or PyThread_get_thread_ident() != self._owner):
            return 0
        while self._deferred and self._ctx != NULL:
            release = self._deferred.pop(0)
            release.run()
        return 0

    cdef defer(self, Deferred release):
        self._deferred.append(release)

    cdef int submit_batch(self) except -1:
        """Insert the tasks an insert_tasks() call has converted so far, so
        that a call made from its iterable keeps the order of the tasks."""
        if self._batch is not None:
            if PyThread_get_thread_ident() != self._batch.thread:
                raise RuntimeError("insert_tasks() is running in another thread")
            self._batch.submit()
        return 0

    @property
    def address(self):
        """The ``parsec_context_t *`` as an int, for C libraries driven from Python."""
        return <uintptr_t>self.ptr()

    @property
    def active(self):
        """True while the compute threads run (between a start and the end of a wait)."""
        return self.is_active()

    @property
    def finalized(self):
        """True once :meth:`fini` has run."""
        return self._ctx == NULL

    def nb_devices(self, int device_type):
        """Number of devices of ``device_type`` (``DEV_CUDA``, ...) PaRSEC uses."""
        cdef int rc = c.parsec_context_query(self.ptr(), c.PARSEC_CONTEXT_QUERY_DEVICES, device_type)
        return max(rc, 0)

    def start(self):
        """Start the compute threads; returns False if they were already running.

        Inserting tasks and waiting start them as needed; start them
        explicitly to have PTG taskpools run while the application does
        something else.
        """
        cdef c.parsec_context_t *ctx = self.ptr()
        self.drain()
        return c.parsec_context_start(ctx) == 0

    def wait(self):
        """Complete every taskpool of the context (``parsec_context_wait``).

        The collections used by DTD taskpools are flushed first, so their data
        is back on its owners on return. The compute threads are then idle
        until more work is scheduled. The first exception raised by a Python
        task body since the last wait is raised here.
        """
        cdef c.parsec_context_t *ctx = self.ptr()
        cdef Taskpool tp
        cdef int rc
        self.submit_batch()
        self.drain()
        if not c.pyparsec_context_is_active(ctx):
            if not self._pending:
                return
            c.parsec_context_start(ctx)
        pending = list(self._pending)
        for tp in pending:
            tp.flush_collections()
        self._in_runtime += 1
        try:
            with nogil:
                rc = c.parsec_context_wait(ctx)
        finally:
            self._in_runtime -= 1
        if rc != c.PARSEC_SUCCESS:
            raise RuntimeError(f"parsec_context_wait failed ({rc})")
        errors = []
        for tp in pending:
            tp._pending = False
            errors.extend(tp.take_errors())
        self._pending.difference_update(pending)
        raise_errors(errors)

    def add_taskpool(self, Taskpool tp not None):
        """Enqueue a taskpool, of any DSL, in this context."""
        cdef int rc
        cdef c.parsec_context_t *ctx = self.ptr()
        self.drain()
        if tp._context is not self:
            raise ValueError("the taskpool was created for another context")
        if tp._enqueued:
            raise RuntimeError("the taskpool has already been added to a context")
        rc = c.parsec_context_add_taskpool(ctx, tp.ptr())
        if rc != c.PARSEC_SUCCESS:
            raise RuntimeError(f"parsec_context_add_taskpool failed ({rc})")
        tp._enqueued = True
        tp.mark_pending()

    def fini(self):
        """Complete and release the taskpools and collections of this context,
        then finalize PaRSEC. Calling it again does nothing."""
        global _live
        cdef Taskpool tp
        if self._ctx == NULL:
            return
        errors = []
        try:
            self.wait()     # also inserts the tasks of an open insert_tasks() batch
        except BaseException as e:
            errors.append(e)
        for tp in list(self._taskpools):
            try:
                tp.release()
            except BaseException as e:
                errors.append(e)
        try:
            self.drain(True)
        except BaseException as e:
            errors.append(e)
        self._in_runtime += 1
        if self.is_active():
            with nogil:
                c.parsec_context_wait(self._ctx)
        # Matrices still referenced keep their storage, so that the tile
        # arrays taken from them stay valid; their PaRSEC side goes now.
        for dc in list(self._collections):
            dc._detach()
        for arena_id in self._dtd_arenas.values():
            c.parsec_dtd_free_arena_datatype(self._ctx, arena_id)
        self._dtd_arenas.clear()
        with nogil:
            c.parsec_fini(&self._ctx)
        self._in_runtime -= 1
        self._ctx = NULL
        self._pending.clear()
        self._deferred.clear()
        _live = None
        if errors:
            raise errors[0]

    cdef int dtd_tile_arena(self, int mtype, int mb, int nb, int ld) except -1:
        """DTD arena id of a full mb x nb tile, created on first use."""
        key = (mtype, mb, nb, ld)
        arena_id = self._dtd_arenas.get(key)
        if arena_id is None:
            arena_id = c.pyparsec_dtd_tile_arena_new(self.ptr(), mtype, mb, nb, ld)
            if arena_id < 0:
                raise RuntimeError(f"cannot create a {mb}x{nb} arena datatype")
            self._dtd_arenas[key] = arena_id
        return arena_id

    cdef register_collection(self, object collection):
        self.drain()
        self._collections.add(collection)

    def __repr__(self):
        if self._ctx == NULL:
            return "<py_parsec.Context finalized>"
        return f"<py_parsec.Context rank {self.rank}/{self.nb_nodes}, {self.nb_cores} cores>"


cdef void _default_destructor(c.parsec_taskpool_t *tp) noexcept nogil:
    c.parsec_taskpool_free(tp)


cdef class Deferred:
    cdef run(self):
        pass


cdef class PendingBatch:
    cdef int submit(self) except -1:
        return 0


@cython.final
cdef class Release(Deferred):
    """The release of a taskpool collected by Python, run later by Context.drain()."""

    cdef run(self):
        cdef uintptr_t addr
        if self.tp == NULL:
            return
        for addr in self.task_classes or ():
            c.parsec_dtd_task_class_release(self.tp, <c.parsec_task_class_t *>addr)
        self.destruct(self.tp)
        self.tp = NULL
        self.task_classes = None
        self.keepalive = None   # the collections outlive the C taskpool


@cython.final
cdef class CollectionRelease(Deferred):
    """The release of a matrix collected by Python, run later by Context.drain()."""

    cdef run(self):
        if self.desc != NULL:
            c.pyparsec_tiled_matrix_free(self.desc, self.dtd_registered)
            self.desc = NULL
        if self.storage != NULL:
            self.free_storage(self.storage)
            self.storage = NULL


cdef CollectionRelease make_collection_release(c.parsec_tiled_matrix_t *desc, bint dtd_registered,
                                               void *storage, c.parsec_data_free_t free_storage):
    cdef CollectionRelease r = CollectionRelease.__new__(CollectionRelease)
    r.desc = desc
    r.dtd_registered = dtd_registered
    r.storage = storage
    r.free_storage = free_storage
    return r


cdef Release make_release(c.parsec_taskpool_t *tp, taskpool_destructor_t destruct,
                          list task_classes, object keepalive):
    cdef Release r = Release.__new__(Release)
    r.tp = tp
    r.destruct = destruct
    r.task_classes = task_classes
    r.keepalive = keepalive
    return r


@cython.no_gc_clear   # see Context
cdef class Taskpool:
    """A PaRSEC taskpool, whatever DSL created it.

    Library calls that build PTG taskpools return instances of this class;
    :meth:`from_address` wraps a ``parsec_taskpool_t *`` created by C code
    (for instance a JDF compiled into a separate shared library), so it can be
    scheduled next to DTD taskpools in the same context.

    :meth:`free` completes the work of the taskpool, if needed, and releases
    it. A taskpool that is simply dropped is completed by the next
    :meth:`Context.wait` and released after that.
    """

    def __cinit__(self, *args, **kwargs):
        global _next_uid
        self._tp = NULL
        self._destruct = _default_destructor
        self._context = _live
        self._keepalive = []
        self._enqueued = False
        self._pending = False
        self.uid = _next_uid
        _next_uid += 1
        if _live is None:
            raise RuntimeError("create a py_parsec.Context before a taskpool")
        _live.drain()
        _live._taskpools.add(self)

    def __dealloc__(self):
        # Subclasses queue their own release and leave _tp NULL. A taskpool
        # with work pending is referenced by its context, so it cannot be
        # collected: whatever reaches this point can be freed without waiting.
        if self._tp != NULL and self._context is not None and self._context._ctx != NULL:
            self._context.defer(make_release(self._tp, self._destruct, None, self._keepalive))
        self._tp = NULL

    @staticmethod
    def from_address(address, keepalive=()):
        """Take ownership of a ``parsec_taskpool_t *`` (int or PyCapsule).

        The taskpool is released with ``parsec_taskpool_free()``;
        ``keepalive`` lists Python objects (data collections, ...) that must
        outlive it.
        """
        cdef uintptr_t addr = address_of(address)
        cdef Taskpool tp
        if addr == 0:
            raise ValueError("NULL taskpool")
        tp = Taskpool.__new__(Taskpool)
        tp._tp = <c.parsec_taskpool_t *>addr
        tp._keepalive = list(keepalive)
        return tp

    cdef parsec_taskpool_t *ptr(self) except NULL:
        if self._tp == NULL:
            raise RuntimeError("the taskpool has been released")
        return self._tp

    cdef Context enqueued_context(self):
        self.ptr()
        if not self._enqueued:
            raise RuntimeError("the taskpool has not been added to a context")
        return self._context

    cdef int mark_pending(self) except -1:
        if not self._pending:
            self._context._pending.add(self)
            self._pending = True
        return 0

    @property
    def address(self):
        """The ``parsec_taskpool_t *`` as an int."""
        return <uintptr_t>self.ptr()

    @property
    def context(self):
        """The context the taskpool was added to, or None."""
        return self._context if self._enqueued else None

    @property
    def released(self):
        """True once the taskpool has been freed."""
        return self._tp == NULL

    def wait(self):
        """Complete this taskpool (``parsec_taskpool_wait``), starting the
        compute threads if needed. Other taskpools keep running.

        For a DTD taskpool, the collections its tasks used are flushed first.
        The first exception raised by one of its Python task bodies is raised.
        """
        cdef c.parsec_taskpool_t *tp
        cdef Context ctx = self.enqueued_context()
        cdef int rc
        ctx.submit_batch()
        tp = self.ptr()
        ctx.drain()
        ctx.ensure_started()
        self.flush_collections()
        ctx._in_runtime += 1
        try:
            with nogil:
                rc = c.parsec_taskpool_wait(tp)
        finally:
            ctx._in_runtime -= 1
        if rc < 0:
            raise RuntimeError(f"parsec_taskpool_wait failed ({rc})")
        if self._pending:
            self._pending = False
            ctx._pending.discard(self)
        raise_errors(self.take_errors())

    def free(self):
        """Release the taskpool, completing its work first if needed.
        Calling it again does nothing."""
        if self._tp != NULL and self._context._ctx != NULL:
            self._context.submit_batch()
        self.release()

    cdef int flush_collections(self) except -1:
        """Hand the data back to the collections before a wait (DTD)."""
        return 0

    cdef list take_errors(self):
        return []

    cdef release(self):
        if self._tp == NULL:
            return
        try:
            if self._pending and self._enqueued and self._context._ctx != NULL:
                Taskpool.wait(self)
        finally:
            # A taskpool whose wait failed is kept: the error tells why. One
            # whose bodies raised has completed and is freed.
            if not self._pending:
                self.destroy()

    cdef destroy(self):
        if self._context._ctx != NULL:
            self._destruct(self._tp)
        self.forget()

    cdef forget(self):
        """Drop the Python side of the taskpool."""
        self._tp = NULL
        if self._pending:
            self._pending = False
            self._context._pending.discard(self)
        self._context._taskpools.discard(self)
        self._keepalive = []

    def __repr__(self):
        where = "released" if self._tp == NULL else hex(<uintptr_t>self._tp)
        return f"<py_parsec.{type(self).__name__} {where}>"
