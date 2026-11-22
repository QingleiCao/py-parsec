# Copyright (c) 2026      The University of Tennessee and The University
#                         of Tennessee Research Foundation.  All rights
#                         reserved.

from libc.stdint cimport uint64_t

from py_parsec.cparsec cimport (parsec_context_t, parsec_taskpool_t, parsec_tiled_matrix_t,
                                parsec_data_free_t)

ctypedef void (*taskpool_destructor_t)(parsec_taskpool_t *tp) noexcept nogil


cdef class Deferred:
    """A release queued by a finalizer, run by Context.drain()."""
    cdef run(self)


cdef class Release(Deferred):
    cdef parsec_taskpool_t *tp
    cdef taskpool_destructor_t destruct
    cdef list task_classes
    cdef object keepalive


cdef class CollectionRelease(Deferred):
    cdef parsec_tiled_matrix_t *desc
    cdef bint dtd_registered
    cdef void *storage
    cdef parsec_data_free_t free_storage


cdef class PendingBatch:
    """Tasks converted by insert_tasks() but not inserted yet."""
    cdef unsigned long thread
    cdef int submit(self) except -1


cdef class Context:
    cdef parsec_context_t *_ctx
    cdef char **_c_argv
    cdef list _argv
    cdef dict _dtd_arenas
    cdef set _pending          # taskpools whose work no wait has completed (strong)
    cdef object _taskpools     # every taskpool created on this context (weak)
    cdef object _collections   # every data collection created on this context (weak)
    cdef list _deferred        # releases requested by finalizers, run by drain()
    cdef unsigned long _owner  # the thread that created the context
    cdef int _in_runtime       # depth of PaRSEC calls that may run tasks, made by the owner
    cdef PendingBatch _batch   # the insert_tasks() batch being filled, if any
    cdef readonly int rank
    cdef readonly int nb_nodes
    cdef readonly int nb_cores
    cdef object __weakref__

    cdef parsec_context_t *ptr(self) except NULL
    cdef bint is_active(self)
    cdef int ensure_started(self) except -1
    cdef int drain(self, bint force=*) except -1
    cdef int submit_batch(self) except -1
    cdef int dtd_tile_arena(self, int mtype, int mb, int nb, int ld) except -1
    cdef register_collection(self, object collection)
    cdef defer(self, Deferred release)


cdef class Taskpool:
    cdef parsec_taskpool_t *_tp
    cdef taskpool_destructor_t _destruct
    cdef Context _context
    cdef list _keepalive
    cdef bint _enqueued
    cdef bint _pending
    cdef readonly uint64_t uid
    cdef object __weakref__

    cdef parsec_taskpool_t *ptr(self) except NULL
    cdef Context enqueued_context(self)
    cdef int mark_pending(self) except -1
    cdef int flush_collections(self) except -1
    cdef list take_errors(self)
    cdef release(self)
    cdef destroy(self)
    cdef forget(self)


cdef Context live_context()
cdef Release make_release(parsec_taskpool_t *tp, taskpool_destructor_t destruct,
                          list task_classes, object keepalive)
cdef CollectionRelease make_collection_release(parsec_tiled_matrix_t *desc, bint dtd_registered,
                                               void *storage, parsec_data_free_t free_storage)
cdef raise_errors(list errors)
