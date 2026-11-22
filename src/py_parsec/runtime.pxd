# Copyright (c) 2026      The University of Tennessee and The University
#                         of Tennessee Research Foundation.  All rights
#                         reserved.

from libc.stdint cimport uint64_t

from py_parsec.cparsec cimport parsec_context_t, parsec_taskpool_t

ctypedef void (*taskpool_destructor_t)(parsec_taskpool_t *tp) noexcept nogil


cdef class Context:
    cdef parsec_context_t *_ctx
    cdef char **_c_argv
    cdef list _argv
    cdef dict _dtd_arenas
    cdef set _pending          # taskpools whose work no wait has completed (strong)
    cdef object _taskpools     # every taskpool created on this context (weak)
    cdef object _collections   # every data collection created on this context (weak)
    cdef list _deferred        # releases requested by finalizers, run by drain()
    cdef readonly int rank
    cdef readonly int nb_nodes
    cdef readonly int nb_cores
    cdef object __weakref__

    cdef parsec_context_t *ptr(self) except NULL
    cdef bint is_active(self)
    cdef int ensure_started(self) except -1
    cdef int drain(self) except -1
    cdef int dtd_tile_arena(self, int mtype, int mb, int nb, int ld) except -1
    cdef register_collection(self, object collection)
    cdef defer(self, Release release)


cdef class Release:
    cdef parsec_taskpool_t *tp
    cdef taskpool_destructor_t destruct
    cdef list task_classes
    cdef object keepalive

    cdef run(self)


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
cdef raise_errors(list errors)
