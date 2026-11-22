# Copyright (c) 2026      The University of Tennessee and The University
#                         of Tennessee Research Foundation.  All rights
#                         reserved.

from libc.stdint cimport uint64_t

from py_parsec.cparsec cimport (parsec_data_collection_t, parsec_tiled_matrix_t,
                                parsec_matrix_block_cyclic_t, parsec_data_free_t)
from py_parsec.runtime cimport Context


cdef object host_array(void *ptr, int mb, int nb, int typenum, int itemsize,
                       bint writeable, object owner)


cdef class TiledMatrix:
    cdef parsec_tiled_matrix_t *_desc     # allocated with calloc(); NULL once released
    cdef void *_storage                   # the local tiles, owned by this object
    cdef parsec_data_free_t _free_storage # the deallocator matching their allocator
    cdef Context _context
    cdef bytes _name
    cdef readonly object dtype
    cdef int _typenum
    cdef int _itemsize
    cdef readonly uint64_t uid
    cdef bint _dtd_registered
    # DTD insertion cache: the taskpool (uid) that last recorded this matrix
    # as used, and its flush epoch at that time (see dtd.Taskpool).
    cdef uint64_t _dtd_used_by
    cdef uint64_t _dtd_used_epoch
    cdef object __weakref__

    cdef parsec_tiled_matrix_t *ptr(self) except NULL
    cdef parsec_data_collection_t *dc(self) except NULL
    cdef int check_tile(self, int m, int n) except -1
    cdef dtd_register(self)
    cdef dtd_unregister(self)
    cdef detach(self)
    cdef free_storage(self)


cdef class BlockCyclic(TiledMatrix):
    cdef parsec_matrix_block_cyclic_t *_bc
