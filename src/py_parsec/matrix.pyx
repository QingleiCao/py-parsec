# Copyright (c) 2026      The University of Tennessee and The University
#                         of Tennessee Research Foundation.  All rights
#                         reserved.
"""Tiled matrices (``parsec_tiled_matrix_t``) and the library operations on them.

The data collections defined here are not tied to a DSL: the same matrix can
be used by DTD tasks (:mod:`py_parsec.dtd`) and by PTG taskpools such as
:func:`redistribute_new`. Tiles are stored in tile (column-major per tile)
layout; :meth:`TiledMatrix.tile` exposes a local tile as a writable
``(mb, nb)`` Fortran-ordered NumPy array that shares the PaRSEC storage.
"""

cimport cython
from libc.stdint cimport uint64_t, uintptr_t
from libc.stdlib cimport calloc, free
from libc.string cimport memset

import numpy as np
cimport numpy as cnp

from py_parsec cimport cparsec as c
from py_parsec.runtime cimport Context, Taskpool, make_collection_release

cnp.import_array()

_MATRIX_TYPES = {
    np.dtype(np.int8): c.PARSEC_MATRIX_BYTE,
    np.dtype(np.uint8): c.PARSEC_MATRIX_BYTE,
    np.dtype(np.int32): c.PARSEC_MATRIX_INTEGER,
    np.dtype(np.float32): c.PARSEC_MATRIX_FLOAT,
    np.dtype(np.float64): c.PARSEC_MATRIX_DOUBLE,
    np.dtype(np.complex64): c.PARSEC_MATRIX_COMPLEX_FLOAT,
    np.dtype(np.complex128): c.PARSEC_MATRIX_COMPLEX_DOUBLE,
}

#: The NumPy dtypes a tiled matrix can hold.
DTYPES = tuple(_MATRIX_TYPES)


def matrix_type(dtype):
    """The ``parsec_matrix_type_t`` matching a NumPy dtype."""
    dt = np.dtype(dtype)
    try:
        return _MATRIX_TYPES[dt]
    except KeyError:
        raise TypeError(f"no PaRSEC matrix type for dtype {dt}; supported: "
                        f"{', '.join(str(d) for d in DTYPES)}") from None


cdef object host_array(void *ptr, int mb, int nb, int typenum, int itemsize,
                       bint writeable, object owner):
    """An (mb, nb) column-major array over ptr, made in one allocation; it
    keeps owner alive."""
    cdef cnp.npy_intp dims[2]
    cdef cnp.npy_intp strides[2]
    cdef int flags = cnp.NPY_ARRAY_F_CONTIGUOUS | cnp.NPY_ARRAY_ALIGNED
    if writeable:
        flags |= cnp.NPY_ARRAY_WRITEABLE
    dims[0] = mb
    dims[1] = nb
    strides[0] = itemsize
    strides[1] = <cnp.npy_intp>itemsize * mb
    arr = cnp.PyArray_New(cnp.ndarray, 2, dims, typenum, strides, ptr, itemsize, flags, None)
    if owner is not None:
        cnp.set_array_base(arr, owner)
    return arr


# Identifies a matrix for the DTD insertion caches; never reused.
cdef uint64_t _next_uid = 1


def _default_grid(int nodes):
    """Near-square P x Q process grid with P <= Q."""
    cdef int p = <int>(nodes ** 0.5)
    while p > 1 and nodes % p:
        p -= 1
    return max(p, 1)


@cython.no_gc_clear   # keeps _context until the matrix is released (see runtime.Context)
cdef class TiledMatrix:
    """Base class of the tiled matrices; see :class:`BlockCyclic`."""

    def __cinit__(self, *args, **kwargs):
        global _next_uid
        self.uid = _next_uid
        _next_uid += 1
        self._desc = NULL
        self._storage = NULL
        self._free_storage = NULL
        self._context = None
        self._dtd_registered = False
        self._dtd_used_by = 0
        self._dtd_used_epoch = 0

    def __dealloc__(self):
        # Taskpools keep the matrices their tasks use alive, so nothing refers
        # to the data any more. Nothing here calls PaRSEC: the release is queued
        # on the context. Once the context is finalized, fini() has already
        # released the PaRSEC side, and only the storage is left to free.
        if self._desc != NULL and self._context is not None and self._context._ctx != NULL:
            self._context.defer(make_collection_release(self._desc, self._dtd_registered,
                                                        self._storage, self._free_storage))
            self._storage = NULL
        self._desc = NULL
        self.free_storage()

    cdef parsec_tiled_matrix_t *ptr(self) except NULL:
        if self._desc == NULL:
            raise RuntimeError("the matrix has been released")
        return self._desc

    cdef parsec_data_collection_t *dc(self) except NULL:
        return &self.ptr().super

    cdef dtd_register(self):
        """Give the collection the DTD tile table; done on first DTD use."""
        if not self._dtd_registered:
            c.parsec_dtd_data_collection_init(self.dc())
            self._dtd_registered = True

    cdef dtd_unregister(self):
        if self._dtd_registered:
            c.parsec_dtd_data_collection_fini(self.dc())
            self._dtd_registered = False
        self._dtd_used_by = 0

    cdef detach(self):
        """Release the PaRSEC side of the matrix, keeping its storage."""
        if self._desc != NULL:
            c.pyparsec_tiled_matrix_free(self._desc, self._dtd_registered)
            self._desc = NULL
            self._dtd_registered = False
            self._dtd_used_by = 0

    cdef free_storage(self):
        if self._storage != NULL:
            self._free_storage(self._storage)
            self._storage = NULL

    def _detach(self):
        # Context.fini(): the tile arrays taken from a matrix that is still
        # referenced stay valid until the matrix object goes away.
        self.detach()

    def release(self):
        """Free the matrix and its storage; tile arrays obtained earlier
        become invalid. Release a matrix only once no taskpool that uses it
        has work left. Otherwise the matrix is freed when it is no longer
        referenced (its tile arrays refer to it)."""
        self.detach()
        self.free_storage()

    @property
    def released(self):
        """True once the storage has been freed."""
        return self._desc == NULL

    @property
    def address(self):
        """The ``parsec_tiled_matrix_t *`` as an int, for C libraries."""
        return <uintptr_t>self.ptr()

    @property
    def name(self):
        return self._name.decode()

    @property
    def m(self):
        """Number of rows."""
        return self.ptr().m

    @property
    def n(self):
        """Number of columns."""
        return self.ptr().n

    @property
    def shape(self):
        """``(m, n)``."""
        return (self.ptr().m, self.ptr().n)

    @property
    def mb(self):
        """Number of rows of a tile."""
        return self.ptr().mb

    @property
    def nb(self):
        """Number of columns of a tile."""
        return self.ptr().nb

    @property
    def mt(self):
        """Number of tile rows."""
        return self.ptr().mt

    @property
    def nt(self):
        """Number of tile columns."""
        return self.ptr().nt

    @property
    def nb_local_tiles(self):
        """Number of tiles stored by this process."""
        return self.ptr().nb_local_tiles

    @property
    def rank(self):
        """Rank of this process in the collection."""
        return self.ptr().super.myrank

    @property
    def nodes(self):
        """Number of processes the collection is distributed over."""
        return self.ptr().super.nodes

    cdef int check_tile(self, int m, int n) except -1:
        cdef parsec_tiled_matrix_t *desc = self.ptr()
        if not (0 <= m < desc.mt and 0 <= n < desc.nt):
            raise IndexError(f"tile ({m}, {n}) outside the {desc.mt}x{desc.nt} tiles of {self.name}")
        return 0

    def rank_of(self, int m, int n):
        """Rank owning tile (m, n)."""
        self.check_tile(m, n)
        return c.pyparsec_rank_of(self.dc(), m, n)

    def is_local(self, int m, int n):
        """True if tile (m, n) is stored by this process."""
        return self.rank_of(m, n) == self.rank

    def tile_shape(self, int m, int n):
        """Number of rows and columns of tile (m, n) inside the matrix: the
        tiles of the last row and column may be partly outside of it."""
        cdef parsec_tiled_matrix_t *desc = self.ptr()
        self.check_tile(m, n)
        return (min(desc.mb, desc.m - m * desc.mb), min(desc.nb, desc.n - n * desc.nb))

    def tile(self, int m, int n):
        """Local tile (m, n) as an (mb, nb) Fortran-ordered array (no copy).

        The array refers to the host copy of the tile: read it once the
        taskpools writing the tile have completed (after a wait).
        """
        cdef void *ptr
        self.check_tile(m, n)
        ptr = c.pyparsec_local_tile(self.dc(), m, n)
        if ptr == NULL:
            raise ValueError(f"tile ({m}, {n}) of {self.name} is owned by rank "
                             f"{self.rank_of(m, n)}, not by rank {self.rank}")
        return host_array(ptr, self.ptr().mb, self.ptr().nb, self._typenum, self._itemsize, True, self)

    def local_tiles(self):
        """Iterate over (m, n, tile array) for the tiles stored by this process."""
        cdef int m, n
        cdef parsec_tiled_matrix_t *desc = self.ptr()
        for m in range(desc.mt):
            for n in range(desc.nt):
                if c.pyparsec_rank_of(&desc.super, m, n) == <int>desc.super.myrank:
                    yield m, n, self.tile(m, n)

    def __repr__(self):
        if self._desc == NULL:
            return f"<py_parsec.{type(self).__name__} released>"
        return (f"<py_parsec.{type(self).__name__} {self.name!r} {self.m}x{self.n} "
                f"tiles {self.mb}x{self.nb} {self.dtype}>")


cdef class BlockCyclic(TiledMatrix):
    """Two-dimensional block-cyclic tiled matrix (``parsec_matrix_block_cyclic_t``).

    Parameters
    ----------
    context : Context
    m, n : int
        Matrix size.
    mb, nb : int
        Tile size.
    dtype : numpy dtype
        float32, float64, complex64, complex128, int32, int8 or uint8.
    P : int, optional
        Rows of the process grid; Q is ``context.nb_nodes // P``. The
        default is a near-square grid.
    kp, kq, ip, jq : int
        k-cyclicity and grid origin, as in ``parsec_matrix_block_cyclic_init``.
    name : str
        Key of the data collection (shown in traces).

    The local tiles are allocated and zeroed.
    """

    def __cinit__(self, Context context not None, int m, int n, int mb, int nb,
                  dtype=np.float64, P=None, int kp=1, int kq=1, int ip=0, int jq=0,
                  name="matrix"):
        cdef int mtype, p, q, nodes = context.nb_nodes
        cdef size_t nbytes

        self._bc = NULL
        context.ptr()
        self._context = context
        if m < 0 or n < 0 or mb <= 0 or nb <= 0:
            raise ValueError("matrix and tile sizes must be positive")
        p = _default_grid(nodes) if P is None else P
        if p <= 0 or nodes % p:
            raise ValueError(f"P={p} does not divide the {nodes} processes")
        q = nodes // p
        if kp <= 0 or kq <= 0 or not (0 <= ip < p) or not (0 <= jq < q):
            raise ValueError("kp, kq must be positive and (ip, jq) inside the process grid")
        self.dtype = np.dtype(dtype)
        self._typenum = self.dtype.num
        self._itemsize = self.dtype.itemsize
        mtype = matrix_type(self.dtype)
        self._name = str(name).encode()

        self._bc = <c.parsec_matrix_block_cyclic_t *>calloc(1, sizeof(c.parsec_matrix_block_cyclic_t))
        if self._bc == NULL:
            raise MemoryError()
        c.parsec_matrix_block_cyclic_init(self._bc, <c.parsec_matrix_type_t>mtype, c.PARSEC_MATRIX_TILE,
                                          context.rank, mb, nb, m, n, 0, 0, m, n,
                                          p, q, kp, kq, ip, jq)
        self._desc = &self._bc.super    # owns _bc (the descriptor starts the structure)
        c.parsec_data_collection_set_key(&self._desc.super, self._name)

        nbytes = <size_t>self._desc.nb_local_tiles * self._desc.bsiz * c.pyparsec_sizeof_matrix_type(mtype)
        if nbytes > 0:
            # PaRSEC's allocator (devices may make it allocate pinned memory),
            # and the matching deallocator, kept for when the storage outlives
            # the runtime.
            self._free_storage = c.parsec_data_free
            self._storage = c.parsec_data_allocate(nbytes)
            if self._storage == NULL:
                self.release()
                raise MemoryError(f"cannot allocate {nbytes} bytes for {name}")
            memset(self._storage, 0, nbytes)
            self._bc.mat = self._storage
        context.register_collection(self)


cdef _check_redistribute(TiledMatrix source, TiledMatrix target):
    # redistribute is compiled for double precision only (DTYPE in
    # parsec/data_dist/matrix/redistribute/redistribute_internal.h).
    if source.dtype != np.float64 or target.dtype != np.float64:
        raise TypeError(f"redistribute supports float64 matrices only, not {source.dtype} -> {target.dtype}")


def redistribute_new(TiledMatrix source not None, TiledMatrix target not None,
                     size_row=None, size_col=None,
                     int disi_source=0, int disj_source=0,
                     int disi_target=0, int disj_target=0):
    """PTG taskpool copying a size_row x size_col submatrix of source into target.

    The submatrix starts at (disi_source, disj_source) in source and lands at
    (disi_target, disj_target) in target; the two matrices may use any
    distribution. Both must be float64. Add the returned taskpool to the
    context and wait for it like any other taskpool. The size defaults to
    the rest of the source matrix.
    """
    cdef c.parsec_taskpool_t *tp
    cdef int rows = source.m - disi_source if size_row is None else size_row
    cdef int cols = source.n - disj_source if size_col is None else size_col

    _check_redistribute(source, target)
    tp = c.parsec_redistribute_New(source.ptr(), target.ptr(), rows, cols,
                                   disi_source, disj_source, disi_target, disj_target)
    if tp == NULL:
        raise ValueError("parsec_redistribute_New rejected the arguments")
    return Taskpool.from_address(<uintptr_t>tp, keepalive=(source, target))


def redistribute_dtd(Context context not None, TiledMatrix source not None, TiledMatrix target not None,
                     size_row=None, size_col=None,
                     int disi_source=0, int disj_source=0,
                     int disi_target=0, int disj_target=0):
    """Blocking DTD version of :func:`redistribute_new`.

    It first completes the work of every taskpool of the context, as
    :meth:`Context.wait` does, then copies the submatrix; on return the
    context is idle.
    """
    cdef int rc
    cdef int rows = source.m - disi_source if size_row is None else size_row
    cdef int cols = source.n - disj_source if size_col is None else size_col
    cdef c.parsec_context_t *ctx = context.ptr()
    cdef c.parsec_tiled_matrix_t *src = source.ptr()
    cdef c.parsec_tiled_matrix_t *dst = target.ptr()

    _check_redistribute(source, target)
    # parsec_redistribute_dtd() waits for the whole context, so the other
    # taskpools must be ready for that (their collections flushed), and it
    # sets up and tears down the DTD tile tables of both collections itself.
    context.wait()
    source.dtd_unregister()
    target.dtd_unregister()
    context._in_runtime += 1
    with nogil:
        rc = c.parsec_redistribute_dtd(ctx, src, dst, rows, cols,
                                       disi_source, disj_source, disi_target, disj_target)
    context._in_runtime -= 1
    if rc != c.PARSEC_SUCCESS:
        raise RuntimeError(f"parsec_redistribute_dtd failed ({rc})")
