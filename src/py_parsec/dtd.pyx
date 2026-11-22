# Copyright (c) 2026      The University of Tennessee and The University
#                         of Tennessee Research Foundation.  All rights
#                         reserved.
"""Dynamic Task Discovery (DTD).

A DTD taskpool is an ordinary taskpool: it is added to the
:class:`py_parsec.Context` with ``add_taskpool`` like any other. Task classes
mirror ``parsec_dtd_create_task_class``: each parameter is either a data tile
(:class:`Data`) or a value (:class:`Value`). Chores are Python callables
(CPU only) or C function pointers (any device). A Python body receives each
tile as an ``(mb, nb)`` column-major NumPy array that aliases the PaRSEC data
copy (read-only for ``INPUT`` tiles), and each value as a Python number, in
declaration order::

    import py_parsec
    from py_parsec import dtd

    with py_parsec.Context() as ctx:
        A = py_parsec.BlockCyclic(ctx, 1000, 1000, 100, 100)
        tp = dtd.Taskpool()
        ctx.add_taskpool(tp)

        def body(tile, alpha):
            tile *= alpha

        scale = tp.task_class("scale", dtd.Data(dtd.INOUT | dtd.AFFINITY, A), dtd.Value("d"))
        scale.add_chore(py_parsec.DEV_CPU, body)
        for m in range(A.mt):
            for n in range(A.nt):
                tp.insert_task(scale, (A, m, n), 2.0)
        tp.wait()   # flushes A, then completes the tasks

A tile argument is ``(matrix, m, n)``, ``(matrix, m, n, flags)`` or
:func:`tile_of`. Inserting a task starts the compute threads if they are
idle.

Python bodies hold the GIL while they run, so CPU parallelism comes from
the code they call releasing it (NumPy, SciPy, ...). Keep the BLAS under
NumPy to one thread (``OPENBLAS_NUM_THREADS=1``) and give each body enough
work (see the README). The arrays handed to a body are only valid during the
call. A body may return ``py_parsec.HOOK_RETURN_AGAIN`` to be rescheduled
later, or ``HOOK_RETURN_DONE``; any other integer is an error, and any other
result (including None) means the task is done. A task whose body raises is
completed, and the exception is raised by the next wait on the taskpool or
the context.
"""

import ctypes
import weakref

cimport cython
from cpython.object cimport PyObject
from cpython.ref cimport Py_INCREF
from cpython.tuple cimport PyTuple_New, PyTuple_SET_ITEM
from libc.stdint cimport int8_t, uint8_t, int16_t, uint16_t, int64_t, uint64_t, uintptr_t
from libc.stdlib cimport calloc, free

import numpy as np

from py_parsec cimport cparsec as c
from py_parsec.matrix cimport TiledMatrix, host_array
from py_parsec.runtime cimport Context, make_release
from py_parsec.runtime cimport Taskpool as BaseTaskpool
from py_parsec.runtime import address_of
from py_parsec.matrix import matrix_type

INPUT = c.PARSEC_INPUT
OUTPUT = c.PARSEC_OUTPUT
INOUT = c.PARSEC_INOUT
AFFINITY = c.PARSEC_AFFINITY
DONT_TRACK = c.PARSEC_DONT_TRACK
PUSHOUT = c.PARSEC_PUSHOUT
PULLIN = c.PARSEC_PULLIN

#: Largest number of parameters of a task class.
MAX_PARAMS = c.PYPARSEC_MAX_PARAMS
#: Largest number of Data parameters (flows) of a task class.
MAX_FLOWS = c.pyparsec_dtd_max_flows()
#: Address of a C chore that does nothing (for tasks that only move data,
#: tests and benchmarks).
NOOP_CHORE = <uintptr_t><void *>c.pyparsec_noop_chore

cdef int _ACCESS_MODES = c.PARSEC_INPUT | c.PARSEC_OUTPUT | c.PARSEC_INOUT
cdef int _DATA_FLAGS = c.PARSEC_AFFINITY | c.PARSEC_DONT_TRACK | c.PARSEC_PUSHOUT | c.PARSEC_PULLIN

# Parameter kinds.
cdef enum:
    _DATA = 1
    _VALUE = 2

# Value codes.
cdef enum:
    _V_INT8, _V_UINT8, _V_INT16, _V_UINT16, _V_INT, _V_UINT, _V_LONG, _V_ULONG
    _V_INT64, _V_UINT64, _V_FLOAT, _V_DOUBLE

_VALUE_CODES = {"b": (_V_INT8, sizeof(int8_t)), "B": (_V_UINT8, sizeof(uint8_t)),
                "h": (_V_INT16, sizeof(int16_t)), "H": (_V_UINT16, sizeof(uint16_t)),
                "i": (_V_INT, sizeof(int)), "I": (_V_UINT, sizeof(unsigned int)),
                "l": (_V_LONG, sizeof(long)), "L": (_V_ULONG, sizeof(unsigned long)),
                "q": (_V_INT64, sizeof(int64_t)), "Q": (_V_UINT64, sizeof(uint64_t)),
                "f": (_V_FLOAT, sizeof(float)), "d": (_V_DOUBLE, sizeof(double))}

# Number of tasks converted under the GIL before insert_tasks() inserts them.
cdef int _BATCH = 256


cdef class Param:
    """A parameter of a task class (see :class:`Data` and :class:`Value`)."""

    cdef int kind


@cython.final
cdef class Data(Param):
    """A tile parameter.

    Parameters
    ----------
    op : int
        ``INPUT``, ``OUTPUT`` or ``INOUT``, possibly or-ed with ``AFFINITY``
        (run the task where this tile lives), ``DONT_TRACK``, ``PUSHOUT`` or
        ``PULLIN``.
    like : TiledMatrix, optional
        A matrix whose tiles go through this parameter; it provides the tile
        shape and dtype. Otherwise give ``shape=(mb, nb)`` and ``dtype``.
    """

    cdef readonly int op
    cdef readonly int mb
    cdef readonly int nb
    cdef readonly object dtype
    cdef int mtype
    cdef int typenum
    cdef int itemsize
    cdef bint writeable
    cdef uint64_t checked_uid   # last matrix found to have matching tiles

    def __init__(self, int op, TiledMatrix like=None, *, shape=None, dtype=None):
        if (op & c.PARSEC_GET_OP_TYPE) not in (c.PARSEC_INPUT, c.PARSEC_OUTPUT, c.PARSEC_INOUT):
            raise ValueError("a Data parameter needs INPUT, OUTPUT or INOUT")
        if op & ~(_ACCESS_MODES | _DATA_FLAGS):
            raise ValueError(f"unsupported flags {op:#x} for a Data parameter")
        if like is not None:
            if shape is not None or dtype is not None:
                raise TypeError("give either like= or shape= and dtype=")
            self.mb, self.nb, self.dtype = like.mb, like.nb, like.dtype
        else:
            if shape is None or dtype is None:
                raise TypeError("a Data parameter needs like= or shape= and dtype=")
            self.mb, self.nb = shape
            self.dtype = np.dtype(dtype)
        if self.mb <= 0 or self.nb <= 0:
            raise ValueError("the tile shape must be positive")
        self.kind = _DATA
        self.op = op
        self.mtype = matrix_type(self.dtype)
        self.typenum = self.dtype.num
        self.itemsize = self.dtype.itemsize
        self.writeable = (op & c.PARSEC_GET_OP_TYPE) != c.PARSEC_INPUT
        self.checked_uid = 0

    cdef int check_matrix(self, TiledMatrix matrix, int i, str tcname) except -1:
        if matrix._desc.mb != self.mb or matrix._desc.nb != self.nb or matrix._typenum != self.typenum:
            raise ValueError(f"argument {i} of {tcname}: {matrix._desc.mb}x{matrix._desc.nb} "
                             f"{matrix.dtype} tiles, the task class expects "
                             f"{self.mb}x{self.nb} {self.dtype}")
        self.checked_uid = matrix.uid
        return 0

    def __repr__(self):
        return f"Data({self.op:#x}, shape=({self.mb}, {self.nb}), dtype={self.dtype})"


@cython.final
cdef class Value(Param):
    """A parameter passed by value, given as a :mod:`struct` format character.

    One of ``b B h H i I l L q Q f d`` (native C char ... double); the
    default ``"i"`` is a C ``int``. Values are copied at insertion.
    """

    cdef readonly str format
    cdef int nbytes
    cdef int code

    def __init__(self, str format="i"):
        if format not in _VALUE_CODES:
            raise ValueError(f"unsupported Value format {format!r}")
        self.kind = _VALUE
        self.format = format
        self.code, self.nbytes = _VALUE_CODES[format]

    cdef object load(self, uint64_t *slot):
        cdef void *p = <void *>slot
        cdef int code = self.code
        if code == _V_INT8: return (<int8_t *>p)[0]
        elif code == _V_UINT8: return (<uint8_t *>p)[0]
        elif code == _V_INT16: return (<int16_t *>p)[0]
        elif code == _V_UINT16: return (<uint16_t *>p)[0]
        elif code == _V_INT: return (<int *>p)[0]
        elif code == _V_UINT: return (<unsigned int *>p)[0]
        elif code == _V_LONG: return (<long *>p)[0]
        elif code == _V_ULONG: return (<unsigned long *>p)[0]
        elif code == _V_INT64: return (<int64_t *>p)[0]
        elif code == _V_UINT64: return (<uint64_t *>p)[0]
        elif code == _V_FLOAT: return (<float *>p)[0]
        return (<double *>p)[0]

    cdef int store(self, object value, uint64_t *slot) except -1:
        cdef void *p = <void *>slot
        cdef int code = self.code
        slot[0] = 0
        if code == _V_INT8: (<int8_t *>p)[0] = value
        elif code == _V_UINT8: (<uint8_t *>p)[0] = value
        elif code == _V_INT16: (<int16_t *>p)[0] = value
        elif code == _V_UINT16: (<uint16_t *>p)[0] = value
        elif code == _V_INT: (<int *>p)[0] = value
        elif code == _V_UINT: (<unsigned int *>p)[0] = value
        elif code == _V_LONG: (<long *>p)[0] = value
        elif code == _V_ULONG: (<unsigned long *>p)[0] = value
        elif code == _V_INT64: (<int64_t *>p)[0] = value
        elif code == _V_UINT64: (<uint64_t *>p)[0] = value
        elif code == _V_FLOAT: (<float *>p)[0] = value
        else: (<double *>p)[0] = value
        return 0

    def __repr__(self):
        return f"Value({self.format!r})"


@cython.final
cdef class Tile:
    """Tile (m, n) of a data collection, as passed to :meth:`Taskpool.insert_task`.

    ``flags`` are added to the access mode declared by the task class for
    this insertion only (for instance ``PUSHOUT`` on the last update). A
    tuple ``(matrix, m, n)`` or ``(matrix, m, n, flags)`` is equivalent and
    cheaper to build.
    """

    cdef readonly TiledMatrix matrix
    cdef readonly int m
    cdef readonly int n
    cdef readonly int flags

    def __repr__(self):
        return f"tile_of({self.matrix.name}, {self.m}, {self.n}, flags={self.flags:#x})"


def tile_of(TiledMatrix matrix not None, int m, int n, int flags=0):
    """Reference to tile (m, n) of ``matrix`` for a DTD insertion
    (the ``PARSEC_DTD_TILE_OF`` of the C interface)."""
    cdef Tile t
    if flags & ~_DATA_FLAGS:
        raise ValueError(f"unsupported insertion flags {flags:#x}")
    matrix.check_tile(m, n)
    t = Tile.__new__(Tile)
    t.matrix = matrix
    t.m = m
    t.n = n
    t.flags = flags
    return t


cdef TiledMatrix _tile_arg(object arg, int i, TaskClass tc, int *pm, int *pn, int *pflags):
    """The matrix and the (m, n, flags) of a tile argument."""
    cdef Tile t
    cdef tuple tup
    cdef Py_ssize_t size
    if type(arg) is tuple:
        tup = <tuple>arg
        size = len(tup)
        if (size == 3 or size == 4) and isinstance(tup[0], TiledMatrix):
            pm[0] = tup[1]
            pn[0] = tup[2]
            pflags[0] = tup[3] if size == 4 else 0
            if pflags[0] & ~_DATA_FLAGS:
                raise ValueError(f"argument {i} of {tc.name}: unsupported insertion flags {pflags[0]:#x}")
            return <TiledMatrix>tup[0]
    elif type(arg) is Tile:
        t = <Tile>arg
        pm[0] = t.m
        pn[0] = t.n
        pflags[0] = t.flags
        return t.matrix
    raise TypeError(f"argument {i} of {tc.name} must be a tile: (matrix, m, n), "
                    f"(matrix, m, n, flags) or tile_of(...), not {type(arg).__name__}")


# ---------------------------------------------------------------------------
# Python chores.
#
# Every task class with a Python body uses the same C chore, which finds the
# TaskClass object from the task's parsec_task_class_t through this table.
# The table holds borrowed references: a task class removes itself before it
# goes away, and its tasks cannot run after that (the context keeps every
# taskpool with work pending, and a taskpool keeps its task classes). It is
# only read and written with the GIL held. Open addressing, rebuilt on removal.
# ---------------------------------------------------------------------------

cdef struct _Slot:
    const void *key
    PyObject *value

cdef _Slot *_table = NULL
cdef size_t _table_size = 0      # a power of two, or 0
cdef size_t _table_used = 0


cdef inline size_t _slot_index(const void *key, size_t mask) noexcept nogil:
    cdef uint64_t k = <uintptr_t>key
    return <size_t>((k * <uint64_t>0x9E3779B97F4A7C15ULL) >> 32) & mask


cdef PyObject *_table_get(const void *key) noexcept nogil:
    cdef size_t i, mask
    if _table == NULL:
        return NULL
    mask = _table_size - 1
    i = _slot_index(key, mask)
    while _table[i].key != NULL:
        if _table[i].key == key:
            return _table[i].value
        i = (i + 1) & mask
    return NULL


cdef int _table_rebuild(size_t size, const void *skip) except -1:
    global _table, _table_size, _table_used
    cdef _Slot *old = _table
    cdef size_t old_size = _table_size, j, i, mask
    cdef _Slot *new = <_Slot *>calloc(size, sizeof(_Slot))
    if new == NULL:
        raise MemoryError()
    mask = size - 1
    _table_used = 0
    for j in range(old_size):
        if old[j].key != NULL and old[j].key != skip:
            i = _slot_index(old[j].key, mask)
            while new[i].key != NULL:
                i = (i + 1) & mask
            new[i] = old[j]
            _table_used += 1
    _table = new
    _table_size = size
    free(old)
    return 0


cdef int _table_put(const void *key, PyObject *value) except -1:
    global _table_used
    cdef size_t i, mask
    if 2 * (_table_used + 1) > _table_size:
        _table_rebuild(max(16, 2 * _table_size), NULL)
    mask = _table_size - 1
    i = _slot_index(key, mask)
    while _table[i].key != NULL and _table[i].key != key:
        i = (i + 1) & mask
    if _table[i].key == NULL:
        _table_used += 1
    _table[i].key = key
    _table[i].value = value
    return 0


cdef void _table_remove(const void *key) noexcept:
    if _table_get(key) != NULL:
        try:
            _table_rebuild(_table_size, key)
        except MemoryError:
            pass   # keep the stale entry: its key is never looked up again


cdef c.parsec_hook_return_t _python_chore(c.parsec_execution_stream_t *es,
                                          c.parsec_task_t *task) noexcept nogil:
    c.pyparsec_keep_thread_state()
    with gil:
        return _run_python_body(task)


cdef c.parsec_hook_return_t _run_python_body(c.parsec_task_t *task) noexcept:
    cdef PyObject *tc = _table_get(c.pyparsec_task_class_of(task))
    if tc == NULL:
        return c.PARSEC_HOOK_RETURN_ERROR
    return (<TaskClass>tc).execute(task)


# Exceptions kept per task class until a wait raises them.
cdef int _MAX_KEPT_ERRORS = 16


@cython.final
cdef class TaskClass:
    """A DTD task class; create it with :meth:`Taskpool.task_class`."""

    cdef c.parsec_task_class_t *_tc
    cdef uint64_t _owner_uid    # the taskpool's uid
    cdef object _taskpool_ref   # weak reference: a task class never keeps its taskpool alive
    cdef bytes _name
    cdef readonly tuple params
    cdef int nparams
    cdef int _kind[c.PYPARSEC_MAX_PARAMS]
    cdef bint _placed           # some Data parameter has AFFINITY, or none writes tracked data
    cdef int _devices
    cdef object _body
    cdef bint _in_table
    cdef list _keepalive
    cdef list _errors
    cdef Py_ssize_t _lost_errors

    def __dealloc__(self):
        self.detach()

    @property
    def name(self):
        return self._name.decode()

    @property
    def devices(self):
        """Mask of the device types that have a chore."""
        return self._devices

    cdef Taskpool taskpool(self):
        """The taskpool of the task class, or None once it is released."""
        if self._taskpool_ref is None:
            return None
        return self._taskpool_ref()

    cdef c.parsec_task_class_t *create(self, Taskpool tp) except NULL:
        """Create the C task class (deferred until its first chore)."""
        cdef int sizes[c.PYPARSEC_MAX_PARAMS]
        cdef int ops[c.PYPARSEC_MAX_PARAMS]
        cdef int i
        cdef Data d
        cdef Context ctx = tp._context
        for i in range(self.nparams):
            if self._kind[i] == _DATA:
                d = <Data>self.params[i]
                sizes[i] = c.PASSED_BY_REF
                ops[i] = d.op | ctx.dtd_tile_arena(d.mtype, d.mb, d.nb, d.mb)
            else:
                sizes[i] = (<Value>self.params[i]).nbytes
                ops[i] = c.PARSEC_VALUE
        self._tc = c.pyparsec_dtd_create_task_class(self._name, self.nparams, sizes, ops)
        if self._tc == NULL:
            raise RuntimeError(f"cannot create task class {self.name}")
        return self._tc

    def add_chore(self, int device_type, body):
        """Add the body run on ``device_type`` (``parsec_dtd_task_class_add_chore``).

        ``body`` is either a Python callable, for ``DEV_CPU`` only, or the
        address of a C function (int, unnamed PyCapsule or ctypes function
        pointer) with the DTD chore signature of that device type.
        """
        cdef void *fn
        cdef int rc
        cdef bint python_body = False
        cdef Taskpool tp = self.taskpool()
        if tp is None or tp._tp == NULL:
            raise RuntimeError("the taskpool of this task class has been released")
        if (device_type <= 0 or device_type & ~c.PARSEC_DEV_ALL
                or device_type & (device_type - 1)):
            raise ValueError(f"a chore runs on exactly one device type, not {device_type:#x}")
        if device_type & self._devices:
            raise ValueError(f"task class {self.name} already has a chore for device {device_type:#x}")
        if tp._context.nb_devices(device_type) == 0:
            raise ValueError(f"the context has no device of type {device_type:#x}")

        if isinstance(body, ctypes._CFuncPtr):
            fn = <void *><uintptr_t>ctypes.cast(body, ctypes.c_void_p).value
            self._keepalive.append(body)
        elif callable(body):
            if device_type != c.PARSEC_DEV_CPU:
                raise TypeError("a Python body runs on DEV_CPU; accelerator chores take a C function address")
            fn = <void *>_python_chore
            python_body = True
        else:
            fn = <void *><uintptr_t>address_of(body)
        if fn == NULL:
            raise ValueError("NULL chore")

        if self._tc == NULL:
            self.create(tp)
        if python_body:
            self._body = body
            _table_put(self._tc, <PyObject *>self)
            self._in_table = True
        rc = c.parsec_dtd_task_class_add_chore(tp._tp, self._tc, device_type, fn)
        if rc != c.PARSEC_SUCCESS:
            if python_body:
                self._body = None
                _table_remove(self._tc)
                self._in_table = False
            if self._devices == 0:
                # Never registered in the taskpool: DTD cannot release it.
                self._tc = NULL
            raise RuntimeError(f"cannot add a chore for device {device_type:#x} to {self.name} ({rc})")
        self._devices |= device_type

    cdef c.parsec_hook_return_t execute(self, c.parsec_task_t *task) noexcept:
        """Run the Python body of a task; called with the GIL held."""
        cdef uint64_t values[c.PYPARSEC_MAX_PARAMS]     # value parameters
        cdef void *tiles[c.PYPARSEC_MAX_PARAMS]         # data parameters
        cdef void *outs[c.PYPARSEC_MAX_PARAMS]
        cdef int i, n = self.nparams
        cdef Data d
        for i in range(n):
            if self._kind[i] == _DATA:
                tiles[i] = NULL
                outs[i] = &tiles[i]
            else:
                values[i] = 0
                outs[i] = &values[i]
        c.pyparsec_dtd_unpack(task, n, outs)
        try:
            args = PyTuple_New(n)
            for i in range(n):
                if self._kind[i] == _DATA:
                    d = <Data>self.params[i]
                    arg = host_array(tiles[i], d.mb, d.nb, d.typenum, d.itemsize, d.writeable, None)
                else:
                    arg = (<Value>self.params[i]).load(&values[i])
                Py_INCREF(arg)
                PyTuple_SET_ITEM(args, i, arg)
            rc = self._body(*args)
            if rc is None or type(rc) is not int:
                return c.PARSEC_HOOK_RETURN_DONE
            if rc == c.PARSEC_HOOK_RETURN_DONE or rc == c.PARSEC_HOOK_RETURN_AGAIN:
                return <c.parsec_hook_return_t><int>rc
            raise ValueError(f"the body of {self.name} returned {rc}: only HOOK_RETURN_DONE "
                             "and HOOK_RETURN_AGAIN are accepted")
        except BaseException as e:
            if self._errors is None:
                self._errors = []
            if len(self._errors) < _MAX_KEPT_ERRORS:
                self._errors.append(e)
            else:
                self._lost_errors += 1
            return c.PARSEC_HOOK_RETURN_DONE

    cdef take_errors(self, list into):
        if self._errors:
            into.extend(self._errors)
            self._errors = None
        if self._lost_errors:
            into.append(RuntimeError(f"{self._lost_errors} more task bodies of {self.name} raised"))
            self._lost_errors = 0

    cdef void detach(self) noexcept:
        """Forget the C task class (its taskpool releases it)."""
        if self._in_table:
            _table_remove(self._tc)
            self._in_table = False
        self._tc = NULL
        self._taskpool_ref = None
        self._body = None
        self._keepalive = None

    cdef release(self, c.parsec_taskpool_t *tp):
        cdef c.parsec_task_class_t *ctc = self._tc
        self.detach()
        if ctc != NULL:
            c.parsec_dtd_task_class_release(tp, ctc)

    def __repr__(self):
        return f"<py_parsec.dtd.TaskClass {self.name!r} {list(self.params)}>"


@cython.no_gc_clear
cdef class Taskpool(BaseTaskpool):
    """A DTD taskpool (``parsec_dtd_taskpool_new``).

    Insert tasks once the taskpool has been added to the context; waiting on
    it (or on the context) completes the tasks inserted so far, and more can
    be inserted afterwards. DTD needs every collection used by the tasks to
    be flushed before a wait: :meth:`wait` and :meth:`Context.wait` first
    call :meth:`data_flush_all` on the collections used since their last
    flush, so that their data is back on its owners when the wait returns.
    """

    cdef list _classes
    cdef dict _collections   # TiledMatrix -> used since its last flush; also
                             # keeps them alive while tasks may refer to them
    cdef uint64_t _epoch     # incremented by every flush of all the collections

    def __cinit__(self):
        self._classes = []
        self._collections = {}
        self._epoch = 1
        self._tp = c.parsec_dtd_taskpool_new()
        if self._tp == NULL:
            raise RuntimeError("parsec_dtd_taskpool_new failed")

    def __dealloc__(self):
        # Runs before the base __dealloc__, while the attributes of this class
        # are still set. Nothing here may call into PaRSEC (this can run on
        # any thread that triggers a garbage collection): the release of the
        # task classes and of the taskpool is queued on the context.
        cdef TaskClass tc
        cdef list addresses = []
        if self._tp == NULL:
            return
        for tc in self._classes:
            if tc._tc != NULL:
                addresses.append(<uintptr_t>tc._tc)
            tc.detach()
        if self._context is not None and self._context._ctx != NULL:
            self._context.defer(make_release(self._tp, self._destruct, addresses,
                                             (self._collections, self._keepalive)))
        self._tp = NULL

    def task_class(self, name, *params):
        """Create a task class whose parameters are :class:`Data` and :class:`Value`.

        The taskpool must already have been added to the context.
        """
        cdef TaskClass tc
        cdef Param p
        cdef int i, flows = 0
        cdef bint affinity = False, writes = False
        self.enqueued_context()
        if len(params) > c.PYPARSEC_MAX_PARAMS:
            raise ValueError(f"a task class has at most {c.PYPARSEC_MAX_PARAMS} parameters")
        for p_obj in params:
            if not isinstance(p_obj, Param):
                raise TypeError(f"task class parameters are Data or Value, not {type(p_obj).__name__}")
        tc = TaskClass.__new__(TaskClass)
        for i, p_obj in enumerate(params):
            p = <Param>p_obj
            tc._kind[i] = p.kind
            if p.kind == _DATA:
                flows += 1
                affinity |= ((<Data>p).op & c.PARSEC_AFFINITY) != 0
                writes |= (((<Data>p).op & c.PARSEC_GET_OP_TYPE) != c.PARSEC_INPUT
                           and not ((<Data>p).op & c.PARSEC_DONT_TRACK))
        if flows > MAX_FLOWS:
            raise ValueError(f"a task class has at most {MAX_FLOWS} Data parameters")
        tc._owner_uid = self.uid
        tc._taskpool_ref = weakref.ref(self)
        tc._name = str(name).encode()
        tc.params = params
        tc.nparams = len(params)
        tc._placed = affinity or not writes
        tc._keepalive = []
        self._classes.append(tc)
        return tc

    cdef Context check_insert(self, TaskClass tc, Py_ssize_t nargs, int device):
        cdef Context ctx = self.enqueued_context()
        if tc._owner_uid != self.uid:
            raise ValueError(f"task class {tc.name} belongs to another taskpool")
        if not (device & tc._devices):
            raise ValueError(f"task class {tc.name} has no chore for device mask {device:#x}")
        if nargs != tc.nparams:
            raise TypeError(f"task class {tc.name} takes {tc.nparams} arguments, {nargs} given")
        return ctx

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef int convert(self, Context ctx, TaskClass tc, tuple args, int *flags, void **ptrs,
                     uint64_t *values, PyObject **matrices, c.parsec_data_key_t *keys) except -1:
        """Check and convert the arguments of one task. No DTD tile is
        created here: one left behind by a failed insertion could not be flushed."""
        cdef int i, m, n, tflags, op, write_flows = 0
        cdef bint affinity = tc._placed
        cdef Data d
        cdef TiledMatrix matrix
        cdef c.parsec_tiled_matrix_t *desc
        for i in range(tc.nparams):
            if tc._kind[i] == _DATA:
                d = <Data>tc.params[i]
                matrix = _tile_arg(args[i], i, tc, &m, &n, &tflags)
                desc = matrix._desc
                if desc == NULL:
                    raise RuntimeError(f"argument {i} of {tc.name}: the matrix has been released")
                if matrix.uid != d.checked_uid:
                    d.check_matrix(matrix, i, tc.name)
                if not (0 <= m < desc.mt and 0 <= n < desc.nt):
                    raise IndexError(f"argument {i} of {tc.name}: tile ({m}, {n}) outside the "
                                     f"{desc.mt}x{desc.nt} tiles of {matrix.name}")
                keys[i] = c.pyparsec_data_key(&desc.super, m, n)
                matrices[i] = <PyObject *>matrix
                flags[i] = tflags
                if not affinity:
                    op = d.op | tflags
                    affinity = (op & c.PARSEC_AFFINITY) != 0
                    if (op & c.PARSEC_GET_OP_TYPE) != c.PARSEC_INPUT and not (op & c.PARSEC_DONT_TRACK):
                        write_flows += 1
            else:
                (<Value>tc.params[i]).store(args[i], &values[i])
                ptrs[i] = &values[i]
                flags[i] = 0
                matrices[i] = NULL
        if not affinity and write_flows and ctx.nb_nodes > 1:
            raise ValueError(f"task class {tc.name} writes a tile but no parameter has AFFINITY: "
                             "a task that writes data must be placed in a distributed run")
        return 0

    cdef int bind_tiles(self, TaskClass tc, void **ptrs, PyObject **matrices,
                        c.parsec_data_key_t *keys) except -1:
        """Look up the DTD tiles of a converted task, recording the
        collections used since their last flush."""
        cdef int i
        cdef TiledMatrix matrix
        for i in range(tc.nparams):
            if matrices[i] != NULL:
                matrix = <TiledMatrix>matrices[i]
                if not matrix._dtd_registered:
                    matrix.dtd_register()
                if matrix._dtd_used_by != self.uid or matrix._dtd_used_epoch != self._epoch:
                    self._collections[matrix] = True
                    matrix._dtd_used_by = self.uid
                    matrix._dtd_used_epoch = self._epoch
                ptrs[i] = c.parsec_dtd_tile_of(&matrix._desc.super, keys[i])
        return 0

    def insert_task(self, TaskClass tc not None, *args, int priority=0, int device=c.PARSEC_DEV_ALL):
        """Insert a task of ``tc`` (``parsec_dtd_insert_task_with_task_class``).

        Data parameters take a tile, ``(matrix, m, n)``, ``(matrix, m, n,
        flags)`` or :func:`tile_of`; value parameters a Python number.
        ``device`` restricts the chores the task may run with.
        """
        cdef int flags[c.PYPARSEC_MAX_PARAMS]
        cdef void *ptrs[c.PYPARSEC_MAX_PARAMS]
        cdef uint64_t values[c.PYPARSEC_MAX_PARAMS]
        cdef PyObject *matrices[c.PYPARSEC_MAX_PARAMS]
        cdef c.parsec_data_key_t keys[c.PYPARSEC_MAX_PARAMS]
        cdef Context ctx = self.check_insert(tc, len(args), device)
        cdef c.parsec_taskpool_t *tp = self._tp
        cdef c.parsec_task_class_t *ctc = tc._tc
        cdef int nargs = tc.nparams

        self.convert(ctx, tc, args, flags, ptrs, values, matrices, keys)
        self.bind_tiles(tc, ptrs, matrices, keys)
        if not c.pyparsec_context_is_active(ctx._ctx):
            ctx.ensure_started()
        with nogil:
            c.pyparsec_dtd_insert(tp, ctc, priority, device, nargs, flags, ptrs)
        if not self._pending:
            self.mark_pending()

    def insert_tasks(self, TaskClass tc not None, tasks, *, int priority=0, int device=c.PARSEC_DEV_ALL):
        """Insert one task of ``tc`` per item of ``tasks``, each item being the
        argument tuple :meth:`insert_task` would take.

        Arguments are converted in batches, and each batch is inserted
        without taking the GIL back between tasks. This is faster than a loop
        of :meth:`insert_task` when Python bodies of other tasks compete for
        the GIL. Returns the number of tasks inserted.
        """
        cdef Context ctx = self.check_insert(tc, tc.nparams, device)
        cdef c.parsec_taskpool_t *tp = self._tp
        cdef c.parsec_task_class_t *ctc = tc._tc
        cdef int nargs = tc.nparams, stride = max(nargs, 1)
        cdef int *flags = NULL
        cdef void **ptrs = NULL
        cdef uint64_t *values = NULL
        cdef PyObject *matrices[c.PYPARSEC_MAX_PARAMS]
        cdef c.parsec_data_key_t keys[c.PYPARSEC_MAX_PARAMS]
        cdef Py_ssize_t count = 0, j, k, batched = 0

        flags = <int *>calloc(_BATCH * stride, sizeof(int))
        ptrs = <void **>calloc(_BATCH * stride, sizeof(void *))
        values = <uint64_t *>calloc(_BATCH * stride, sizeof(uint64_t))
        try:
            if flags == NULL or ptrs == NULL or values == NULL:
                raise MemoryError()
            it = iter(tasks)
            done = False
            while not done:
                batched = 0
                error = None
                try:
                    for item in it:
                        args = item if type(item) is tuple else tuple(item)
                        if len(<tuple>args) != nargs:
                            raise TypeError(f"task class {tc.name} takes {nargs} arguments, "
                                            f"{len(<tuple>args)} given (task {count + batched})")
                        j = batched * stride
                        self.convert(ctx, tc, <tuple>args, &flags[j], &ptrs[j], &values[j], matrices, keys)
                        self.bind_tiles(tc, &ptrs[j], matrices, keys)
                        batched += 1
                        if batched == _BATCH:
                            break
                    else:
                        done = True
                except BaseException as e:
                    # The tasks converted before the failing one are inserted:
                    # their DTD tiles are already looked up.
                    error = e
                    done = True
                if batched:
                    if not c.pyparsec_context_is_active(ctx._ctx):
                        ctx.ensure_started()
                    with nogil:
                        for k in range(batched):
                            c.pyparsec_dtd_insert(tp, ctc, priority, device, nargs,
                                                  &flags[k * stride], &ptrs[k * stride])
                    count += batched
                    if not self._pending:
                        self.mark_pending()
                if error is not None:
                    raise error
        finally:
            free(flags)
            free(ptrs)
            free(values)
        return count

    cdef c.parsec_dtd_tile_t *dtd_tile(self, TiledMatrix matrix, int m, int n) except NULL:
        matrix.check_tile(m, n)
        matrix.dtd_register()
        self._collections.setdefault(matrix, False)
        return c.parsec_dtd_tile_of(matrix.dc(), c.pyparsec_data_key(matrix.dc(), m, n))

    def data_flush(self, tile):
        """Flush one tile back to its owner (``parsec_dtd_data_flush``).
        ``tile`` is ``(matrix, m, n)`` or :func:`tile_of`."""
        cdef c.parsec_taskpool_t *tp = self.ptr()
        cdef Context ctx = self.enqueued_context()
        cdef c.parsec_dtd_tile_t *dtile
        cdef int m, n, flags
        cdef TiledMatrix matrix
        if type(tile) is Tile:
            matrix, m, n = (<Tile>tile).matrix, (<Tile>tile).m, (<Tile>tile).n
        elif type(tile) is tuple and len(tile) == 3 and isinstance(tile[0], TiledMatrix):
            matrix, m, n = tile
        else:
            raise TypeError("data_flush takes (matrix, m, n) or tile_of(...)")
        dtile = self.dtd_tile(matrix, m, n)
        ctx.ensure_started()
        with nogil:
            c.parsec_dtd_data_flush(tp, dtile)
        self.mark_pending()

    def data_flush_all(self, TiledMatrix matrix not None):
        """Flush every tile of ``matrix`` back to its owner
        (``parsec_dtd_data_flush_all``); :meth:`wait` does it for the
        collections that need it."""
        cdef c.parsec_taskpool_t *tp = self.ptr()
        cdef Context ctx = self.enqueued_context()
        cdef c.parsec_data_collection_t *dc
        matrix.dtd_register()
        dc = matrix.dc()
        ctx.ensure_started()
        with nogil:
            c.parsec_dtd_data_flush_all(tp, dc)
        self._collections[matrix] = False
        if matrix._dtd_used_by == self.uid:
            matrix._dtd_used_by = 0
        self.mark_pending()

    cdef list take_errors(self):
        cdef TaskClass tc
        cdef list errors = []
        for tc in self._classes:
            tc.take_errors(errors)
        return errors

    cdef int flush_collections(self) except -1:
        cdef TiledMatrix matrix
        for matrix, used in list(self._collections.items()):
            if used:
                self.data_flush_all(matrix)
        self._epoch += 1
        return 0

    cdef destroy(self):
        # A DTD taskpool can be freed whether or not it was enqueued.
        cdef TaskClass tc
        cdef c.parsec_taskpool_t *tp = self._tp
        for tc in self._classes:
            tc.release(tp)
        self._classes = []
        if self._context._ctx != NULL:
            self._destruct(tp)
        BaseTaskpool.forget(self)
        # The collections are kept alive until no task can refer to them.
        self._collections = {}
