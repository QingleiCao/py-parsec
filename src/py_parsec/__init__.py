# Copyright (c) 2026      The University of Tennessee and The University
#                         of Tennessee Research Foundation.  All rights
#                         reserved.
"""Python bindings for the PaRSEC task-based runtime.

* :class:`Context` -- the PaRSEC runtime of the process (``parsec_init``).
* :class:`Taskpool` -- any taskpool; :mod:`py_parsec.dtd` builds DTD ones,
  library calls such as :func:`py_parsec.matrix.redistribute_new` return PTG
  ones, and all are scheduled with :meth:`Context.add_taskpool`.
* :class:`BlockCyclic` -- a 2D block-cyclic tiled matrix usable by every DSL.
"""

try:
    from importlib.metadata import PackageNotFoundError, version as _dist_version
    __version__ = _dist_version("py-parsec")
except PackageNotFoundError:  # pragma: no cover - running from a source tree
    __version__ = "0+unknown"

from .runtime import (
    DEV_ALL,
    DEV_CPU,
    DEV_CUDA,
    DEV_HIP,
    DEV_LEVEL_ZERO,
    HOOK_RETURN_AGAIN,
    HOOK_RETURN_ASYNC,
    HOOK_RETURN_DISABLE,
    HOOK_RETURN_DONE,
    HOOK_RETURN_ERROR,
    HOOK_RETURN_NEXT,
    Context,
    Taskpool,
    build_info,
    version,
)
from .matrix import BlockCyclic, TiledMatrix
from . import dtd, matrix

__all__ = [
    "__version__",
    "Context",
    "Taskpool",
    "TiledMatrix",
    "BlockCyclic",
    "dtd",
    "matrix",
    "version",
    "build_info",
    "DEV_CPU",
    "DEV_CUDA",
    "DEV_HIP",
    "DEV_LEVEL_ZERO",
    "DEV_ALL",
    "HOOK_RETURN_DONE",
    "HOOK_RETURN_AGAIN",
    "HOOK_RETURN_NEXT",
    "HOOK_RETURN_DISABLE",
    "HOOK_RETURN_ASYNC",
    "HOOK_RETURN_ERROR",
]
