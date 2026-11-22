# Copyright (c) 2026      The University of Tennessee and The University
#                         of Tennessee Research Foundation.  All rights
#                         reserved.
"""Build script of the extension modules of py-parsec.

PaRSEC (a shared ``libparsec`` and its headers) is located, in order, from:

* ``PARSEC_INCLUDE_DIRS`` and ``PARSEC_LIBRARY_DIRS`` (``os.pathsep``
  separated lists), to build against a PaRSEC that has no pkg-config file;
* ``pkg-config parsec``, looking first in ``$PARSEC_ROOT/lib*/pkgconfig``
  when ``PARSEC_ROOT`` is set to the PaRSEC installation prefix.

``CC`` selects the C compiler. The modules link ``libparsec`` with a run path
to its directory. The bindings read some PaRSEC structures directly, so the
modules are rebuilt whenever the PaRSEC they are built against changes, and
they refuse to load with another PaRSEC build (see ``py_parsec.build_info``).
"""

import hashlib
import os
import shlex
import subprocess
from pathlib import Path

from setuptools import Extension, setup
from setuptools.command.build_ext import build_ext

HERE = Path(__file__).resolve().parent
PACKAGE = Path("src") / "py_parsec"


def _split(value):
    return [p for p in (value or "").split(os.pathsep) if p]


def _pkg_config():
    """Include directories, other C flags and library directories of PaRSEC."""
    env = dict(os.environ)
    root = os.environ.get("PARSEC_ROOT")
    if root:
        dirs = [str(Path(root) / d / "pkgconfig") for d in ("lib", "lib64")]
        env["PKG_CONFIG_PATH"] = os.pathsep.join(dirs + _split(env.get("PKG_CONFIG_PATH")))

    def query(*options):
        return subprocess.run(["pkg-config", *options, "parsec"], env=env, check=True,
                              capture_output=True, text=True).stdout

    try:
        cflags = shlex.split(query("--cflags"))
        libdir = query("--variable=libdir").strip()
    except (OSError, subprocess.CalledProcessError) as e:
        detail = getattr(e, "stderr", "") or str(e)
        raise SystemExit("py-parsec: PaRSEC not found. Set PARSEC_ROOT to the PaRSEC installation "
                         "prefix (the directory holding lib*/pkgconfig/parsec.pc), or set "
                         f"PARSEC_INCLUDE_DIRS and PARSEC_LIBRARY_DIRS.\npkg-config: {detail.strip()}") from None
    return ([f[2:] for f in cflags if f.startswith("-I")], [libdir],
            [f for f in cflags if not f.startswith("-I")])


def _stamp(include_dirs, library_dirs, cflags):
    """A file whose content identifies the PaRSEC build; rewritten only when
    that changes, so that setuptools recompiles the modules that depend on it."""
    h = hashlib.sha256()
    for item in include_dirs + library_dirs + cflags:
        h.update(item.encode() + b"\0")
    for d in include_dirs:
        config = Path(d) / "parsec" / "parsec_config.h"
        if config.is_file():
            h.update(config.read_bytes())
            break
    stamp = HERE / "build" / "parsec-build.stamp"
    content = h.hexdigest() + "\n"
    if not stamp.is_file() or stamp.read_text() != content:
        stamp.parent.mkdir(parents=True, exist_ok=True)
        stamp.write_text(content)
    return str(stamp)


class BuildExt(build_ext):
    """Locate PaRSEC and cythonize only when the extension modules are built,
    so that making an sdist needs neither."""

    def build_extensions(self):
        import numpy
        from Cython.Build import cythonize

        include_dirs = _split(os.environ.get("PARSEC_INCLUDE_DIRS"))
        library_dirs = _split(os.environ.get("PARSEC_LIBRARY_DIRS"))
        cflags = []
        if not include_dirs:
            include_dirs, library_dirs, cflags = _pkg_config()
        stamp = _stamp(include_dirs, library_dirs, cflags)
        for ext in self.extensions:
            ext.include_dirs = [str(PACKAGE), numpy.get_include()] + include_dirs
            ext.library_dirs = library_dirs
            ext.runtime_library_dirs = library_dirs
            ext.extra_compile_args = cflags
            ext.depends = [str(PACKAGE / "pyparsec.h"), stamp]
        # Keep the Extension objects setuptools has prepared; take the C
        # sources of the cythonized copies.
        sources = {ext.name: ext.sources for ext in cythonize(
            self.extensions,
            include_path=["src"],
            compiler_directives={"language_level": 3, "embedsignature": True},
        )}
        for ext in self.extensions:
            ext.sources = sources[ext.name]
        super().build_extensions()


setup(
    ext_modules=[
        Extension(f"py_parsec.{name}", [str(PACKAGE / f"{name}.pyx")],
                  libraries=["parsec"],
                  define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")])
        for name in ("runtime", "matrix", "dtd")
    ],
    cmdclass={"build_ext": BuildExt},
)
