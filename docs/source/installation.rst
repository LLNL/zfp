.. include:: defs.rst

.. _installation:

Installation
============

|zfp| consists of four distinct parts: a compression library written in C,
a set of C++ header files that implement compressed arrays and corresponding
C wrappers, optional Python and Fortran bindings, and a set of C and C++
examples and utilities.  The main compression codec is written in C and
should conform to both the ISO C89 and C99 standards.  The C++ array classes
are implemented entirely in header files and can be included as is, but since
they call the compression library, applications must link with |libzfp|.

|zfp| is preferably built using `CMake <https://cmake.org>`_, although the
core library can also be built using GNU make on Linux, macOS, and MinGW.
|zfp| has successfully been built and tested using these compilers:

* gcc versions 4.4.7, 4.7.3, 4.8.5, 4.9.4, 5.5.0, 6.1.0, 6.4.0, 7.1.0, 7.3.0, 8.1.0
* icc versions 14.0.3, 15.0.6, 16.0.4, 17.0.2, 18.0.2, 19.0.3
* clang versions 3.9.1, 4.0.0, 5.0.0, 6.0.0
* MinGW version 5.3.0
* Visual Studio versions 14 (2015), 15 (2017)

|zfp| conforms to various language standards, including C89, C99, C++98,
C++11, and C++14.

.. note::
  |zfp| requires 64-bit compiler and operating system support.

.. _cmake_builds:

CMake Builds
------------

To build |zfp| using `CMake <https://cmake.org>`_ on Linux or macOS, start
a Unix shell and type::

    cd zfp-0.5.5
    mkdir build
    cd build
    cmake ..
    make

To also build the examples, replace the cmake line with::

    cmake -DBUILD_EXAMPLES=ON ..

By default, CMake builds will attempt to locate and use
`OpenMP <http://www.openmp.org>`_.  To disable OpenMP, type::

    cmake -DZFP_WITH_OPENMP=OFF ..

To build |zfp| using Visual Studio on Windows, start a DOS shell
and type::

    cd zfp-0.5.5
    mkdir build
    cd build
    cmake ..
    cmake --build . --config Release

This builds |zfp| in release mode.  Replace 'Release' with 'Debug' to
build |zfp| in debug mode.  See the instructions for Linux on how to
change the cmake line to also build the example programs.


.. _gnu_builds:

GNU Builds 
----------

To build |zfp| using `gcc <https://gcc.gnu.org>`_ without
`OpenMP <http://www.openmp.org>`_, type::

    cd zfp-0.5.5
    gmake

This builds |libzfp| as a static library as well as the |zfp|
command-line utility.  To enable OpenMP parallel compression, type::

    gmake ZFP_WITH_OPENMP=1

.. note::
  GNU builds expose only limited functionality of |zfp|.  For instance,
  CUDA and Python support are not included.  For full functionality,
  build |zfp| using CMake.


Testing
-------

To test that |zfp| is working properly, type::

    ctest

or using GNU make::

    gmake test

If the GNU build or regression tests fail, it is possible that some of
the macros in the file :file:`Config` have to be adjusted.  Also, the tests
may fail due to minute differences in the computed floating-point fields
being compressed, which will be indicated by checksum errors.  If most
tests succeed and the failures result in byte sizes and error values
reasonably close to the expected values, then it is likely that the
compressor is working correctly.


.. index::
   single: Build Targets
.. _targets:

Build Targets
-------------

To specify which components to build, set the macros below to
:code:`ON` (CMake) or :code:`1` (GNU make), e.g.,
::

  cmake -DBUILD_UTILITIES=OFF -DBUILD_EXAMPLES=ON ..

or using GNU make
::

  gmake BUILD_UTILITIES=0 BUILD_EXAMPLES=1

Regardless of the settings below, |libzfp| will always be built.

.. c:macro:: BUILD_ALL

  Build all subdirectories; enable all options (except
  :c:macro:`BUILD_SHARED_LIBS`).
  Default: off.

.. c:macro:: BUILD_CFP

  Build |libcfp| for C bindings to compressed arrays.
  Default: off.

.. c:macro:: BUILD_ZFPY

  Build |zfpy| for Python bindings to the C API.

  CMake will attempt to automatically detect the Python installation to use.
  If CMake finds multiple Python installations, it will use the newest one.
  To specify a specific Python installation to use, set
  :c:macro:`PYTHON_LIBRARY` and :c:macro:`PYTHON_INCLUDE_DIR` on the
  cmake line::

      cmake -DBUILD_ZFPY=ON -DPYTHON_LIBRARY=/path/to/lib/libpython2.7.so -DPYTHON_INCLUDE_DIR=/path/to/include/python2.7 ..

  CMake default: off.
  GNU make default: off and ignored.

.. c:macro:: BUILD_ZFORP

  Build |libzforp| for Fortran bindings to the C API.  Requires Fortran
  standard 2003 or later.  GNU make users may specify the Fortran compiler
  to use via
  ::

      gmake BUILD_ZFORP=1 FC=/path/to/fortran-compiler

  Default: off.

.. c:macro:: BUILD_UTILITIES

  Build |zfpcmd| command-line utility for compressing binary files.
  Default: on.

.. c:macro:: BUILD_EXAMPLES

  Build code examples.
  Default: off.

.. c:macro:: BUILD_TESTING

  Build |testzfp| and (when on the GitHub
  `develop branch <https://github.com/LLNL/zfp/tree/develop>`_) unit tests.
  Default: on.

.. c:macro:: BUILD_SHARED_LIBS

  Build shared objects (:file:`.so`, :file:`.dylib`, or :file:`.dll` files).
  On macOS, the :code:`SOFLAGS` line in the :file:`Config` file may have
  to be uncommented when building with GNU make.
  CMake default: on.
  GNU make default: off.


.. index::
   single: Configuration
.. _config:

Configuration
-------------

The behavior of |zfp| can be configured at compile time via a set of macros
in the same manner that :ref:`build targets <targets>` are specified, e.g.,
::

    cmake -DZFP_WITH_OPENMP=OFF ..

.. c:macro:: ZFP_INT64
.. c:macro:: ZFP_INT64_SUFFIX
.. c:macro:: ZFP_UINT64
.. c:macro:: ZFP_UINT64_SUFFIX

  64-bit signed and unsigned integer types and their literal suffixes.
  Platforms on which :code:`long int` is 32 bits wide may require
  :code:`long long int` as type and :code:`ll` as suffix.  These macros
  are relevant **only** when compiling in C89 mode.  When compiling in
  C99 mode, integer types are taken from :file:`stdint.h`.
  Defaults: :code:`long int`, :code:`l`, :code:`unsigned long int`, and
  :code:`ul`, respectively.

.. c:macro:: ZFP_WITH_OPENMP

  CMake and GNU make macro for enabling or disabling OpenMP support.  CMake
  builds will by default enable OpenMP when available.  Set this macro to
  0 or OFF to disable OpenMP support.  For GNU builds, OpenMP is disabled by
  default.  Set this macro to 1 or ON to enable OpenMP support.  See also
  OMPFLAGS in :file:`Config` in case the compiler does not recognize
  :code:`-fopenmp`.  NOTE: clang currently does not support OpenMP on macOS.
  CMake default: on.
  GNU make default: off.

.. c:macro:: ZFP_WITH_CUDA

  CMake macro for enabling or disabling CUDA support for
  GPU compression and decompression.  When enabled, CUDA and a compatible
  host compiler must be installed.  For a full list of compatible compilers,
  please consult the
  `NVIDIA documentation <https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/>`_.
  If a CUDA installation is in the user's path, it will be
  automatically found by CMake.  Alternatively, the CUDA binary directory 
  can be specified using the :envvar:`CUDA_BIN_DIR` environment variable.
  CMake default: off.
  GNU make default: off and ignored.

.. c:macro:: ZFP_WITH_ALIGNED_ALLOC

  Use aligned memory allocation in an attempt to align compressed blocks
  on hardware cache lines.
  Default: undefined/off.

.. c:macro:: ZFP_WITH_CACHE_TWOWAY

  Use a two-way skew-associative rather than direct-mapped cache.  This
  incurs some overhead that may be offset by better cache utilization.
  Default: undefined/off.

.. c:macro:: ZFP_WITH_CACHE_FAST_HASH

  Use a simpler hash function for cache line lookup.  This is faster but may
  lead to more collisions.
  Default: undefined/off.

.. c:macro:: ZFP_WITH_CACHE_PROFILE

  Enable cache profiling to gather and print statistics on cache hit and miss
  rates.
  Default: undefined/off.

.. c:macro:: BIT_STREAM_WORD_TYPE

  Unsigned integer type used for buffering bits.  Wider types tend to give
  higher performance at the expense of lower bit rate granularity.  For
  portability of compressed files between little and big endian platforms,
  :c:macro:`BIT_STREAM_WORD_TYPE` should be set to :c:type:`uint8`.
  Default: :c:type:`uint64`.

.. c:macro:: ZFP_BIT_STREAM_WORD_SIZE

  CMake macro for indirectly setting :c:macro:`BIT_STREAM_WORD_TYPE`.  Valid
  values are 8, 16, 32, 64.
  Default: 64.

.. c:macro:: BIT_STREAM_STRIDED

  Enable support for strided bit streams that allow for non-contiguous memory
  layouts, e.g., to enable progressive access.
  Default: undefined/off.

.. c:macro:: CFP_NAMESPACE

  Macro for renaming the outermost |cfp| namespace, e.g., to avoid name
  clashes.
  Default: :code:`cfp`.

.. c:macro:: PYTHON_LIBRARY

  Path to the Python library, e.g., :file:`/usr/lib/libpython2.7.so`.
  CMake default: undefined/off.
  GNU make default: off and ignored.

.. c:macro:: PYTHON_INCLUDE_DIR

  Path to the Python include directory, e.g., :file:`/usr/include/python2.7`.
  CMake default: undefined/off.
  GNU make default: off and ignored.


Dependencies
------------

The core |zfp| library and compressed arrays require only a C89 and C++98
compiler.  The optional components have additional dependencies, as outlined
in the sections below.

CMake
^^^^^

CMake builds require version 3.1 or later on Linux and macOS and version
3.4 or later on Windows.  CMake is available `here <https://cmake.org>`_.

CUDA
^^^^

CUDA support requires CMake and a compatible host compiler (see
:c:macro:`ZFP_WITH_CUDA`).

Python
^^^^^^

The optional Python bindings require CMake and the following minimum
versions:

* Python: Python 2.7 & Python 3.5
* Cython: 0.22
* NumPy: 1.8.0

The necessary dependencies can be installed using ``pip`` and the |zfp|
:file:`requirements.txt`::

  pip install -r $ZFP_ROOT/python/requirements.txt

Fortran
^^^^^^^

The optional Fortran bindings require a Fortran 2003 compiler.
