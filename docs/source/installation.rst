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

|zfp| is preferably built using `CMake <https://cmake.org>`__, although the
core library can also be built using GNU make on Linux, macOS, and MinGW.

|zfp| conforms to various language standards, including C89, C99, C++98,
C++11, and C++14.

.. note::
  |zfp| requires 64-bit compiler and operating system support.

.. _cmake_builds:

CMake Builds
------------

To build |zfp| using `CMake <https://cmake.org>`__ on Linux or macOS, start
a Unix shell and type::

    cd zfp-1.0.0
    mkdir build
    cd build
    cmake ..
    make

To also build the examples, replace the cmake line with::

    cmake -DBUILD_EXAMPLES=ON ..

By default, CMake builds will attempt to locate and use
`OpenMP <http://www.openmp.org>`__.  To disable OpenMP, type::

    cmake -DZFP_WITH_OPENMP=OFF ..

To build |zfp| using Visual Studio on Windows, start a DOS shell
and type::

    cd zfp-1.0.0
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

To build |zfp| using `gcc <https://gcc.gnu.org>`__ without
`OpenMP <http://www.openmp.org>`__, type::

    cd zfp-1.0.0
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

  Build |libcfp| for C bindings to the compressed-array classes.
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
  standard 2018 or later.  GNU make users may specify the Fortran compiler
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

  Build |testzfp| tests.
  Default: on.


.. c:macro:: BUILD_TESTING_FULL

  Build all unit tests.
  Default: off.


.. c:macro:: BUILD_SHARED_LIBS

  Build shared objects (:file:`.so`, :file:`.dylib`, or :file:`.dll` files).
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
  ``-fopenmp``.  For example, Apple clang requires
  ``OMPFLAGS=-Xclang -fopenmp``, ``LDFLAGS=-lomp``, and an installation of
  ``libomp``.
  CMake default: on.
  GNU make default: off.


.. c:macro:: ZFP_WITH_CUDA

  CMake macro for enabling or disabling CUDA support for
  GPU compression and decompression.  When enabled, CUDA and a compatible
  host compiler must be installed.  For a full list of compatible compilers,
  please consult the
  `NVIDIA documentation <https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/>`__.
  If a CUDA installation is in the user's path, it will be
  automatically found by CMake.  Alternatively, the CUDA binary directory 
  can be specified using the :envvar:`CUDA_BIN_DIR` environment variable.
  CMake default: off.
  GNU make default: off and ignored.

.. _rounding:
.. c:macro:: ZFP_ROUNDING_MODE

  **Experimental feature**.  By default, |zfp| coefficients are truncated,
  not rounded, which can result in biased errors (see
  FAQ :ref:`#30 <q-err-dist>`).  To counter this, two rounding modes are
  available: :code:`ZFP_ROUND_FIRST` (round during compression; analogous
  to mid-tread quantization) and :code:`ZFP_ROUND_LAST` (round during
  decompression; analogous to mid-riser quantization).  With
  :code:`ZFP_ROUND_LAST`, the values returned on decompression are slightly
  modified (and usually closer to the original values) without impacting the
  compressed data itself.  This rounding mode works with all
  :ref:`compression modes <modes>`.
  With :code:`ZFP_ROUND_FIRST`, the values are modified before compression,
  thus impacting the compressed stream.  This rounding mode tends to be more
  effective at reducing bias, but is invoked only with
  :ref:`fixed-precision <mode-fixed-precision>` and
  :ref:`fixed-accuracy <mode-fixed-accuracy>` compression modes.
  Both of these rounding modes break the regression tests since they alter
  the compressed or decompressed representation, but they may be used with
  libraries built with the default rounding mode, :code:`ZFP_ROUND_NEVER`,
  and versions of |zfp| that do not support a rounding mode with no adverse
  effects.
  Note: :c:macro:`ZFP_ROUNDING_MODE` is currently supported only by the
  :code:`serial` and :code:`omp` :ref:`execution policies <execution>`.
  Default: :code:`ZFP_ROUND_NEVER`.

.. c:macro:: ZFP_WITH_TIGHT_ERROR

  **Experimental feature**.  When enabled, this feature takes advantage of the
  error reduction associated with proper rounding; see
  :c:macro:`ZFP_ROUNDING_MODE`.  The reduced error due to rounding
  allows the tolerance in :ref:`fixed-accuracy mode <mode-fixed-accuracy>`
  to be satisfied using fewer bits of compressed data.  As a result, when
  enabled, the observed maximum absolute error is closer to the tolerance and
  the compression ratio is increased.  This feature requires the rounding mode
  to be :code:`ZFP_ROUND_FIRST` or :code:`ZFP_ROUND_LAST` and is supported
  only by the :code:`serial` and :code:`omp`
  :ref:`execution policies <execution>`.
  Default: undefined/off.

.. c:macro:: ZFP_WITH_DAZ

  When enabled, blocks consisting solely of subnormal floating-point numbers
  (tiny numbers close to zero) are treated as blocks of all zeros
  (DAZ = denormals-are-zero).  The main purpose of this option is to avoid the
  potential for floating-point overflow in the |zfp| implementation that may
  occur in step 2 of the
  :ref:`lossy compression algorithm <algorithm-lossy>` when converting to
  |zfp|'s block-floating-point representation (see
  `Issue #119 <https://github.com/LLNL/zfp/issues/119>`__).
  Such overflow tends to be benign but loses all precision and usually
  results in "random" subnormals upon decompression.  When enabled, compressed
  streams may differ slightly but are decompressed correctly by libraries
  built without this option.  This option may break some regression tests.
  Note: :c:macro:`ZFP_WITH_DAZ` is currently ignored by all
  :ref:`execution policies <execution>` other than :code:`serial` and
  :code:`omp`.
  Default: undefined/off.

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
  higher performance at the expense of lower
  :ref:`bit rate granularity <q-granularity>`.  For portability of compressed
  files between little and big endian platforms,
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

CMake builds require version 3.9 or later.  CMake is available
`here <https://cmake.org>`__.

OpenMP
^^^^^^

OpenMP support requires OpenMP 2.0 or later.

CUDA
^^^^

CUDA support requires CUDA 7.0 or later, CMake, and a compatible host
compiler (see :c:macro:`ZFP_WITH_CUDA`).

C/C++
^^^^^

The |zfp| C library and |cfp| C wrappers around the compressed-array
classes conform to the C90 standard
(`ISO/IEC 9899:1990 <https://www.iso.org/standard/17782.html>`__).
The C++ classes conform to the C++98 standard
(`ISO/IEC 14882:1998 <https://www.iso.org/standard/25845.html>`__).

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

The optional Fortran bindings require a Fortran 2018 compiler.
