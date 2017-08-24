.. include:: defs.rst

.. _installation:

Installation
============

|zfp| consists of three distinct parts: a compression library written in C,
a set of C++ header files that implement compressed arrays, and a set of
C and C++ examples.  The main compression codec is written in C and should
conform to both the ISO C89 and C99 standards.  The C++ array classes are
implemented entirely in header files and can be included as is, but since
they call the compression library, applications must link with
:file:`libzfp`.

On Linux, macOS, and MinGW, |zfp| is easiest compiled using gcc and gmake.
CMake support is also available, e.g. for Windows builds.  See below for
instructions on GNU and CMake builds.

|zfp| has successfully been built and tested using these compilers:

* gcc versions 4.4.7, 4.7.2, 4.8.2, 4.9.2, 5.3.1, 6.2.1
* icc versions 12.0.5, 12.1.5, 15.0.4, 16.0.1
* clang version 3.6.0
* xlc version 12.1
* mingw32-gcc version 4.8.1
* Visual Studio version 14.0

**NOTE: zfp requires 64-bit compiler and operating system support**.

GNU builds 
----------

To compile |zfp| using `gcc <https://gcc.gnu.org>`_, type::

    make

from the |zfp| root directory.  This builds :file:`libzfp` as a static
library as well as utilities and example programs.  To optionally create
a shared library, type::

    make shared

and set :envvar:`LD_LIBRARY_PATH` to point to :file:`lib`.  To test the compressor,
type::

    make test

If the compilation or regression tests fail, it is possible that some of
the macros in the file :file:`Config` have to be adjusted.  Also, the tests
may fail due to minute differences in the computed floating-point fields
being compressed (as indicated by checksum errors).  It is surprisingly
difficult to portably generate a floating-point array that agrees
bit-for-bit across platforms.  If most tests succeed and the failures
result in byte sizes and error values reasonably close to the expected
values, then it is likely that the compressor is working correctly.

CMake builds
------------

To build |zfp| using `CMake <https://cmake.org>`_ on Linux or macOS, start
a Unix shell and type::

    mkdir build
    cd build
    cmake ..
    make

To also build the examples, replace the cmake line with::

    cmake -DBUILD_EXAMPLES=ON ..

To build |zfp| using Visual Studio on Windows, start an
`MSBuild shell <https://msdn.microsoft.com/en-us/library/ms164311.aspx>`_
and type::

    mkdir build
    cd build
    cmake ..
    msbuild /p:Configuration=Release zfp.sln
    msbuild /p:Configuration=Debug   zfp.sln

This builds |zfp| in both debug and release mode.  See the instructions for
Linux on how to change the cmake line to also build the example programs.

.. index::
   single: Configuration
.. _config:

Compile-time macros
-------------------

The behavior of |zfp| can be configured at compile time via a set of macros.
For GNU builds, these macros are set in the file :file:`Config`.  For CMake
builds, use the :code:`-D` option on the cmake line as in the example above.

.. c:macro:: ZFP_INT64
.. c:macro:: ZFP_INT64_SUFFIX
.. c:macro:: ZFP_UINT64
.. c:macro:: ZFP_UINT64_SUFFIX

  64-bit signed and unsigned integer types and their literal suffixes.
  Platforms on which :code:`long int` is 32 bits wide may require
  :code:`long long int` as type and :code:`ll` as suffix.  These macros
  are primarily relevant when compiling in C89 mode.
  Defaults: :code:`long int`, :code:`l`, :code:`unsigned long int`, and
  :code:`ul`, respectively.

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
