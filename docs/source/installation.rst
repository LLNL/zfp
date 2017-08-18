.. include:: defs.rst

Installation
============

|zfp| consists of three distinct parts: a compression library written in C;
a set of C++ header files that implement compressed arrays; and a set of
C and C++ examples.  The main compression codec is written in C and should
conform to both the ISO C89 and C99 standards.  The C++ array classes are
implemented entirely in header files and can be included as is, but since
they call the compression library, applications must link with libzfp.

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

To compile |zfp| using gcc, type::

    make

from this directory.  This builds libzfp as a static library as well as
utilities and example programs.  To optionally create a shared library,
type::

    make shared

and set LD_LIBRARY_PATH to point to ./lib.  To test the compressor, type::

    make test

If the compilation or regression tests fail, it is possible that some of
the macros in the file 'Config' have to be adjusted.  Also, the tests may
fail due to minute differences in the computed floating-point fields
being compressed (as indicated by checksum errors).  It is surprisingly
difficult to portably generate a floating-point array that agrees
bit-for-bit across platforms.  If most tests succeed and the failures
result in byte sizes and error values reasonably close to the expected
values, then it is likely that the compressor is working correctly.

CMake builds
------------

To build |zfp| using CMake on Linux or macOS, start a Unix shell and type::

    mkdir build
    cd build
    cmake ..
    make

To also build the examples, replace the cmake line with::

    cmake -DBUILD_EXAMPLES=ON ..

To build |zfp| using Visual Studio on Windows, start an MSBuild shell and type::

    mkdir build
    cd build
    cmake ..
    msbuild /p:Configuration=Release zfp.sln
    msbuild /p:Configuration=Debug   zfp.sln

This builds |zfp| in both debug and release mode.  See the instructions for
Linux on how to change the cmake line to also build the example programs.

Compile-time macros
-------------------

**To be added**
