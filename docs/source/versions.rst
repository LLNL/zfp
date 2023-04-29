.. include:: defs.rst

Release Notes
=============

1.0.0 (2022-08-01)
------------------

This release is not ABI compatible with prior releases due to numerous changes
to function signatures and data structures like ``zfp_field``.  However, few of
the API changes, other than to the |cfp| C API for compressed arrays, should
impact existing code.  Note that numerous header files have been renamed or
moved relative to prior versions.

**Added**

- ``zfp::const_array``: read-only variable-rate array that supports
  fixed-precision, fixed-accuracy, and reversible modes.
- Compressed-array classes for 4D data.
- ``const`` versions of array references, pointers, and iterators.
- A more complete API for pointers and iterators.
- |cfp| support for proxy references and pointers, iterators, and 
  (de)serialization.
- Support for pointers and iterators into array views.
- ``zfp::array::size_bytes()`` allows querying the size of different components
  of an array object (e.g., payload, cache, index, metadata, ...).
- Templated C++ wrappers around the low-level C API.
- A generic codec for storing blocks of uncompressed scalars in |zfp|'s
  C++ arrays.
- Additional functions for querying ``zfp_field`` and ``zfp_stream`` structs.
- ``zfp_config``: struct that encapsulates compression mode and parameters.
- Rounding modes for reducing bias in compression errors.
- New examples: ``array``, ``iteratorC``, and ``ppm``.

**Changed**

- Headers from ``array/``, ``cfp/include/``, and ``include/`` have been renamed
  and reorganized into a common ``include/`` directory.

  * The libzfp API is now confined to ``zfp.h``, ``zfp.hpp``, and ``zfp.mod``
    for C, C++, and Fortran bindings, respectively.  These all appear in
    the top-level ``include/`` directory upon installation.
  * C++ headers now use a ``.hpp`` suffix; C headers use a ``.h`` suffix.
  * C++ headers like ``array/zfparray.h`` have been renamed ``zfp/array.hpp``.
  * C headers like ``cfp/include/cfparrays.h`` have been renamed
    ``zfp/array.h``.

- ``size_t`` and ``ptrdiff_t`` replace ``uint`` and ``int`` for array sizes and
  strides in the array classes and C/Fortran APIs.
- ``zfp_bool`` replaces ``int`` as Boolean type in the C API.
- ``bitstream_offset`` and ``bitstream_size`` replace ``size_t`` to ensure
  support for 64-bit offsets into and lengths of bit streams.  Consequently,
  the ``bitstream`` API has changed accordingly.
- All array and view iterators are now random-access iterators.
- Array inspectors now return ``const_reference`` rather than a scalar
  type like ``float`` to allow obtaining a ``const_pointer`` to an element
  of an immutable array.
- ``zfp::array::compressed_data()`` now returns ``void*`` instead of
  ``uchar*``.
- The array (de)serialization API has been revised, resulting in new
  ``zfp::array::header`` and ``zfp::exception`` classes with new exception
  messages.
- The array ``codec`` class is now responsible for all details regarding
  compression.
- The compressed-array C++ implementation has been completely refactored to
  make it more modular, extensible, and reusable across array types.
- Array block shapes are now computed on the fly rather than stored.
- The |cfp| C API now wraps array objects in structs.
- The |zfpy| Python API now supports the more general ``memoryview`` over
  ``bytes`` objects for decompression.
- The zFORp Fortran module name is now ``zfp`` instead of ``zforp_module``.
- Some command-line options for the ``diffusion`` example have changed.
- CMake 3.9 or later is now required for CMake builds.

**Removed**

- ``zfp::array::get_header()`` has been replaced with a ``zfp::array::header``
  constructor that accepts an array object.
- ``ZFP_VERSION_RELEASE`` is no longer defined (use ``ZFP_VERSION_PATCH``).

**Fixed**

- #66: ``make install`` overwrites googletest.
- #84: Incorrect order of parameters in CUDA ``memset()``.
- #86: C++ compiler warns when ``__STDC_VERSION__`` is undefined.
- #87: ``CXXFLAGS`` is misspelled in ``cfp/src/Makefile``.
- #98: ``zfp_stream_maximum_size()`` underestimates size in reversible mode.
- #99: Incorrect ``private_view`` reads due to missing writeback.
- #109: Unused CPython array is incompatible with PyPy.
- #112: PGI compiler bug causes issues with memory alignment.
- #119: All-subnormal blocks may cause floating-point overflow.
- #121: CUDA bit offsets are limited to 32 bits.
- #122: ``make install`` does not install |zfp| command-line utility.
- #125: OpenMP bit offsets are limited to 32 bits.
- #126: ``make install`` does not install Fortran module.
- #127: Reversible mode reports incorrect compressed block size.
- #150: cmocka tests do not build on macOS.
- #154: Thread safety is broken in ``private_view`` and ``private_const_view``.
- ``ZFP_MAX_BITS`` is off by one.
- ``diffusionC``, ``iteratorC`` are not being built with ``gmake``.

----

0.5.5 (2019-05-05)
------------------

**Added**

- Support for reversible (lossless) compression of floating-point and
  integer data.
- Methods for serializing and deserializing |zfp|'s compressed arrays.
- Python bindings for compressing NumPy arrays.
- Fortran bindings to |zfp|'s high-level C API.

**Changed**

- The default compressed-array cache size is now a function of the total
  number of array elements, irrespective of array shape.

**Fixed**

- Incorrect handling of execution policy in |zfp| utility.
- Incorrect handling of decompression via header in |zfp| utility.
- Incorrect cleanup of device memory in CUDA decompress.
- Missing tests for failing mallocs.
- CMake does not install CFP when built.
- ``zfp_write_header()`` and ``zfp_field_metadata()`` succeed even if array
  dimensions are too large to fit in header.

----

0.5.4 (2018-10-01)
------------------

**Added**

- Support for CUDA fixed-rate compression and decompression.
- Views into compressed arrays for thread safety, nested array indexing,
  slicing, and array subsetting.
- C language bindings for compressed arrays.
- Support for compressing and decompressing 4D data.

**Changed**

- Execution policy now applies to both compression and decompression.
- Compressed array accessors now return Scalar type instead of
  ``const Scalar&`` to avoid stale references to evicted cache lines.

**Fixed**

- Incorrect handling of negative strides.
- Incorrect handling of arrays with more than 2\ :sup:`32` elements in |zfp|
  command-line tool.
- ``bitstream`` is not C++ compatible.
- Minimum cache size request is not respected.

----

0.5.3 (2018-03-28)
------------------

**Added**

- Support for OpenMP multithreaded compression (but not decompression).
- Options for OpenMP execution in |zfp| command-line tool.
- Compressed-array support for copy construction and assignment via deep
  copies.
- Virtual destructors to enable inheritance from |zfp| arrays.

**Changed**

- ``zfp_decompress()`` now returns the number of compressed bytes processed so
  far, i.e., the same value returned by ``zfp_compress()``.

----

0.5.2 (2017-09-28)
------------------

**Added**

- Iterators and proxy objects for pointers and references.
- Example illustrating how to use iterators and pointers.

**Changed**

- Diffusion example now optionally uses iterators.
- Moved internal headers under array to ``array/zfp``.
- Modified 64-bit integer typedefs to avoid the C89 non-compliant ``long long``
  and allow for user-supplied types and literal suffixes.
- Renamed compile-time macros that did not have a ``ZFP`` prefix.
- Rewrote documentation in reStructuredText and added complete documentation
  of all public functions, classes, types, and macros.

**Fixed**

- Issue with setting stream word type via CMake.

----

0.5.1 (2017-03-28)
------------------

This release primarily fixes a few minor issues but also includes changes in
anticipation of a large number of planned future additions to the library.
No changes have been made to the compressed format, which is backwards
compatible with version 0.5.0.

**Added**

- High-level API support for integer types.
- Example that illustrates in-place compression.
- Support for CMake builds.
- Documentation that discusses common issues with using |zfp|.

**Changed**

- Separated library version from CODEC version and added version string.
- Corrected inconsistent naming of ``BIT_STREAM`` macros in code and
  documentation.
- Renamed some of the header bit mask macros.
- ``stream_skip()`` and ``stream_flush()`` now return the number of bits
  skipped or output.
- Renamed ``stream_block()`` and ``stream_delta()`` to make it clear that they
  refer to strided streams.  Added missing definition of
  ``stream_stride_block()``.
- Changed ``int`` and ``uint`` types in places to use ``ptrdiff_t`` and
  ``size_t`` where appropriate.
- Changed API for ``zfp_set_precision()`` and ``zfp_set_accuracy()`` to not
  require the scalar type.
- Added missing ``static`` keyword in ``decode_block()``.
- Changed ``testzfp`` to allow specifying which tests to perform on the
  command line.
- Modified directory structure.

**Fixed**

- Bug that prevented defining uninitialized arrays.
- Incorrect computation of array sizes in ``zfp_field_size()``.
- Minor issues that prevented code from compiling on Windows.
- Issue with fixed-accuracy headers that caused unnecessary storage.

----

0.5.0 (2016-02-29)
------------------

This version introduces backwards incompatible changes to the CODEC.

**Added**

- Modified CODEC to more efficiently encode blocks whose values are all
  zero or are smaller in magnitude than the absolute error tolerance.
  This allows representing "empty" blocks using only one bit each.
- Added functions for compactly encoding the compression parameters
  and field meta data, e.g., for producing self-contained compressed
  streams.  Also added functions for reading and writing a header
  containing these parameters.

**Changed**

- Changed behavior of ``zfp_compress()`` and ``zfp_decompress()`` to not
  automatically rewind the bit stream.  This makes it easier to concatenate
  multiple compressed bit streams, e.g., when compressing vector fields or
  multiple scalars together.
- Changed the |zfp| example program interface to allow reading and writing
  compressed streams, optionally with a header.  The |zfp| tool can now be
  used to compress and decompress files as a stand alone utility.

----

0.4.1 (2015-12-28)
------------------

**Added**

- Added ``simple.c`` as a minimal example of how to call the compressor.

**Changed**

- Changed compilation of diffusion example to output two executables:
  one with and one without compression.

**Fixed**

- Bug that caused segmentation fault when compressing 3D arrays whose
  dimensions are not multiples of four.  Specifically, arrays of dimensions
  *nx* |times| *ny* |times| *nz*, with *ny* not a multiple of four, were not
  handled correctly.
- Modified ``examples/fields.h`` to ensure standard compliance.  Previously,
  C99 support was needed to handle the hex float constants, which are
  not supported in C++98.

----

0.4.0 (2015-12-05)
------------------

This version contains substantial changes to the compression algorithm that
improve PSNR by about 6 dB and speed by a factor of 2-3.  These changes are
not backward compatible with previous versions of |zfp|.

**Added**

- Support for 31-bit and 63-bit integer data, as well as shorter integer types.
- New examples for evaluating the throughput of the (de)compressor and for
  compressing grayscale images in the pgm format.
- Frequently asked questions.

**Changed**

- Rewrote compression codec entirely in C to make linking and calling
  easier from other programming languages, and to expose the low-level
  interface through C instead of C++.  This necessitated significant
  changes to the API as well.
- Minor changes to the C++ compressed array API, as well as major
  implementation changes to support the C library.  The namespace and
  public types are now all in lower case.

**Removed**

- Support for general fixed-point decorrelating transforms.

----

0.3.2 (2015-12-03)
------------------

**Fixed**

- Bug in ``Array::get()`` that caused the wrong cached block to be looked up,
  thus occasionally copying incorrect values back to parts of the array.

----

0.3.1 (2015-05-06)
------------------

**Fixed**

- Rare bug caused by exponent underflow in blocks with no normal and some
  subnormal numbers.

----

0.3.0 (2015-03-03)
------------------

This version modifies the default decorrelating transform to one that uses
only additions and bit shifts.  This new transform, in addition to being
faster, also has some theoretical optimality properties and tends to improve
rate distortion.  This change is not backwards compatible.

**Added**

- Compile-time support for parameterized transforms, e.g., to support other
  popular transforms like DCT, HCT, and Walsh-Hadamard.
- Floating-point traits to reduce the number of template parameters.  It is
  now possible to declare a 3D array as ``Array3<float>``, for example.
- Functions for setting the array scalar type and dimensions.
- ``testzfp`` for regression testing.

**Changed**

- Made forward transform range preserving: (-1, 1) is mapped to (-1, 1).
  Consequently Q1.62 fixed point can be used throughout.
- Changed the order in which bits are emitted within each bit plane to be more
  intelligent.  Group tests are now deferred until they are needed, i.e., just
  before the value bits for the group being tested.  This improves the quality
  of fixed-rate encodings, but has no impact on compressed size.
- Made several optimizations to improve performance.
- Consolidated several header files.

----

0.2.1 (2014-12-12)
------------------

**Added**

- Win64 support via Microsoft Visual Studio compiler.
- Documentation of the expected output for the diffusion example.

**Changed**

- Made several minor changes to suppress compiler warnings.

**Fixed**

- Broken support for IBM's ``xlc`` compiler.

----

0.2.0 (2014-12-02)
------------------

The compression interface from ``zfpcompress`` was relocated to a separate
library, called ``libzfp``, and modified to be callable from C.  This API now
uses a parameter object (``zfp_params``) to specify array type and dimensions
as well as compression parameters.

**Added**

- Several utility functions were added to simplify ``libzfp`` usage:

  * Functions for setting the rate, precision, and accuracy.
    Corresponding functions were also added to the ``Codec`` class.
  * A function for estimating the buffer size needed for compression.

- The ``Array`` class functionality was expanded:

  * Support for accessing the compressed bit stream stored with an array,
    e.g., for offline compressed storage and for initializing an already
    compressed array.
  * Functions for dynamically specifying the cache size.
  * The default cache is now direct-mapped instead of two-way associative.

**Fixed**

- Corrected the value of the lowest possible bit plane to account for both
  the smallest exponent and the number of bits in the significand.
- Corrected inconsistent use of rate and precision.  The rate refers to the
  number of compressed bits per floating-point value, while the precision
  refers to the number of uncompressed bits.  The ``Array`` API was changed
  accordingly.

----

0.1.0 (2014-11-12)
------------------

Initial beta release.
