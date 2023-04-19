.. include:: defs.rst

.. py:module:: zfpy

.. _zfpy:

Python Bindings
===============

|zfp| |zfpyrelease| adds |zfpy|: Python bindings that allow compressing
and decompressing `NumPy <https://www.numpy.org>`_ integer and
floating-point arrays.  The |zfpy| implementation is based on
`Cython <https://cython.org>`_ and requires both NumPy and Cython
to be installed. The |zfpy| API is limited to two functions, for compression 
and decompression, which are described below.

Compression
-----------

.. py:function:: zfpy.compress_numpy(arr, tolerance = -1, rate = -1, precision = -1, write_header = True, policy = policy_serial)

  Compress NumPy array, *arr*, and return a compressed byte stream.  The
  non-expert :ref:`compression mode <modes>` is selected by setting one of
  *tolerance*, *rate*, or *precision*.  If none of these arguments is
  specified, then :ref:`reversible mode <mode-reversible>` is used.  By
  default, a header that encodes array shape and scalar type as well as
  compression parameters is prepended, which can be omitted by setting
  *write_header* to *False*.  If this function fails for any reason, an
  exception is thrown.

|zfpy| compression currently requires a NumPy array
(`ndarray <https://www.numpy.org/devdocs/reference/arrays.ndarray.html>`_)
populated with the data to be compressed.  The array metadata (i.e.,
shape, strides, and scalar type) are used to automatically populate the
:c:type:`zfp_field` structure passed to :c:func:`zfp_compress`.  By default,
all that is required to be passed to the compression function is the
NumPy array; this will result in a stream that includes a header and is
losslessly compressed using the :ref:`reversible mode <mode-reversible>`.
For example::

  import zfpy
  import numpy as np

  my_array = np.arange(1, 20)
  compressed_data = zfpy.compress_numpy(my_array)
  decompressed_array = zfpy.decompress_numpy(compressed_data)

  # confirm lossless compression/decompression
  np.testing.assert_array_equal(my_array, decompressed_array)

Using the fixed-accuracy, fixed-rate, or fixed-precision modes simply requires
setting one of the *tolerance*, *rate*, or *precision* arguments, respectively.
For example::

  compressed_data = zfpy.compress_numpy(my_array, tolerance=1e-3)
  decompressed_array = zfpy.decompress_numpy(compressed_data)

  # Note the change from "equal" to "allclose" due to the lossy compression
  np.testing.assert_allclose(my_array, decompressed_array, atol=1e-3)

Since NumPy arrays are C-ordered by default (i.e., the rightmost index
varies fastest) and :c:func:`zfp_compress` assumes Fortran ordering
(i.e., the leftmost index varies fastest), :py:func:`compress_numpy`
automatically reverses the order of dimensions and strides in order to
improve the expected memory access pattern during compression.
The :py:func:`decompress_numpy` function also reverses the order of
dimensions and strides, and therefore decompression will restore the
shape of the original array.  Note, however, that the |zfp| stream does
not encode the memory layout of the original NumPy array, and therefore
layout information like strides, contiguity, and C vs. Fortran order
may not be preserved.  Nevertheless, |zfpy| correctly compresses NumPy
arrays with any memory layout, including Fortran ordering and non-contiguous
storage.

Byte streams produced by :py:func:`compress_numpy` can be decompressed
by the :ref:`zfp command-line tool <zfpcmd>`.  In general, they cannot
be :ref:`deserialized <serialization>` as compressed arrays, however.

.. note::
  :py:func:`decompress_numpy` requires a header to decompress properly, so do
  not set *write_header* = *False* during compression if you intend to
  decompress the stream with |zfpy|.

Decompression
-------------

.. py:function:: zfpy.decompress_numpy(compressed_data, policy = policy_serial)

  Decompress a byte stream, *compressed_data*, produced by
  :py:func:`compress_numpy` (with header enabled) and return the
  decompressed NumPy array. This function throws an exception upon error.

:py:func:`decompress_numpy` consumes a compressed stream that includes a
header and produces a NumPy array with metadata populated based on the
contents of the header.  Stride information is not stored in the |zfp|
header, so :py:func:`decompress_numpy` assumes that the array was compressed
with the first (leftmost) dimension varying fastest (typically referred to as
Fortran-ordering).  The returned NumPy array is in C-ordering (the default
for NumPy arrays), so the shape of the returned array is reversed from
the shape information stored in the embedded header.  For example, if the
header declares the array to be of shape (*nx*, *ny*, *nz*) = (2, 4, 8),
then the returned NumPy array will have a shape of (8, 4, 2).
Since the :py:func:`compress_numpy` function also reverses the order of
dimensions, arrays both compressed and decompressed with |zfpy| will have
compatible shape.

.. note::
  Decompressing a stream without a header requires using the
  internal :py:func:`_decompress` Python function (or the
  :ref:`C API <hl-api>`).

.. py:function:: zfpy._decompress(compressed_data, ztype, shape, out = None, tolerance = -1, rate = -1, precision = -1)

  Decompress a headerless compressed stream (if a header is present in
  the stream, it will be incorrectly interpreted as compressed data).
  *ztype* specifies the array scalar type while *shape* specifies the array
  dimensions; both must be known by the caller.  The compression mode is
  selected by specifying one (or none) of *tolerance*, *rate*, and
  *precision*, as in :py:func:`compress_numpy`, and also must be known
  by the caller.  If *out = None*, a new NumPy array is allocated.  Otherwise,
  *out* specifies the NumPy array or memory buffer to decompress into.
  Regardless, the decompressed NumPy array is returned unless an error occurs,
  in which case an exception is thrown.G

In :py:func:`_decompress`, *ztype* is one of the |zfp| supported scalar types
(see :c:type:`zfp_type`), which are available in |zfpy| as
::

    type_int32 = zfp_type_int32
    type_int64 = zfp_type_int64
    type_float = zfp_type_float
    type_double = zfp_type_double

These can be manually specified (e.g., :code:`zfpy.type_int32`) or generated
from a NumPy *dtype* (e.g., :code:`zfpy.dtype_to_ztype(array.dtype)`).

If *out* is specified, the data is decompressed into the *out* buffer.
*out* can be a NumPy array or a pointer to memory large enough to hold the
decompressed data.  Regardless of the type of *out* and whether it is provided,
:py:func:`_decompress` always returns a NumPy array.  If *out* is not
provided, then the array is allocated for the user.  If *out* is provided,
then the returned NumPy array is just a pointer to or wrapper around the
user-supplied *out*.  If *out* is a NumPy array, then its shape and scalar
type must match the required arguments *shape* and *ztype*.  To avoid this
constraint check, use :code:`out = ndarray.data` rather than
:code:`out = ndarray` when calling :py:func:`_decompress`.

.. warning::
  :py:func:`_decompress` is an "experimental" function currently used
  internally for testing.  It does allow decompression of streams without
  headers, but providing too small of an output buffer or incorrectly
  specifying the shape or strides can result in segmentation faults.
  Use with care.

Execution Policy
----------------

|zfp| |zfpyversrelease| adds execution policy support for |zfpy| compression and 
decompression. These are passed as optional policy parameters to 
:code:`zfpy.compress_numpy` and :code:`zfpy.decompress_numpy`. The following 
policies are supported (see :c:type:`zfp_exec_policy`)
::

  zfpy.policy_serial
  zfpy.policy_omp
  zfpy.policy_cuda

.. note::
  Both `zfpy.policy_omp` and `zfpy.policy_cuda` expect that the underlying |zfp| 
  library be built with support for the equivalent |zfp| execution policies. See 
  the :ref:`compression modes table <compression_mode_support>` for details on 
  execution policy support.

Version
-------

.. py:module:: zfpy.version

|zfp| |zfpyversrelease| adds support for querying version info much like that 
provided by the |zfp| `version.h` header. This is accessible as 
`zfpy.version`. It may also be accessed seperately via 
:code:`import zfpy_version`.

.. py:function:: zfpy.version.geq(major, minor, patch, tweak=zfpy.version.tweak)

  Returns whether the current version of zfpy is greater than or equal to the 
  given version.

.. py:data:: zfpy.version.version

  Version number as a string with the form "major.minor.patch".

.. py:data:: zfpy.version.full_version

  Version number as a string with the form "major.minor.patch.tweak".

.. py:data:: zfpy.version.version_string

  Full zfp version string (see :c:var:`zfp_version_string`).

.. py:data:: zfpy.version.major

  Integer representation of zfp major version

.. py:data:: zfpy.version.minor

  Integer representation of zfp minor version

.. py:data:: zfpy.version.patch

  Integer representation of the patch version

.. py:data:: zfpy.version.tweak

  Integer representation of the tweak version

.. py:data:: zfpy.version.codec

  Integer representation of zfp codec version

Much as with `version.h`, `zfpy_version` serves as a standalone method for 
validating that code will have access to version-dependent API calls prior to 
any attempts to access those calls. As some early versions of |zfpy| lack this 
functionality it is a good idea to verify it exists as part of version 
checking. For example::

  import zfpy

  if not hasattr(zfpy, 'version'):
      # Using version too old for zfpy_version, fall back to alternate option

  if zfpy.version.geq(req_major, req_minor, req_patch):
      # Use new API call
  else:
      # Fall back to alternate option
