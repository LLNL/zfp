.. include:: defs.rst

.. py:module:: zfpy

.. _zfpy:

Python Bindings
===============

|zfp| |zfpyrelease| includes |zfpy|: Python bindings that allow |zfp| 
style compressed array access as well as functionality for
compressing and decompressing `NumPy <https://www.numpy.org>`_ integer 
and floating-point arrays. The |zfpy| implementation is based on
`Cython <https://cython.org>`_ and requires both NumPy and Cython
to be installed. Currently, |zfpy| supports only serial execution.

.. Note::
  zfpy compressed arrays are new as of <version number here>.

Arrays
------

|zfpy| arrays are designed to, as much as possible, mimic their NumPy 
counterparts. The current structure splits array types by data type 
and dimensionality (similarly to |cfp|). It is planned to eventually 
merge these into a single array type to simplify usage.

.. _array:
.. py:class:: zfpy.array1f
.. py:class:: zfpy.array1d
.. py:class:: zfpy.array2f
.. py:class:: zfpy.array2d
.. py:class:: zfpy.array3f
.. py:class:: zfpy.array3d

    .. py:attribute:: dtype

       String representation of C++ data type for uncompressed array
       scalars. Similar to NumPy array dtype.

    .. py:attribute:: shape

       Tuple representation of array dimensionality and sizes. 
       Equivalent to NumPy array shape.

.. _pyarray_ctor:
.. py:function:: zfpy.array1(size_x, rate, cache_size = 0)
.. py:function:: zfpy.array2(size_x, size_y, rate, cache_size = 0)
.. py:function:: zfpy.array3(size_x, size_y, size_z, rate, cache_size = 0)

  Currently |zfpy| provides fixed-rate array constructors. These are 
  likely to be replaced soon by external functions that will operate
  similarly to NumPy's array() functions. These are fixed-rate arrays 
  equivalent to those in |zfp| save that they do not allow for value 
  initialization via pointer. A cache size of zero will cause |zfpy| 
  to default to a standard cache size.

.. _pyarray_rate:
.. py:function:: zfpy.arrayALL.rate()

  Get current rate for array

.. _pyarray_set_rate:
.. py:function:: zfpy.arrayALL.set_rate(rate)

  Update rate for array. |zfpy| will adjust its rate to the closest acceptable 
  rate.

.. warning::
  Updating the rate will reset all values.

.. _pyarray_get:
.. py:function:: zfpy.array1.get(i)
.. py:function:: zfpy.array2.get(i, j)
.. py:function:: zfpy.array3.get(i, j, k)

  Multi-dimensional accessor for a single array element.

.. note::
  |zfpy| does not yet support multi-dimensional access via standard array 
  notation. Implementing this correctly requires supporting slicing which 
  will be coming alongside support for views.

.. _pyarray_set:
.. py:function:: zfpy.array1.set(i, val)
.. py:function:: zfpy.array2.set(i, j, val)
.. py:function:: zfpy.array3.set(i, j, k, val)

  Multi-dimensional mutator for a single array element.

.. _pyarray_flat_get:
.. py:function:: zfpy.array2.flat_get(i)
.. py:function:: zfpy.array3.flat_get(i)

  Flat indexed arrary accessor.

.. _pyarray_flat_set:
.. py:function:: zfpy.array2.flat_set(i, val)
.. py:function:: zfpy.array3.flat_set(i, val)

  Flat indexed arrary mutator.

.. _pyarray_subscript_op:
.. py:function:: zfpy.arrayANY[i]

  Flat indexed array accessor. Equivalent to the |zfp| compressed 
  array [] operator. Supports python array negative indexing.

.. _pyarray_compressed_size:
.. py:function:: zfpy.arrayANY.compressed_size()

  See :cpp:func:`array::compressed_size`.

.. _pyarray_compressed_data:
.. py:function:: zfpy.arrayANY.compressed_data()

  See :cpp:func:`array::compressed_data`. Note that this returns a 
  python bytes type.

Compression
-----------

.. py:function:: compress_numpy(arr, tolerance = -1, rate = -1, precision = -1, write_header = True)

  Compress NumPy array, *arr*, and return a compressed byte stream.  The
  non-expert :ref:`compression mode <modes>` is selected by setting one of
  *tolerance*, *rate*, or *precision*.  If none of these arguments is
  specified, then :ref:`reversible mode <mode-reversible>` is used.  By
  default, a header that encodes array shape and scalar type as well as
  compression parameters is prepended, which can be omitted by setting
  *write_header* to *False*.  If this function fails for any reason, an
  exception is thrown.

|zfpy| NumPy compression takes a NumPy array
(`ndarray <https://www.numpy.org/devdocs/reference/arrays.ndarray.html>`_)
populated with the data to be compressed. The array metadata (i.e.,
shape, strides, and scalar type) is used to automatically populate the
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

.. py:function:: decompress_numpy(compressed_data)

  Decompress a byte stream, *compressed_data*, produced by
  :py:func:`compress_numpy` (with header enabled) and return the
  decompressed NumPy array.  This function throws on exception upon error.

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

.. py:function:: _decompress(compressed_data, ztype, shape, out = None, tolerance = -1, rate = -1, precision = -1)

  Decompress a headerless compressed stream (if a header is present in
  the stream, it will be incorrectly interpreted as compressed data).
  *ztype* specifies the array scalar type while *shape* specifies the array
  dimensions; both must be known by the caller.  The compression mode is
  selected by specifying one (or none) of *tolerance*, *rate*, and
  *precision*, as in :py:func:`compress_numpy`, and also must be known
  by the caller.  If *out = None*, a new NumPy array is allocated.  Otherwise,
  *out* specifies the NumPy array or memory buffer to decompress into.
  Regardless, the decompressed NumPy array is returned unless an error occurs,
  in which case an exception is thrown.

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
