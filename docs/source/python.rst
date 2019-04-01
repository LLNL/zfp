.. include:: defs.rst
.. _python:

Python
=======================

Dependencies
------------

Minimum Tested Versions:

* Python: Python 2.7 & Python 3.5
* Cython: 0.22
* Numpy: 1.8.0

You can install the necessary dependencies using ``pip`` and the zfp
``requirements.txt``::

  pip install -r $ZFP_ROOT/python/requirements.txt

Installation
------------

To build the python bindings, add ``-DBUILD_PYTHON=on`` to the cmake line. Cmake
will attempt to automatically detect the python installation to use.  If cmake
finds multiple python installations, it will use the newest one.  To specify a
specific python installation to use, set ``PYTHON_LIBRARY`` and
``PYTHON_INCLUDE_DIR`` in the cmake line. Putting it all together::

    cmake -DBUILD_PYTHON=on -DPYTHON_LIBRARY=/path/to/lib/libpython2.7.so -DPYTHON_INCLUDE_DIR=/path/to/include/python2.7 ..

Compression
-----------

.. py:function:: compress_numpy(arr, tolerance = -1, rate = -1, precision = -1, write_header=True)

Compression through the python bindings currently requires a numpy array
populated with the data to be compressed.  The numpy metadata (i.e., shape,
strides, and type) are used to automatically populate ``zfp_field`` structure.
By default, all that is required to be passed to the compression function is the
numpy array; this will result in a stream that includes a header and is
compressed with the ``reversible`` mode.  For example::

  import zfp
  import numpy as np

  my_array = np.arange(1, 20)
  compressed_data = zfp.compress_numpy(my_array)
  decompressed_array = zfp.decompress_numpy(compressed_data)

  # confirm lossless compression/decompression
  np.testing.assert_array_equal(my_array, decompressed_array)

Using the fixed-accuracy, fixed-rate, or fixed-precision modes simply requires
setting one of the tolerance, rate, or precision arguments, respectively. For example::

  compressed_data = zfp.compress_numpy(my_array, tolerance=1e-4)
  decompressed_array = zfp.decompress_numpy(compressed_data)

  # Note the change from "equal" to "allclose" due to the lossy compression
  np.testing.assert_allclose(my_array, decompressed_array, atol=1e-3)

Since numpy arrays are C-ordered by default and ``zfp_compress`` expects the
fastest changing stride to the first (i.e., Fortran-ordering),
``compress_numpy`` automatically flips the reverses the stride in order to
optimize the compression ratio for C-ordered numpy arrays. Since the
``decompress_numpy`` function also reverses the stride order, data both
compressed and decompressed with the python bindings should have the same shape
before and after.

.. note:: ``decompress_numpy`` requires a header to decompress properly, so do
   not use ``write_header=False`` if you intend to decompress the stream with
   the python bindings.

Decompression
-------------

.. py:function:: decompress_numpy(compressed_data)

``decompress_numpy`` consumes a compressed stream that includes a header and
produces a numpy array with metadata populated based on the contents of the
header.  Stride information is not stored in the zfp header, so the
``decompress_numpy`` function assumes that the array was compressed with the
fastest changing dimension first (typically referred to as Fortran-ordering).
The returned numpy array is in C-ordering (the default for numpy arrays), so the
shape of the returned array is reversed from that of the shape in the
compression header.  For example, if the header declares the array to be of
shape (2, 4, 8), then the returned numpy array will have a shape of (8, 4, 2).
Since the ``compress_numpy`` function also reverses the stride order, data both
compressed and decompressed with the python bindings should have the same shape
before and after.

.. note:: Decompressing a stream without a header requires using the
   internal ``_decompress`` python function (or the C API).

.. py:function:: _decompress(compressed_data, out=None, ztype=type_none, shape=None, strides=[0,0,0,0], tolerance = -1, rate = -1, precision = -1,)

.. warning:: ``_decompress`` is a function mainly for internal use within the
             zfp library.  It does allow decompression of streams without
             headers, but providing too small of an output bufffer or
             incorrectly specifying the shape or strides can result in
             segmentation faults.  Use with care.

Decompresses a compressed stream (with or without a header).  If a header is
present in the stream, all user-provided metadata is compared against the stream
header metadata, and an exception is thrown if they disagree.  If ``out`` is
specified, the data is decompressed into the ``out`` buffer.  ``out`` can be a
numpy array or a pointer to memory large enough to hold the decompressed data.
Regardless if ``out`` is provided or its type, ``_decompress`` always returns a
numpy array.  If ``out`` is not provided, the array is allocated for the user,
and if ``out`` is provided, then the returned numpy is just a pointer to or
wrapper around the user-supplied ``out``.  If ``strides`` is provided, it is
added to the ``zfp_field`` before decompression, but it is not added to the
output numpy array metadata.  ``strides`` should be a zfp-style strides (i.e.,
number of elements) rather than a numpy-style strides (i.e., number of bytes).
