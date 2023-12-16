.. include:: defs.rst

.. _ll-api:

Low-Level C API
===============

The |libzfp| low-level C API provides functionality for compressing individual
*d*-dimensional blocks of up to |4powd| values.  If a block is not complete,
i.e., contains fewer than |4powd| values, then |zfp|'s partial
block support should be favored over padding the block with, say, zeros
or other fill values.  The blocks (de)compressed need not be contiguous
and can be gathered from or scattered to a larger array by setting
appropriate strides.  As of |zfp| |cpprelease|, templated C++ wrappers
are also available to simplify calling the low-level API from C++.
The C API is declared in :file:`zfp.h`; the C++ wrappers are found in
:file:`zfp.hpp`.

.. note::
  Because the unit of parallel work in |zfp| is a *block*, and because the
  low-level API operates on individual blocks, this API supports only the
  the serial :ref:`execution policy <exec-policies>`.  Any other execution
  policy set in :c:type:`zfp_stream` is silently ignored.  For parallel
  execution, see the :ref:`high-level API <hl-api>`.

The following topics are available:

* :ref:`ll-stream`

* :ref:`ll-encoder`

  * :ref:`ll-1d-encoder`
  * :ref:`ll-2d-encoder`
  * :ref:`ll-3d-encoder`
  * :ref:`ll-4d-encoder`

* :ref:`ll-decoder`

  * :ref:`ll-1d-decoder`
  * :ref:`ll-2d-decoder`
  * :ref:`ll-3d-decoder`
  * :ref:`ll-4d-decoder`

* :ref:`ll-utilities`

* :ref:`ll-cpp-wrappers`

.. _ll-stream:

Stream Manipulation
-------------------

.. c:function:: size_t zfp_stream_flush(zfp_stream* stream)

  Flush bit stream to write out any buffered bits.  This function must be
  must be called after the last encode call.  The bit stream is aligned on
  a stream word boundary following this call.  The number of zero-bits
  written, if any, is returned.

----

.. c:function:: size_t zfp_stream_align(zfp_stream* stream)

  Align bit stream on next word boundary.  This function is analogous to
  :c:func:`zfp_stream_flush`, but for decoding.  That is, wherever the
  encoder flushes the stream, the decoder should align it to ensure
  synchronization between encoder and decoder.  The number of bits skipped,
  if any, is returned.

.. _ll-encoder:

Encoder
-------

A function is available for encoding whole or partial blocks of each scalar
type and dimensionality.  These functions return the number of bits of
compressed storage for the block being encoded, or zero upon failure.

.. _ll-1d-encoder:

1D Data
^^^^^^^

.. c:function:: size_t zfp_encode_block_int32_1(zfp_stream* stream, const int32* block)
.. c:function:: size_t zfp_encode_block_int64_1(zfp_stream* stream, const int64* block)
.. c:function:: size_t zfp_encode_block_float_1(zfp_stream* stream, const float* block)
.. c:function:: size_t zfp_encode_block_double_1(zfp_stream* stream, const double* block)

  Encode 1D contiguous block of 4 values.

----

.. c:function:: size_t zfp_encode_block_strided_int32_1(zfp_stream* stream, const int32* p, ptrdiff_t sx)
.. c:function:: size_t zfp_encode_block_strided_int64_1(zfp_stream* stream, const int64* p, ptrdiff_t sx)
.. c:function:: size_t zfp_encode_block_strided_float_1(zfp_stream* stream, const float* p, ptrdiff_t sx)
.. c:function:: size_t zfp_encode_block_strided_double_1(zfp_stream* stream, const double* p, ptrdiff_t sx)

  Encode 1D complete block from strided array with stride *sx*.

----

.. c:function:: size_t zfp_encode_partial_block_strided_int32_1(zfp_stream* stream, const int32* p, size_t nx, ptrdiff_t sx)
.. c:function:: size_t zfp_encode_partial_block_strided_int64_1(zfp_stream* stream, const int64* p, size_t nx, ptrdiff_t sx)
.. c:function:: size_t zfp_encode_partial_block_strided_float_1(zfp_stream* stream, const float* p, size_t nx, ptrdiff_t sx)
.. c:function:: size_t zfp_encode_partial_block_strided_double_1(zfp_stream* stream, const double* p, size_t nx, ptrdiff_t sx)

  Encode 1D partial block of size *nx* from strided array with stride *sx*.

.. _ll-2d-encoder:

2D Data
^^^^^^^

.. c:function:: size_t zfp_encode_block_int32_2(zfp_stream* stream, const int32* block)
.. c:function:: size_t zfp_encode_block_int64_2(zfp_stream* stream, const int64* block)
.. c:function:: size_t zfp_encode_block_float_2(zfp_stream* stream, const float* block)
.. c:function:: size_t zfp_encode_block_double_2(zfp_stream* stream, const double* block)

  Encode 2D contiguous block of |4by4| values.

----

.. c:function:: size_t zfp_encode_block_strided_int32_2(zfp_stream* stream, const int32* p, ptrdiff_t sx, ptrdiff_t sy)
.. c:function:: size_t zfp_encode_block_strided_int64_2(zfp_stream* stream, const int64* p, ptrdiff_t sx, ptrdiff_t sy)
.. c:function:: size_t zfp_encode_block_strided_float_2(zfp_stream* stream, const float* p, ptrdiff_t sx, ptrdiff_t sy)
.. c:function:: size_t zfp_encode_block_strided_double_2(zfp_stream* stream, const double* p, ptrdiff_t sx, ptrdiff_t sy)

  Encode 2D complete block from strided array with strides *sx* and *sy*.

----

.. c:function:: size_t zfp_encode_partial_block_strided_int32_2(zfp_stream* stream, const int32* p, size_t nx, size_t ny, ptrdiff_t sx, ptrdiff_t sy)
.. c:function:: size_t zfp_encode_partial_block_strided_int64_2(zfp_stream* stream, const int64* p, size_t nx, size_t ny, ptrdiff_t sx, ptrdiff_t sy)
.. c:function:: size_t zfp_encode_partial_block_strided_float_2(zfp_stream* stream, const float* p, size_t nx, size_t ny, ptrdiff_t sx, ptrdiff_t sy)
.. c:function:: size_t zfp_encode_partial_block_strided_double_2(zfp_stream* stream, const double* p, size_t nx, size_t ny, ptrdiff_t sx, ptrdiff_t sy)

  Encode 2D partial block of size *nx* |times| *ny* from strided array with
  strides *sx* and *sy*.

.. _ll-3d-encoder:

3D Data
^^^^^^^

.. c:function:: size_t zfp_encode_block_int32_3(zfp_stream* stream, const int32* block)
.. c:function:: size_t zfp_encode_block_int64_3(zfp_stream* stream, const int64* block)
.. c:function:: size_t zfp_encode_block_float_3(zfp_stream* stream, const float* block)
.. c:function:: size_t zfp_encode_block_double_3(zfp_stream* stream, const double* block)

  Encode 3D contiguous block of |4by4by4| values.

----

.. c:function:: size_t zfp_encode_block_strided_int32_3(zfp_stream* stream, const int32* p, ptrdiff_t sx, ptrdiff_t sy, ptrdiff_t sz)
.. c:function:: size_t zfp_encode_block_strided_int64_3(zfp_stream* stream, const int64* p, ptrdiff_t sx, ptrdiff_t sy, ptrdiff_t sz)
.. c:function:: size_t zfp_encode_block_strided_float_3(zfp_stream* stream, const float* p, ptrdiff_t sx, ptrdiff_t sy, ptrdiff_t sz)
.. c:function:: size_t zfp_encode_block_strided_double_3(zfp_stream* stream, const double* p, ptrdiff_t sx, ptrdiff_t sy, ptrdiff_t sz)

  Encode 3D complete block from strided array with strides *sx*, *sy*, and
  *sz*.

----

.. c:function:: size_t zfp_encode_partial_block_strided_int32_3(zfp_stream* stream, const int32* p, size_t nx, size_t ny, size_t nz, ptrdiff_t sx, ptrdiff_t sy, ptrdiff_t sz)
.. c:function:: size_t zfp_encode_partial_block_strided_int64_3(zfp_stream* stream, const int64* p, size_t nx, size_t ny, size_t nz, ptrdiff_t sx, ptrdiff_t sy, ptrdiff_t sz)
.. c:function:: size_t zfp_encode_partial_block_strided_float_3(zfp_stream* stream, const float* p, size_t nx, size_t ny, size_t nz, ptrdiff_t sx, ptrdiff_t sy, ptrdiff_t sz)
.. c:function:: size_t zfp_encode_partial_block_strided_double_3(zfp_stream* stream, const double* p, size_t nx, size_t ny, size_t nz, ptrdiff_t sx, ptrdiff_t sy, ptrdiff_t sz)

  Encode 3D partial block of size *nx* |times| *ny* |times| *nz* from strided
  array with strides *sx*, *sy*, and *sz*.

.. _ll-4d-encoder:

4D Data
^^^^^^^

.. c:function:: size_t zfp_encode_block_int32_4(zfp_stream* stream, const int32* block)
.. c:function:: size_t zfp_encode_block_int64_4(zfp_stream* stream, const int64* block)
.. c:function:: size_t zfp_encode_block_float_4(zfp_stream* stream, const float* block)
.. c:function:: size_t zfp_encode_block_double_4(zfp_stream* stream, const double* block)

  Encode 4D contiguous block of |4by4by4by4| values.

----

.. c:function:: size_t zfp_encode_block_strided_int32_4(zfp_stream* stream, const int32* p, ptrdiff_t sx, ptrdiff_t sy, ptrdiff_t sz, ptrdiff_t sw)
.. c:function:: size_t zfp_encode_block_strided_int64_4(zfp_stream* stream, const int64* p, ptrdiff_t sx, ptrdiff_t sy, ptrdiff_t sz, ptrdiff_t sw)
.. c:function:: size_t zfp_encode_block_strided_float_4(zfp_stream* stream, const float* p, ptrdiff_t sx, ptrdiff_t sy, ptrdiff_t sz, ptrdiff_t sw)
.. c:function:: size_t zfp_encode_block_strided_double_4(zfp_stream* stream, const double* p, ptrdiff_t sx, ptrdiff_t sy, ptrdiff_t sz, ptrdiff_t sw)

  Encode 4D complete block from strided array with strides *sx*, *sy*, *sz*, and
  *sw*.

----

.. c:function:: size_t zfp_encode_partial_block_strided_int32_4(zfp_stream* stream, const int32* p, size_t nx, size_t ny, size_t nz, size_t nw, ptrdiff_t sx, ptrdiff_t sy, ptrdiff_t sz, ptrdiff_t sw)
.. c:function:: size_t zfp_encode_partial_block_strided_int64_4(zfp_stream* stream, const int64* p, size_t nx, size_t ny, size_t nz, size_t nw, ptrdiff_t sx, ptrdiff_t sy, ptrdiff_t sz, ptrdiff_t sw)
.. c:function:: size_t zfp_encode_partial_block_strided_float_4(zfp_stream* stream, const float* p, size_t nx, size_t ny, size_t nz, size_t nw, ptrdiff_t sx, ptrdiff_t sy, ptrdiff_t sz, ptrdiff_t sw)
.. c:function:: size_t zfp_encode_partial_block_strided_double_4(zfp_stream* stream, const double* p, size_t nx, size_t ny, size_t nz, size_t nw, ptrdiff_t sx, ptrdiff_t sy, ptrdiff_t sz, ptrdiff_t sw)

  Encode 4D partial block of size *nx* |times| *ny* |times| *nz* |times| *nw*
  from strided array with strides *sx*, *sy*, *sz*, and *sw*.

.. _ll-decoder:

Decoder
-------

Each function below decompresses a single block and returns the number of bits
of compressed storage consumed.  See corresponding encoder functions above for
further details.

.. _ll-1d-decoder:

1D Data
^^^^^^^

.. c:function:: size_t zfp_decode_block_int32_1(zfp_stream* stream, int32* block)
.. c:function:: size_t zfp_decode_block_int64_1(zfp_stream* stream, int64* block)
.. c:function:: size_t zfp_decode_block_float_1(zfp_stream* stream, float* block)
.. c:function:: size_t zfp_decode_block_double_1(zfp_stream* stream, double* block)

  Decode 1D contiguous block of 4 values.

----

.. c:function:: size_t zfp_decode_block_strided_int32_1(zfp_stream* stream, int32* p, ptrdiff_t sx)
.. c:function:: size_t zfp_decode_block_strided_int64_1(zfp_stream* stream, int64* p, ptrdiff_t sx)
.. c:function:: size_t zfp_decode_block_strided_float_1(zfp_stream* stream, float* p, ptrdiff_t sx)
.. c:function:: size_t zfp_decode_block_strided_double_1(zfp_stream* stream, double* p, ptrdiff_t sx)

  Decode 1D complete block to strided array with stride *sx*.

----

.. c:function:: size_t zfp_decode_partial_block_strided_int32_1(zfp_stream* stream, int32* p, size_t nx, ptrdiff_t sx)
.. c:function:: size_t zfp_decode_partial_block_strided_int64_1(zfp_stream* stream, int64* p, size_t nx, ptrdiff_t sx)
.. c:function:: size_t zfp_decode_partial_block_strided_float_1(zfp_stream* stream, float* p, size_t nx, ptrdiff_t sx)
.. c:function:: size_t zfp_decode_partial_block_strided_double_1(zfp_stream* stream, double* p, size_t nx, ptrdiff_t sx)

  Decode 1D partial block of size *nx* to strided array with stride *sx*.

.. _ll-2d-decoder:

2D Data
^^^^^^^

.. c:function:: size_t zfp_decode_block_int32_2(zfp_stream* stream, int32* block)
.. c:function:: size_t zfp_decode_block_int64_2(zfp_stream* stream, int64* block)
.. c:function:: size_t zfp_decode_block_float_2(zfp_stream* stream, float* block)
.. c:function:: size_t zfp_decode_block_double_2(zfp_stream* stream, double* block)

  Decode 2D contiguous block of |4by4| values.

----

.. c:function:: size_t zfp_decode_block_strided_int32_2(zfp_stream* stream, int32* p, ptrdiff_t sx, ptrdiff_t sy)
.. c:function:: size_t zfp_decode_block_strided_int64_2(zfp_stream* stream, int64* p, ptrdiff_t sx, ptrdiff_t sy)
.. c:function:: size_t zfp_decode_block_strided_float_2(zfp_stream* stream, float* p, ptrdiff_t sx, ptrdiff_t sy)
.. c:function:: size_t zfp_decode_block_strided_double_2(zfp_stream* stream, double* p, ptrdiff_t sx, ptrdiff_t sy)

  Decode 2D complete block to strided array with strides *sx* and *sy*.

----

.. c:function:: size_t zfp_decode_partial_block_strided_int32_2(zfp_stream* stream, int32* p, size_t nx, size_t ny, ptrdiff_t sx, ptrdiff_t sy)
.. c:function:: size_t zfp_decode_partial_block_strided_int64_2(zfp_stream* stream, int64* p, size_t nx, size_t ny, ptrdiff_t sx, ptrdiff_t sy)
.. c:function:: size_t zfp_decode_partial_block_strided_float_2(zfp_stream* stream, float* p, size_t nx, size_t ny, ptrdiff_t sx, ptrdiff_t sy)
.. c:function:: size_t zfp_decode_partial_block_strided_double_2(zfp_stream* stream, double* p, size_t nx, size_t ny, ptrdiff_t sx, ptrdiff_t sy)

  Decode 2D partial block of size *nx* |times| *ny* to strided array with
  strides *sx* and *sy*.

.. _ll-3d-decoder:

3D Data
^^^^^^^

.. c:function:: size_t zfp_decode_block_int32_3(zfp_stream* stream, int32* block)
.. c:function:: size_t zfp_decode_block_int64_3(zfp_stream* stream, int64* block)
.. c:function:: size_t zfp_decode_block_float_3(zfp_stream* stream, float* block)
.. c:function:: size_t zfp_decode_block_double_3(zfp_stream* stream, double* block)

  Decode 3D contiguous block of |4by4by4| values.

----

.. c:function:: size_t zfp_decode_block_strided_int32_3(zfp_stream* stream, int32* p, ptrdiff_t sx, ptrdiff_t sy, ptrdiff_t sz)
.. c:function:: size_t zfp_decode_block_strided_int64_3(zfp_stream* stream, int64* p, ptrdiff_t sx, ptrdiff_t sy, ptrdiff_t sz)
.. c:function:: size_t zfp_decode_block_strided_float_3(zfp_stream* stream, float* p, ptrdiff_t sx, ptrdiff_t sy, ptrdiff_t sz)
.. c:function:: size_t zfp_decode_block_strided_double_3(zfp_stream* stream, double* p, ptrdiff_t sx, ptrdiff_t sy, ptrdiff_t sz)

  Decode 3D complete block to strided array with strides *sx*, *sy*, and *sz*.

----

.. c:function:: size_t zfp_decode_partial_block_strided_int32_3(zfp_stream* stream, int32* p, size_t nx, size_t ny, size_t nz, ptrdiff_t sx, ptrdiff_t sy, ptrdiff_t sz)
.. c:function:: size_t zfp_decode_partial_block_strided_int64_3(zfp_stream* stream, int64* p, size_t nx, size_t ny, size_t nz, ptrdiff_t sx, ptrdiff_t sy, ptrdiff_t sz)
.. c:function:: size_t zfp_decode_partial_block_strided_float_3(zfp_stream* stream, float* p, size_t nx, size_t ny, size_t nz, ptrdiff_t sx, ptrdiff_t sy, ptrdiff_t sz)
.. c:function:: size_t zfp_decode_partial_block_strided_double_3(zfp_stream* stream, double* p, size_t nx, size_t ny, size_t nz, ptrdiff_t sx, ptrdiff_t sy, ptrdiff_t sz)

  Decode 3D partial block of size *nx* |times| *ny* |times| *nz* to strided
  array with strides *sx*, *sy*, and *sz*.

.. _ll-4d-decoder:

4D Data
^^^^^^^

.. c:function:: size_t zfp_decode_block_int32_4(zfp_stream* stream, int32* block)
.. c:function:: size_t zfp_decode_block_int64_4(zfp_stream* stream, int64* block)
.. c:function:: size_t zfp_decode_block_float_4(zfp_stream* stream, float* block)
.. c:function:: size_t zfp_decode_block_double_4(zfp_stream* stream, double* block)

  Decode 4D contiguous block of |4by4by4by4| values.

----

.. c:function:: size_t zfp_decode_block_strided_int32_4(zfp_stream* stream, int32* p, ptrdiff_t sx, ptrdiff_t sy, ptrdiff_t sz, ptrdiff_t sw)
.. c:function:: size_t zfp_decode_block_strided_int64_4(zfp_stream* stream, int64* p, ptrdiff_t sx, ptrdiff_t sy, ptrdiff_t sz, ptrdiff_t sw)
.. c:function:: size_t zfp_decode_block_strided_float_4(zfp_stream* stream, float* p, ptrdiff_t sx, ptrdiff_t sy, ptrdiff_t sz, ptrdiff_t sw)
.. c:function:: size_t zfp_decode_block_strided_double_4(zfp_stream* stream, double* p, ptrdiff_t sx, ptrdiff_t sy, ptrdiff_t sz, ptrdiff_t sw)

  Decode 4D complete block to strided array with strides *sx*, *sy*, *sz*, and *sw*.

----

.. c:function:: size_t zfp_decode_partial_block_strided_int32_4(zfp_stream* stream, int32* p, size_t nx, size_t ny, size_t nz, size_t nw, ptrdiff_t sx, ptrdiff_t sy, ptrdiff_t sz, ptrdiff_t sw)
.. c:function:: size_t zfp_decode_partial_block_strided_int64_4(zfp_stream* stream, int64* p, size_t nx, size_t ny, size_t nz, size_t nw, ptrdiff_t sx, ptrdiff_t sy, ptrdiff_t sz, ptrdiff_t sw)
.. c:function:: size_t zfp_decode_partial_block_strided_float_4(zfp_stream* stream, float* p, size_t nx, size_t ny, size_t nz, size_t nw, ptrdiff_t sx, ptrdiff_t sy, ptrdiff_t sz, ptrdiff_t sw)
.. c:function:: size_t zfp_decode_partial_block_strided_double_4(zfp_stream* stream, double* p, size_t nx, size_t ny, size_t nz, size_t nw, ptrdiff_t sx, ptrdiff_t sy, ptrdiff_t sz, ptrdiff_t sw)

  Decode 4D partial block of size *nx* |times| *ny* |times| *nz* |times| *nw*
  to strided array with strides *sx*, *sy*, *sz*, and *sw*.

.. _ll-utilities:

Utility Functions
-----------------

These functions convert 8- and 16-bit signed and unsigned integer data to
(by promoting) and from (by demoting) 32-bit integers that can be
(de)compressed by |zfp|'s :code:`int32` functions.  These conversion functions
are preferred over simple casting since they eliminate the redundant leading
zeros that would otherwise have to be compressed, and they apply the
appropriate bias for unsigned integer data.

----

.. c:function:: void zfp_promote_int8_to_int32(int32* oblock, const int8* iblock, uint dims)
.. c:function:: void zfp_promote_uint8_to_int32(int32* oblock, const uint8* iblock, uint dims)
.. c:function:: void zfp_promote_int16_to_int32(int32* oblock, const int16* iblock, uint dims)
.. c:function:: void zfp_promote_uint16_to_int32(int32* oblock, const uint16* iblock, uint dims)

  Convert *dims*-dimensional contiguous block to 32-bit integer type.
  Use *dims* = 0 to promote a single value.

----

.. c:function:: void zfp_demote_int32_to_int8(int8* oblock, const int32* iblock, uint dims)
.. c:function:: void zfp_demote_int32_to_uint8(uint8* oblock, const int32* iblock, uint dims)
.. c:function:: void zfp_demote_int32_to_int16(int16* oblock, const int32* iblock, uint dims)
.. c:function:: void zfp_demote_int32_to_uint16(uint16* oblock, const int32* iblock, uint dims)

  Convert *dims*-dimensional contiguous block from 32-bit integer type.
  Use *dims* = 0 to demote a single value.

.. _ll-cpp-wrappers:

C++ Wrappers
------------

.. cpp:namespace:: zfp

To facilitate calling the low-level API from C++, a number of wrappers are
available (as of |zfp| |cpprelease|) that are templated on scalar type and
dimensionality.  Each function of the form :code:`zfp_function_type_dims`,
where *type* denotes scalar type and *dims* denotes dimensionality, has a
corresponding C++ wrapper :code:`zfp::function<type, dims>`.  For example,
the C function :c:func:`zfp_encode_block_float_2` has a C++ wrapper
:cpp:func:`zfp::encode_block\<float, 2>`.  Often *dims* can be inferred from
the parameters of overloaded functions, in which case it is omitted as
template parameter.  The C++ wrappers are defined in :file:`zfp.hpp`.

Encoder
^^^^^^^

.. cpp:function:: template<typename Scalar, uint dims> size_t encode_block(zfp_stream* stream, const Scalar* block)

  Encode contiguous block of dimensionality *dims*.

----

.. cpp:function:: template<typename Scalar> size_t encode_block_strided(zfp_stream* stream, const Scalar* p, ptrdiff_t sx)
.. cpp:function:: template<typename Scalar> size_t encode_block_strided(zfp_stream* stream, const Scalar* p, ptrdiff_t sx, ptrdiff_t sy)
.. cpp:function:: template<typename Scalar> size_t encode_block_strided(zfp_stream* stream, const Scalar* p, ptrdiff_t sx, ptrdiff_t sy, ptrdiff_t sz)
.. cpp:function:: template<typename Scalar> size_t encode_block_strided(zfp_stream* stream, const Scalar* p, ptrdiff_t sx, ptrdiff_t sy, ptrdiff_t sz, ptrdiff_t sw)

  Encode complete block from strided array with strides *sx*, *sy*, *sz*, and
  *sw*.

----

.. cpp:function:: template<typename Scalar> size_t encode_partial_block_strided(zfp_stream* stream, const Scalar* p, size_t nx, ptrdiff_t sx)
.. cpp:function:: template<typename Scalar> size_t encode_partial_block_strided(zfp_stream* stream, const Scalar* p, size_t nx, size_t ny, ptrdiff_t sx, ptrdiff_t sy)
.. cpp:function:: template<typename Scalar> size_t encode_partial_block_strided(zfp_stream* stream, const Scalar* p, size_t nx, size_t ny, size_t nz, ptrdiff_t sx, ptrdiff_t sy, ptrdiff_t sz)
.. cpp:function:: template<typename Scalar> size_t encode_partial_block_strided(zfp_stream* stream, const Scalar* p, size_t nx, size_t ny, size_t nz, size_t nw, ptrdiff_t sx, ptrdiff_t sy, ptrdiff_t sz, ptrdiff_t sw)

  Encode partial block of size *nx* |times| *ny* |times| *nz* |times| *nw*
  from strided array with strides *sx*, *sy*, *sz*, and *sw*.

Decoder
^^^^^^^

.. cpp:function:: template<typename Scalar, uint dims> size_t decode_block(zfp_stream* stream, Scalar* block)

  Decode contiguous block of dimensionality *dims*.

----

.. cpp:function:: template<typename Scalar> size_t decode_block_strided(zfp_stream* stream, Scalar* p, ptrdiff_t sx)
.. cpp:function:: template<typename Scalar> size_t decode_block_strided(zfp_stream* stream, Scalar* p, ptrdiff_t sx, ptrdiff_t sy)
.. cpp:function:: template<typename Scalar> size_t decode_block_strided(zfp_stream* stream, Scalar* p, ptrdiff_t sx, ptrdiff_t sy, ptrdiff_t sz)
.. cpp:function:: template<typename Scalar> size_t decode_block_strided(zfp_stream* stream, Scalar* p, ptrdiff_t sx, ptrdiff_t sy, ptrdiff_t sz, ptrdiff_t sw)

  Decode complete block to strided array with strides *sx*, *sy*, *sz*, and
  *sw*.

----

.. cpp:function:: template<typename Scalar> size_t decode_partial_block_strided(zfp_stream* stream, Scalar* p, size_t nx, ptrdiff_t sx)
.. cpp:function:: template<typename Scalar> size_t decode_partial_block_strided(zfp_stream* stream, Scalar* p, size_t nx, size_t ny, ptrdiff_t sx, ptrdiff_t sy)
.. cpp:function:: template<typename Scalar> size_t decode_partial_block_strided(zfp_stream* stream, Scalar* p, size_t nx, size_t ny, size_t nz, ptrdiff_t sx, ptrdiff_t sy, ptrdiff_t sz)
.. cpp:function:: template<typename Scalar> size_t decode_partial_block_strided(zfp_stream* stream, Scalar* p, size_t nx, size_t ny, size_t nz, size_t nw, ptrdiff_t sx, ptrdiff_t sy, ptrdiff_t sz, ptrdiff_t sw)

  Decode partial block of size *nx* |times| *ny* |times| *nz* |times| *nw* to
  strided array with strides *sx*, *sy*, *sz*, and *sw*.
