.. index::
   single: Codec
.. _codec:

Codec
-----

.. cpp:namespace:: zfp

|zfp| arrays are partitioned into independent blocks that are compressed
and decompressed using a *codec* (encoder/decoder).  This codec defaults
to the |zfp| compression scheme, but can in principle be any compression
scheme or number representation that represents *d*-dimensional blocks
of |4powd| values.  The :cpp:class:`zfp::array` and
:cpp:class:`zfp::const_array` classes take such a codec class as an
optional template parameter.

This section documents the API that prospective codecs must support to
interface with the |zfp| compressed-array classes.  Any one codec supports a
specific scalar type (e.g., :code:`float` or :code:`double`), denoted
:code:`Scalar` below, and data dimensionality (1D, 2D, 3D, or 4D).  If the
codec does not support a certain compression mode, it should throw an
:ref:`exception <exception>` when the user attempts to invoke that mode.
Codecs reside in the :code:`zfp::codec` namespace, e.g.,
:code:`zfp::codec::zfp3<Scalar>` is the default codec for 3D arrays.

As of |zfp| |cpprelease|, there is in addition to the default |zfp| codec
a "generic" codec that allows storing data in |zfp| arrays in "uncompressed"
form using any scalar type (specified as a template parameter).  This
"internal" scalar type may differ from the "external" scalar type exposed
to the user through the :cpp:class:`zfp::array` API.  For instance, the
internal type may be :code:`float` while the external type is :code:`double`,
which provides for 2:1 fixed-rate "compression" using IEEE 754 floating point.

.. cpp:namespace:: zfp::codec

.. cpp:class:: codec

  Fictitious class encapsulating the codec API.  This may be thought of as
  a base class for the classes below specialized on dimensionality.

----

.. cpp:class:: codec1
.. cpp:class:: codec2
.. cpp:class:: codec3
.. cpp:class:: codec4

  Fictitious classes encapsulating the codec API specialized for a given
  data dimensionality (1D, 2D, 3D, or 4D).

----

.. cpp:function:: codec& codec::operator=(const codec& c)

  Assignment operator.  Performs a deep copy.  This method is invoked when
  performing a :ref:`deep copy <array_copy>` of an array.

----

.. cpp:function:: size_t codec::buffer_size(const zfp_field* field) const

  Maximum buffer size needed to encode the *field* of given scalar type and
  dimensions (see :c:func:`zfp_stream_maximum_size`).  The size should be
  based on the current compression mode and parameters.  This method is
  called to determine how large a buffer to allocate and pass to
  :cpp:func:`codec::open`.

----

.. cpp:function:: void codec::open(void* data, size_t size)

  Open codec for (de)compression to/from buffer pointed to by *data* of
  *size* bytes.  The caller is responsible for allocating and deallocating
  this buffer, whose *size* is given by :cpp:func:`codec::buffer_size`.

----

.. cpp:function:: void codec::close()

  Close codec for (de)compression.

----

.. cpp:function:: zfp_mode codec::mode() const

  Currently selected :ref:`compression mode <modes>`.  See :c:enum:`zfp_mode`.

----

.. cpp:function:: double codec::rate() const

  Rate in compressed bits/value when :ref:`fixed-rate mode <mode-fixed-rate>`
  is selected.  See :c:func:`zfp_stream_rate`.

----

.. cpp:function:: uint codec::precision() const

  Precision in uncompressed bits/value when
  :ref:`fixed-precision mode <mode-fixed-precision>` is selected.  See
  :c:func:`zfp_stream_precision`.

----

.. cpp:function:: double codec::accuracy() const

  Accuracy as absolute error tolerance when
  :ref:`fixed-accuracy mode <mode-fixed-accuracy>` is selected.  See
  :c:func:`zfp_stream_accuracy`.

----

.. cpp:function:: void codec::params(uint* minbits, uint* maxbits, uint* maxprec, int* minexp) const

  Compression parameters for any compression mode.  These pointer parameters
  may be :code:`null` if only a subset of parameters is requested.
  See :c:func:`zfp_stream_params`.

----

.. cpp:function:: void codec::set_reversible()

  Enable :ref:`reversible mode <mode-reversible>`.

----

.. cpp:function:: double codec::set_rate(double rate, bool align)

  Set desired *rate* in number of compressed bits/value.  When *align* = true,
  blocks are word aligned, as needed for random access writes.  Return
  the closest rate supported.  See :c:func:`zfp_stream_set_rate`.

----

.. cpp:function:: uint codec::set_precision(uint precision)

  Set precision in number of uncompressed bits/value.  Return the actual
  precision selected.  See :c:func:`zfp_stream_set_precision`.

----

.. cpp:function:: double codec::set_accuracy(double tolerance)

  Set accuracy as absolute error tolerance.  Return the closest tolerance
  supported.  See :c:func:`zfp_stream_set_accuracy`.

----

.. cpp:function:: bool codec::set_params(uint minbits, uint maxbits, uint maxprec, int minexp)

  Set expert mode parameters.  Return :code:`true` on success.
  See :c:func:`zfp_stream_set_params`.

----

.. cpp:function:: bool codec::set_thread_safety(bool safety)

  Enable or disable thread safety.  This function is called whenever |zfp|
  is built with OpenMP support and when the number of mutable or immutable
  :ref:`private views <private_immutable_view>` of an array changes.  When
  two or more private views of an array are accessed by separate threads,
  multiple blocks may be compressed or decompressed simultaneously.  The
  codec then has to take care that there are no race conditions on the data
  structures (e.g., :c:type:`bitstream`) used for (de)compression.

----

.. cpp:function:: size_t codec::size_bytes(uint mask = ZFP_DATA_ALL) const

  Return storage size of components of codec data structure indicated by
  *mask*.  The mask is constructed via bitwise OR of
  :ref:`predefined constants <data-macros>`.

----

.. cpp:function:: static size_t codec::alignment()

  Memory alignment in number of bytes required by codec.

.. cpp:var:: static const zfp_type codec::type;

  :c:type:`Scalar type <zfp_type>` compressed by codec.

----

.. cpp:function:: size_t codec::encode_block(bitstream_offset offset, const Scalar* block) const

  Encode contiguous *block* of |4powd| scalars and store at specified bit
  *offset* within compressed-data buffer.  Return the number of bits of
  compressed storage for the block, excluding any necessary padding.  This
  method must flush any buffered compressed data without counting any padding
  (e.g., for byte alignment) in the compressed size (unless the codec requires
  alignment of the bit offsets).

----

.. cpp:function:: size_t codec::decode_block(bitstream_offset offset, Scalar* block) const

  Decode contiguous *block* of |4powd| scalars from specified bit *offset*
  within compressed-data buffer (see :cpp:func:`codec::encode_block`).
  Return number of bits of compressed data decoded, excluding any padding
  bits, i.e., the same value reported in encoding.

----

.. cpp:function:: size_t codec1::encode_block(bitstream_offset offset, uint shape, const Scalar* block) const
.. cpp:function:: size_t codec2::encode_block(bitstream_offset offset, uint shape, const Scalar* block) const
.. cpp:function:: size_t codec3::encode_block(bitstream_offset offset, uint shape, const Scalar* block) const
.. cpp:function:: size_t codec4::encode_block(bitstream_offset offset, uint shape, const Scalar* block) const

  Encode contiguous *block* of data of given *shape* and store at specified
  bit *offset* within compressed-data buffer.  Return the number of bits of
  compressed storage for the block (see also :cpp:func:`codec::encode_block`).

  The *shape* is a (2 |times| *d*)-bit encoding of the size of the
  *d*-dimensional block.  For each successive pair of bits *s* of *shape*,
  the block size in the corresponding dimension is *n* = 4 - *s*, where
  0 |leq| *s* |leq| 3.  Thus, *shape* = 0 implies a full block of |4powd|
  values.  The size of the fastest varying dimension is specified in the
  least significant bits of *shape*.

----

.. cpp:function:: size_t codec1::decode_block(bitstream_offset offset, uint shape, Scalar* block) const
.. cpp:function:: size_t codec2::decode_block(bitstream_offset offset, uint shape, Scalar* block) const
.. cpp:function:: size_t codec3::decode_block(bitstream_offset offset, uint shape, Scalar* block) const
.. cpp:function:: size_t codec4::decode_block(bitstream_offset offset, uint shape, Scalar* block) const

  Decode contiguous *block* of data of given *shape* from specified bit
  *offset* within compressed-data buffer (see also
  :cpp:func:`codec1::encode_block`).  Return number of bits of compressed
  data decoded, excluding any padding bits, i.e., the same value reported
  in encoding.

----

.. cpp:function:: size_t codec1::encode_block_strided(bitstream_offset offset, uint shape, const Scalar* p, ptrdiff_t sx) const
.. cpp:function:: size_t codec2::encode_block_strided(bitstream_offset offset, uint shape, const Scalar* p, ptrdiff_t sx, ptrdiff_t sy) const
.. cpp:function:: size_t codec3::encode_block_strided(bitstream_offset offset, uint shape, const Scalar* p, ptrdiff_t sx, ptrdiff_t sy, ptrdiff_t sz) const
.. cpp:function:: size_t codec4::encode_block_strided(bitstream_offset offset, uint shape, const Scalar* p, ptrdiff_t sx, ptrdiff_t sy, ptrdiff_t sz, ptrdiff_t sw) const

  Encode block of data stored at *p* with strides *sx*, *sy*, *sz*,
  and *sw*.  See :c:type:`zfp_field` for information on strided storage.
  The *shape*, *offset*, and return value are as in
  :cpp:func:`codec1::encode_block`.

----

.. cpp:function:: size_t codec1::decode_block_strided(bitstream_offset offset, uint shape, Scalar* p, ptrdiff_t sx) const
.. cpp:function:: size_t codec2::decode_block_strided(bitstream_offset offset, uint shape, Scalar* p, ptrdiff_t sx, ptrdiff_t sy) const
.. cpp:function:: size_t codec3::decode_block_strided(bitstream_offset offset, uint shape, Scalar* p, ptrdiff_t sx, ptrdiff_t sy, ptrdiff_t sz) const
.. cpp:function:: size_t codec4::decode_block_strided(bitstream_offset offset, uint shape, Scalar* p, ptrdiff_t sx, ptrdiff_t sy, ptrdiff_t sz, ptrdiff_t sw) const

  Decode block to strided storage pointed to by *p* with strides *sx*, *sy*,
  *sz*, and *sw*.  See :c:type:`zfp_field` for information on strided storage.
  The *shape*, *offset*, and return value are as in 
  :cpp:func:`codec1::decode_block`.
