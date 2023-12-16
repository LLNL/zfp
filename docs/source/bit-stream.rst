.. include:: defs.rst

.. _bs-api:

Bit Stream API
==============

|zfp| relies on low-level functions for bit stream I/O, e.g., for
reading/writing single bits or groups of bits.  |zfp|'s bit streams
support random access (with some caveats) and, optionally, strided
access.  The functions read from and write to main memory allocated
by the user.  Buffer overruns are for performance reasons not guarded
against.

From an implementation standpoint, bit streams are read from and written
to memory in increments of *words* of bits.  The constant power-of-two
word size is configured at :ref:`compile time <config>`, and is limited
to 8, 16, 32, or 64 bits.

The bit stream API is publicly exposed and may be used to write additional
information such as metadata into the |zfp| compressed stream and to
manipulate whole or partial bit streams.  Moreover, we envision releasing
the bit stream functions as a separate library in the future that may be
used, for example, in other compressors.

Stream readers and writers are synchronized by making corresponding calls.
For each write call, there is a corresponding read call.  This ensures
that reader and writer agree on the position within the stream and the
number of bits buffered, if any.  The API below reflects this duality.

A bit stream is either in read or write mode, or either, if rewound to
the beginning.  When in read mode, only read calls should be made,
and similarly for write mode.

.. _bs-strides:

Strided Streams
---------------

Bit streams may be strided by sequentially reading/writing a few words at
a time and then skipping over some user-specified number of words.  This
allows, for instance, |zfp| to interleave the first few bits of all
compressed blocks in order to support progressive access.  To enable
strided access, which does carry a small performance penalty, the
macro :c:macro:`BIT_STREAM_STRIDED` must be defined during compilation.

Strides are specified in terms of a *block size*---a power-of-two number
of contiguous words---and a *delta*, which specifies how many words to
advance the stream by to get to the next contiguous block.  These bit
stream blocks are entirely independent of the |4powd| blocks used for
compression in |zfp|.  Setting *delta* to zero ensures a non-strided,
sequential layout.

.. _bs-macros:

Macros
------

Two compile-time macros are used to influence the behavior:
:c:macro:`BIT_STREAM_WORD_TYPE` and :c:macro:`BIT_STREAM_STRIDED`.
These are documented in the :ref:`installation <installation>`
section.

.. _bs-types:

Types
-----

.. c:type:: bitstream_word

  Bits are buffered and read/written in units of words.  By default, the
  bit stream word type is 64 bits, but may be set to 8, 16, or 32 bits
  by setting the macro :c:macro:`BIT_STREAM_WORD_TYPE` to :c:type:`uint8`,
  :c:type:`uint16`, or :c:type:`uint32`, respectively.  Larger words
  tend to give higher throughput, while 8-bit words are needed to ensure
  endian independence (see FAQ :ref:`#11 <q-portability>`).

.. note::
  To avoid potential name clashes, this type was renamed in
  |zfp| |64bitrelease| from the shorter and more ambiguous type name
  :code:`word`.

----

.. c:type:: bitstream_offset

  Type holding the offset, measured in number of bits, into the bit stream
  where the next bit will be read or written.  This type allows referencing
  bits in streams at least 2\ :sup:`64` bits long.  Note that it is possible
  that :code:`sizeof(bitstream_offset) > sizeof(size_t)` since a stream may
  be as long as `sizeof(size_t) * CHAR_BIT` bits.

----

.. c:type:: bitstream_size

  Alias for :c:type:`bitstream_offset` that signifies the bit length of a
  stream or substream rather than an offset into it.

----

.. c:type:: bitstream_count

  Type sufficient to count the number of bits read or written in functions
  like :c:func:`stream_read_bits` and :c:func:`stream_write_bits`.
  :code:`sizeof(bitstream_count) <= sizeof(bitstream_size)`.

----

.. c:type:: bitstream

  The bit stream struct maintains all the state associated with a bit
  stream.  This struct is passed to all bit stream functions.  Its members
  should not be accessed directly.
  ::

    struct bitstream {
      bitstream_count bits;  // number of buffered bits (0 <= bits < word size)
      bitstream_word buffer; // incoming/outgoing bits (buffer < 2^bits)
      bitstream_word* ptr;   // pointer to next word to be read/written
      bitstream_word* begin; // beginning of stream
      bitstream_word* end;   // end of stream (not enforced)
      size_t mask;           // one less the block size in number of words (if BIT_STREAM_STRIDED)
      ptrdiff_t delta;       // number of words between consecutive blocks (if BIT_STREAM_STRIDED)
    };

.. _bs-data:

Constants
---------

.. c:var:: const size_t stream_word_bits

  The number of bits in a word.  The size of a flushed bit stream will be
  a multiple of this number of bits.  See :c:macro:`BIT_STREAM_WORD_TYPE`
  and :c:func:`stream_alignment`.

.. _bs-functions:

Functions
---------

.. c:function:: bitstream* stream_open(void* buffer, size_t bytes)

  Allocate a :c:type:`bitstream` struct and associate it with the memory
  buffer allocated by the caller.

----

.. c:function:: void stream_close(bitstream* stream)

  Close the bit stream and deallocate *stream*.

----

.. c:function:: bitstream* stream_clone(const bitstream* stream)

  Create a copy of *stream* that points to the same memory buffer.

----

.. c:function:: bitstream_count stream_alignment()

  Word size in bits.  This is a functional form of the constant
  :c:var:`stream_word_bits` and returns the same value.
  Available since |zfp| |crpirelease|.

----

.. c:function:: void* stream_data(const bitstream* stream)

  Return pointer to the beginning of bit stream *stream*.

----

.. c:function:: size_t stream_size(const bitstream* stream)

  Return position of stream pointer in number of bytes, which equals the
  end of stream if no seeks have been made.  Note that additional bits
  may be buffered and not reported unless the stream has been flushed.

----

.. c:function:: size_t stream_capacity(const bitstream* stream)

  Return byte size of memory buffer associated with *stream* specified
  in :c:func:`stream_open`.

----

.. c:function:: uint stream_read_bit(bitstream* stream)

  Read a single bit from *stream*.

----

.. c:function:: uint stream_write_bit(bitstream* stream, uint bit)

  Write single *bit* to *stream*.  *bit* must be one of 0 or 1.
  The value of *bit* is returned.

----

.. c:function:: uint64 stream_read_bits(bitstream* stream, bitstream_count n)

  Read and return 0 |leq| *n* |leq| 64 bits from *stream*.

----

.. c:function:: uint64 stream_write_bits(bitstream* stream, uint64 value, bitstream_count n)

  Write 0 |leq| *n* |leq| 64 low bits of *value* to *stream*.  Return any
  remaining bits from *value*, i.e., *value* >> *n*.

----

.. c:function:: bitstream_offset stream_rtell(const bitstream* stream)

  Return bit offset to next bit to be read.

----

.. c:function:: bitstream_offset stream_wtell(const bitstream* stream)

  Return bit offset to next bit to be written.

----

.. c:function:: void stream_rewind(bitstream* stream)

  Rewind stream to beginning of memory buffer.  Following this call, the
  stream may either be read or written.

----

.. c:function:: void stream_rseek(bitstream* stream, bitstream_offset offset)

  Position stream for reading at given bit offset.  This places the
  stream in read mode.

----

.. c:function:: void stream_wseek(bitstream* stream, bitstream_offset offset)

  Position stream for writing at given bit offset.  This places the
  stream in write mode.

----

.. c:function:: void stream_skip(bitstream* stream, bitstream_count n)

  Skip over the next *n* bits, i.e., without reading them.

----

.. c:function:: void stream_pad(bitstream* stream, bitstream_count n)

  Append *n* zero-bits to *stream*.

----

.. c:function:: bitstream_count stream_align(bitstream* stream)

  Align stream on next word boundary by skipping bits, i.e., without reading
  them.  No skipping is done if the stream is already word aligned.  Return
  the number of skipped bits, if any.

----

.. c:function:: bitstream_count stream_flush(bitstream* stream)

  Write out any remaining buffered bits.  When one or more bits are
  buffered, append zero-bits to the stream to align it on a word boundary.
  Return the number of bits of padding, if any.

----

.. c:function:: void stream_copy(bitstream* dst, bitstream* src, bitstream_size n)

  Copy *n* bits from *src* to *dst*, advancing both bit streams.

----

.. c:function:: size_t stream_stride_block(const bitstream* stream)

  Return stream block size in number of words.  The block size is always
  one word unless strided streams are enabled.  See :ref:`bs-strides`
  for more information.

----

.. c:function:: ptrdiff_t stream_stride_delta(const bitstream* stream)

  Return stream delta in number of words between blocks.  See
  :ref:`bs-strides` for more information.

----

.. c:function:: int stream_set_stride(bitstream* stream, size_t block, ptrdiff_t delta)

  Set block size, *block*, in number of words and spacing, *delta*, in number
  of blocks for :ref:`strided access <bs-strides>`.  Return nonzero upon
  success.  Requires :c:macro:`BIT_STREAM_STRIDED`.
