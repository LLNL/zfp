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

.. c:type:: word

  Bits are buffered and read/written in units of words.  By default, the
  bit stream word type is 64 bits, but may be set to 8, 16, or 32 bits
  by setting the macro :c:macro:`BIT_STREAM_WORD_TYPE` to :c:type:`uint8`,
  :c:type:`uint16`, or :c:type:`uint32`, respectively.  Larger words
  tend to give higher throughput, while 8-bit words are needed to ensure
  endian independence (see FAQ :ref:`#11 <q-portability>`).

.. c:type:: bitstream

  The bit stream struct maintains all the state associated with a bit
  stream.  This struct is passed to all bit stream functions.  Its members
  should not be accessed directly.
  ::

    struct bitstream {
      uint bits;       // number of buffered bits (0 <= bits < word size)
      word buffer;     // buffer for incoming/outgoing bits (buffer < 2^bits)
      word* ptr;       // pointer to next word to be read/written
      word* begin;     // beginning of stream
      word* end;       // end of stream (currently unused)
      size_t mask;     // one less the block size in number of words (if BIT_STREAM_STRIDED)
      ptrdiff_t delta; // number of words between consecutive blocks (if BIT_STREAM_STRIDED)
    };

.. _bs-data:

Constants
---------

.. c:var:: const size_t stream_word_bits

  The number of bits in a word.  The size of a flushed bit stream will be
  a multiple of this number of bits.  See :c:macro:`BIT_STREAM_WORD_TYPE`.

.. _bs-functions:

Functions
---------

.. c:function:: bitstream* stream_open(void* buffer, size_t bytes)

  Allocate a :c:type:`bitstream` struct and associate it with the memory
  buffer allocated by the caller.

.. c:function:: void stream_close(bitstream* stream)

  Close the bit stream and deallocate *stream*.

.. c:function:: bitstream* stream_clone(const bitstream* stream)

  Create a copy of *stream* that points to the same memory buffer.

.. c:function:: void* stream_data(const bitstream* stream)

  Return pointer to the beginning of bit stream *stream*.

.. c:function:: size_t stream_size(const bitstream* stream)

  Return position of stream pointer in number of bytes, which equals the
  end of stream if no seeks have been made.  Note that additional bits
  may be buffered and not reported unless the stream has been flushed.

.. c:function:: size_t stream_capacity(const bitstream* stream)

  Return byte size of memory buffer associated with *stream*.

.. c:function:: uint stream_read_bit(bitstream* stream)

  Read a single bit from *stream*.

.. c:function:: uint stream_write_bit(bitstream* stream, uint bit)

  Write single *bit* to *stream*.  *bit* must be one of 0 or 1.

.. c:function:: uint64 stream_read_bits(bitstream* stream, uint n)

  Read and return 0 |leq| *n* |leq| 64 bits from *stream*.

.. c:function:: uint64 stream_write_bits(bitstream* stream, uint64 value, uint n)

  Write 0 |leq| *n* |leq| 64 low bits of *value* to *stream*.  Return any
  remaining bits from *value*, i.e., *value* >> *n*.

.. c:function:: size_t stream_rtell(const bitstream* stream)

  Return bit offset to next bit to be read.

.. c:function:: size_t stream_wtell(const bitstream* stream)

  Return bit offset to next bit to be written.

.. c:function:: void stream_rewind(bitstream* stream)

  Rewind stream to beginning of memory buffer.  Following this call, the
  stream may either be read or written.

.. c:function:: void stream_rseek(bitstream* stream, size_t offset)

  Position stream for reading at given bit offset.  This places the
  stream in read mode.

.. c:function:: void stream_wseek(bitstream* stream, size_t offset)

  Position stream for writing at given bit offset.  This places the
  stream in write mode.

.. c:function:: void stream_skip(bitstream* stream, uint n)

  Skip over the next *n* bits, i.e., without reading them.

.. c:function:: void stream_pad(bitstream* stream, uint n)

  Append *n* zero-bits to *stream*.

.. c:function:: size_t stream_align(bitstream* stream)

  Align stream on next word boundary by skipping bits.  No skipping is
  done if the stream is already word aligned.  Return the number of
  skipped bits, if any.

.. c:function:: size_t stream_flush(bitstream* stream)

  Write out any remaining buffered bits.  When one or more bits are
  buffered, append zero-bits to the stream to align it on a word boundary.
  Return the number of bits of padding, if any.

.. c:function:: void stream_copy(bitstream* dst, bitstream* src, size_t n)

  Copy *n* bits from *src* to *dst*, advancing both bit streams.

.. c:function:: size_t stream_stride_block(const bitstream* stream)

  Return stream block size in number of words.  The block size is always
  one word unless strided streams are enabled.  See :ref:`bs-strides`
  for more information.

.. c:function:: ptrdiff_t stream_stride_delta(const bitstream* stream)

  Return stream delta in number of words between blocks.  See
  :ref:`bs-strides` for more information.

.. c:function:: int stream_set_stride(bitstream* stream, size_t block, ptrdiff_t delta)

  Set block size, *block*, in number of words and spacing, *delta*, in number
  of blocks for strided access.  Requires :c:macro:`BIT_STREAM_STRIDED`.
