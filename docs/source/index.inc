.. index::
   single: Index
.. _index:

Index
-----

.. cpp:namespace:: zfp

To support random access, |zfp| arrays must know where each block is stored
in memory.  For fixed-rate arrays, the number of compressed bits per block
is constant, and the bit offset to each block can be quickly computed.  For
variable-rate arrays, the compressed block size is data dependent, and
additional information must be stored to index the blocks.  Toward this end,
|zfp| arrays make use of an index class that reports the offset and
size (in number of bits) of each block.  The :cpp:class:`zfp::array`
and :cpp:class:`zfp::const_array` classes take such an index class as a
template parameter.  This index class is new as of |zfp| |carrrelease|, which
introduced variable-rate arrays.

Because |zfp| is designed primarily for very large arrays, the bit offset
may exceed 32 bits.  A straightforward implementation stores the bit
offset to each block as a 64-bit integer, with the block size given by
the difference of consecutive offsets.  However, this overhead of
64 bits/block may exceed the payload compressed data for low-dimensional
arrays or in applications like visualization that may store less than one
bit per value (amortized).  It is therefore important to consider more
compact representations of the block index.

|zfp| provides multiple index classes in the :code:`zfp::index` namespace
that balance storage size, range of representable block offsets and sizes,
and speed of access:

.. cpp:namespace:: zfp::index

* :cpp:class:`implicit`: Used for fixed-rate storage where only the
  fixed number of bits per block is kept.  This is the default index for
  fixed-rate arrays.

* :cpp:class:`verbatim`: This and subsequent classes support variable-rate
  storage.  A full 64-bit offset is stored per block.

* :cpp:class:`hybrid4`: Four consecutive offsets are encoded together.
  The top 32 bits of a 44-bit base offset are stored, with the 12 least
  significant bits of this base set to zero.  Four unsigned 16-bit deltas
  from the base offset complete the representation.  The default for
  variable-rate arrays, this index offers a good tradeoff between storage,
  offset range, and speed.

* :cpp:class:`hybrid8`: Eight consecutive offsets are encoded together
  as two 64-bit words that store the offset to the first block (the base
  offset) and the sizes of the first seven blocks, from which the eight
  offsets are derived as a prefix sum.  One 64-bit word holds the 8 least
  significant bits of the base offset and block sizes.  The other word
  holds another 2 (*d* - 1) bits for the seven block sizes plus the top
  78 - 14 *d* bits of the base offset, where 1 |leq| *d* |leq| 4 is the
  data dimensionality.

Properties of these index classes are summarized in :numref:`index-classes`.

.. _index-classes:
.. table:: Properties of index classes.  Storage is measured in amortized
           bits/block; offset and size denote supported ranges
           in number of bits.

  +-------------+----------+---------+-------------+-----------+--------+
  | index class | variable | storage | offset      | size      | speed  |
  |             | rate     |         |             |           |        |
  +=============+==========+=========+=============+===========+========+
  | implicit    |          |     0   |     64      |    64     |  high  |
  +-------------+----------+---------+-------------+-----------+--------+
  | verbatim    | |check|  |    64   |     64      |    64     |  high  |
  +-------------+----------+---------+-------------+-----------+--------+
  | hybrid4     | |check|  |    24   |     44      |    16     | medium |
  +-------------+----------+---------+-------------+-----------+--------+
  | hybrid8     | |check|  |    16   | 86 - 14 *d* | 6 + 2 *d* |   low  |
  +-------------+----------+---------+-------------+-----------+--------+

This section documents the API that prospective block indices must support to
interface with the |zfp| compressed-array classes.

.. cpp:class:: index

  Fictitious class encapsulating the index API.

----

.. cpp:function:: index::index(size_t blocks)

  Construct index supporting the given number of *blocks*.

----

.. cpp:function:: size_t index::size_bytes(uint mask = ZFP_DATA_ALL) const

----

.. cpp:function:: bitstream_size index::range() const

  Range of bit offsets spanned by index.  This equals the total number of
  bits of compressed-array data.

----

.. cpp:function:: size_t index::block_size(size_t block_index) const

  Size of compressed block in number of bits.

----

.. cpp:function:: bitstream_offset index::block_offset(size_t block_index) const

  Bit offset to compressed block data.

----

.. cpp:function:: void resize(size_t blocks)

  Resize index to accommodate requested number of blocks.  Any stored
  index data is destroyed.

----

.. cpp:function:: void clear()

  Clear all data stored by index.

----

.. cpp:function:: void flush()

  Flush any buffered index data.  This method is called after all blocks
  have been compressed, e.g., in :cpp:func:`array::set`.

----

.. cpp:function:: void set_block_size(size_t size)

  Set a fixed compressed block size in number of bits for all blocks.  This
  method is called when fixed-rate mode is selected.

----

.. cpp:function:: void set_block_size(size_t block_index, size_t size)

  Set compressed block size in number of bits for a single block.  For
  variable-rate arrays, the zero-based *block_index* is guaranteed to
  increase sequentially between calls.  This method throws an exception
  if the index cannot support the block size or offset.  The user may
  wish to restrict the block size, e.g., by setting :code:`maxbits` in
  :ref:`expert mode <mode-expert>`, to guard against such overflow.

----

.. cpp:function:: static bool has_variable_rate()

  Return true if index supports variable-sized blocks.
