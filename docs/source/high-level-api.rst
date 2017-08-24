.. include:: defs.rst

.. _hl-api:

High-Level C API
================

The C API is broken down into a :ref:`high-level API <hl-api>`,
which handles compression of entire arrays, and a
:ref:`low-level-api <ll-api>` for processing individual blocks
and managing the underlying bit stream.

The high-level API should be the API of choice for applications that
compress and decompress entire arrays.  A :ref:`low-level API <ll-api>`
exists for processing individual, possibly partial blocks as well as
reduced-precision integer data less than 32 bits wide.

The following sections are available:

* :ref:`hl-macros`
* :ref:`hl-types`
* :ref:`hl-data`
* :ref:`hl-functions`

  * :ref:`hl-func-bitstream`
  * :ref:`hl-func-stream`
  * :ref:`hl-func-field`
  * :ref:`hl-func-codec`

.. _hl-macros:

Macros
------

.. c:macro:: ZFP_VERSION_MAJOR
.. c:macro:: ZFP_VERSION_MINOR
.. c:macro:: ZFP_VERSION_PATCH
.. c:macro:: ZFP_VERSION
.. c:macro:: ZFP_VERSION_STRING

  Macros identifying the |zfp| library version.  :c:macro:`ZFP_VERSION` is
  a single integer constructed from the previous three macros.
  :c:macro:`ZFP_VERSION_STRING` is a string literal.  See also
  :c:data:`zfp_library_version` and :c:data:`zfp_version_string`.

.. c:macro:: ZFP_CODEC

  Macro identifying the version of the compression CODEC.  See also
  :c:data:`zfp_codec_version`.

.. c:macro:: ZFP_MIN_BITS
.. c:macro:: ZFP_MAX_BITS
.. c:macro:: ZFP_MAX_PREC
.. c:macro:: ZFP_MIN_EXP

  Default compression parameter settings that impose no constraints.
  The largest possible compressed block size, corresponding to 3D blocks
  of doubles, is given by :c:macro:`ZFP_MAX_BITS`.  See also
  :c:type:`zfp_stream`.

.. c:macro:: ZFP_HEADER_MAGIC
.. c:macro:: ZFP_HEADER_META
.. c:macro:: ZFP_HEADER_MODE
.. c:macro:: ZFP_HEADER_FULL

  Bit masks for specifying which portions of a header to output (if any).
  These constants should be bitwise ORed together.  Use
  :c:macro:`ZFP_HEADER_FULL` to output all header information available.
  The compressor and decompressor must agree on which parts of the header
  to read/write.

  :c:macro:`ZFP_HEADER_META` in essence encodes the information stored in
  the :c:type:`zfp_field` struct, while :c:macro:`ZFP_HEADER_MODE` encodes
  the compression parameters stored in the :c:type:`zfp_stream` struct.
  The magic can be used to uniquely identify the stream as a |zfp| stream,
  and includes the CODEC version.

  See :c:func:`zfp_read_header` and :c:func:`zfp_write_header` for
  how to read and write header information.

.. c:macro:: ZFP_MAGIC_BITS
.. c:macro:: ZFP_META_BITS
.. c:macro:: ZFP_MODE_SHORT_BITS
.. c:macro:: ZFP_MODE_LONG_BITS
.. c:macro:: ZFP_HEADER_MAX_BITS
.. c:macro:: ZFP_MODE_SHORT_MAX

  Number of bits used by each portion of the header.  These macros are
  primarily informational and should not be accessed by the user through
  the high-level API.  For most common compression parameter settings,
  only :c:macro:`ZFP_MODE_SHORT_BITS` bits of header information are stored
  to encode the mode (see :c:func:`zfp_stream_mode`).

.. _hl-types:

Types
-----

.. c:type:: zfp_stream

  The :c:type:`zfp_stream` struct encapsulates all information about the
  compressed stream for a single block or a collection of blocks that
  represent an array.  See the section on :ref:`compression modes <modes>`
  for a description of the members of this struct.
  ::

    typedef struct {
      uint minbits;      // minimum number of bits to store per block
      uint maxbits;      // maximum number of bits to store per block
      uint maxprec;      // maximum number of bit planes to store
      int minexp;        // minimum floating point bit plane number to store
      bitstream* stream; // compressed bit stream
    } zfp_stream;

.. c:type:: zfp_type

  Enumerates the scalar types supported by the compressor, and is used to
  describe the uncompressed array.  The compressor and decompressor must use
  the same :c:type:`zfp_type`, e.g. one cannot compress doubles and decompress
  to floats or integers.
  ::

    typedef enum {
      zfp_type_none   = 0, // unspecified type
      zfp_type_int32  = 1, // 32-bit signed integer
      zfp_type_int64  = 2, // 64-bit signed integer
      zfp_type_float  = 3, // single precision floating point
      zfp_type_double = 4  // double precision floating point
    } zfp_type;

.. c:type:: zfp_field

  The uncompressed array is described by the :c:type:`zfp_field` struct, which
  encodes the array's scalar type, dimensions, and memory layout.
  ::

    typedef struct {
      zfp_type type;   // scalar type (e.g. int32, double)
      uint nx, ny, nz; // sizes (zero for unused dimensions)
      int sx, sy, sz;  // strides (zero for contiguous array a[nz][ny][nx])
      void* data;      // pointer to array data
    } zfp_field;

  For example, a static multidimensional C array declared as
  ::

    double array[n1][n2][n3];

  would be described by a :c:type:`zfp_field` with members
  ::

    type = zfp_type_double;
    nx = n3; ny = n2; nz = n1;
    sx = 1; sy = n3; sz = n2 * n3;
    data = &array[0][0][0];

.. _hl-data:

Constants
---------

.. c:var:: const uint zfp_codec_version

  The version of the compression CODEC implemented by this version of the |zfp|
  library.  The library can decompress files generated by the same CODEC only.
  To ensure that the :file:`zfp.h` header matches the binary library linked to,
  :c:data:`zfp_codec_version` should match :c:macro:`ZFP_CODEC`.

.. c:var:: const uint zfp_library_version

  The library version.  The binary library and headers are compatible if
  :c:data:`zfp_library_version` matches :c:macro:`ZFP_VERSION`.

.. c:var:: const char* const zfp_version_string

  A constant string representing the |zfp| library version and release date.
  One can search for this string in executables and libraries that use |zfp|
  to determine which version of the library the application was compiled
  against.

.. _hl-functions:

Functions
---------

.. c:function:: size_t zfp_type_size(zfp_type type)

  Return byte size of the given scalar type, e.g.
  :code:`zfp_type_size(zfp_type_float) = 4`.

.. _hl-func-bitstream:

Compressed Stream
^^^^^^^^^^^^^^^^^

.. c:function:: zfp_stream* zfp_stream_open(bitstream* stream)

  Allocate compressed stream and associate it with bit stream for reading
  and writing bits to/from memory.  *stream* may be :c:macro:`NULL` and
  attached later via :c:func:`zfp_stream_set_bit_stream`.

.. c:function:: void zfp_stream_close(zfp_stream* stream)

  Close and deallocate compressed stream.  This does not affect the
  attached bit stream.

.. c:function:: bitstream* zfp_stream_bit_stream(const zfp_stream* stream)

  Return bit stream associated with compressed stream.

.. c:function:: uint64 zfp_stream_mode(const zfp_stream* zfp)

  Return compact encoding of compression parameters.  If the return value
  is no larger than :c:macro:`ZFP_MODE_SHORT_MAX`, then the least significant
  :c:macro:`ZFP_MODE_SHORT_BITS` (12 in the current version) suffice to
  encode the parameters.  Otherwise all 64 bits are needed, and the low
  :c:macro:`ZFP_MODE_SHORT_BITS` bits will be all ones.  Thus, this
  variable-length encoding can be used to economically encode and decode
  the compression parameters, which is especially important if the parameters
  are to vary spatially over small regions.  Such spatially adaptive coding
  would have to be implemented via the low-level API.

.. c:function:: void zfp_stream_params(const zfp_stream* stream, uint* minbits, uint* maxbits, uint* maxprec, int* minexp)

  Query :ref:`compression parameters <mode-expert>`.  For any parameter not
  needed, pass :c:macro:`NULL` for the corresponding pointer.

.. c:function:: size_t zfp_stream_compressed_size(const zfp_stream* stream)

  Number of bytes of compressed storage.  This function returns the
  current byte offset within the bit stream from the beginning of the
  bit stream memory buffer.  To ensure all buffered compressed data has
  been output call :c:func:`zfp_stream_flush` first.

.. c:function:: size_t zfp_stream_maximum_size(const zfp_stream* stream, const zfp_field* field)

  Conservative estimate of the compressed byte size for the compression
  parameters stored in *stream* and the array whose scalar type and dimensions
  are given by *field*.  This function may be used to determine how large a
  memory buffer to allocate to safely hold the entire compressed array.

.. c:function:: void zfp_stream_set_bit_stream(zfp_stream* stream, bitstream* bs)

  Associate bit stream with compressed stream.

.. c:function:: void zfp_stream_rewind(zfp_stream* stream)

  Rewind bit stream to beginning for compression or decompression.


.. _hl-func-stream:

Compression Parameters
^^^^^^^^^^^^^^^^^^^^^^

.. c:function:: double zfp_stream_set_rate(zfp_stream* stream, double rate, zfp_type type, uint dims, int wra)

  Set *rate* for :ref:`fixed-rate mode <mode-fixed-rate>` in compressed bits
  per value.  The target scalar *type* and array *dimensionality* are needed
  to correctly translate the rate to the number of bits per block.  The
  parameter *wra* should be nonzero if random access writes of blocks into
  the compressed bit stream is needed, for example for implementing |zfp|'s
  :ref:`compressed arrays <arrays>`.  This requires blocks to be aligned on
  :ref:`bit stream word <q-portability>` boundaries, and therefore constrains
  the rate.  The closest supported rate is returned, which may differ from
  the desired rate.

.. c:function:: uint zfp_stream_set_precision(zfp_stream* stream, uint precision)

  Set *precision* for :ref:`fixed-precision mode <mode-fixed-precision>`.
  The precision specifies how many uncompressed bits per value to store,
  and indirectly governs the relative error.  The actual precision is
  returned, e.g. in case the desired precision is out of range.  To
  preserve a certain floating-point mantissa or integer precision in the
  decompressed data, see :ref:`FAQ #21 <q-lossless>`.

.. c:function:: double zfp_stream_set_accuracy(zfp_stream* stream, double tolerance)

  Set absolute error *tolerance* for
  :ref:`fixed-accuracy mode <mode-fixed-accuracy>`.  The tolerance ensures
  that values in the decompressed array differ from the input array by no
  more than this tolerance (in all but exceptional circumstances; see
  :ref:`FAQ #17 <q-tolerance>`).  This compression mode should be used only
  with floating-point (not integer) data.

.. c:function:: int zfp_stream_set_mode(zfp_stream* stream, uint64 mode)

  Set all compression parameters from compact integer representation.
  See :c:func:`zfp_stream_mode` for how to encode the parameters.  The
  return value is nonzero upon success.

.. c:function:: int zfp_stream_set_params(zfp_stream* stream, uint minbits, uint maxbits, uint maxprec, int minexp)

  Set all compression parameters directly.  See the section on
  :ref:`expert mode <mode-expert>` for a discussion of the parameters.
  The return value is nonzero upon success.

.. _hl-func-field:

Array Metadata
^^^^^^^^^^^^^^

.. c:function:: zfp_field* zfp_field_alloc()

  Allocates and returns a default initialized :c:type:`zfp_field` struct.
  The caller must free this struct using :c:func:`zfp_field_free`.

.. c:function:: zfp_field* zfp_field_1d(void* pointer, zfp_type type, uint nx)

  Allocate and return a field struct that describes an existing 1D array,
  :code:`a[nx]`, of *nx* uncompressed scalars of given *type* stored at
  *pointer*, which may be :c:macro:`NULL` and specified later.

.. c:function:: zfp_field* zfp_field_2d(void* pointer, zfp_type type, uint nx, uint ny)

  Allocate and return a field struct that describes an existing 2D array,
  :code:`a[ny][nx]`, of *nx* |times| *ny* uncompressed scalars of given
  *type* stored at *pointer*, which may be :c:macro:`NULL` and specified
  later.

.. c:function:: zfp_field* zfp_field_3d(void* pointer, zfp_type type, uint nx, uint ny, uint nz)

  Allocate and return a field struct that describes an existing 3D array,
  :code:`a[nz][ny][nx]`, of *nx* |times| *ny* |times| *nz* uncompressed
  scalars of given *type* stored at *pointer*, which may be :c:macro:`NULL`
  and specified later.

.. c:function:: void zfp_field_free(zfp_field* field)

  Free :c:type:`zfp_field` struct previously allocated by one of the functions
  above.

.. c:function:: void* zfp_field_pointer(const zfp_field* field)

  Return pointer to the first scalar in the array.

.. c:function:: zfp_type zfp_field_type(const zfp_field* field)

  Return array scalar type.

.. c:function:: uint zfp_field_precision(const zfp_field* field)

  Return scalar precision in number of bits, e.g. 32 for
  :code:`zfp_type_float`.

.. c:function:: uint zfp_field_dimensionality(const zfp_field* field)

  Return array dimensionality (1, 2, or 3).

.. c:function:: size_t zfp_field_size(const zfp_field* field, uint* size)

  Return total number of scalars stored in the array, e.g.
  *nx* |times| *ny* |times| *nz* for a 3D array.  If *size* is not
  :c:macro:`NULL`, then store the number of scalars for each dimension,
  e.g. :code:`size[0] = nx; size[1] = ny; size[2] = nz` for a 3D array.

.. c:function:: int zfp_field_stride(const zfp_field* field, int* stride)

  Return zero if array is stored contiguously; nonzero if it is strided.
  If *stride* is not :c:macro:`NULL`, then store the stride for each
  dimension, e.g. :code:`stride[0] = sx; stride[1] = sy; stride[2] = sz`
  for a 3D array.  See below for more information on strides.

.. c:function:: uint64 zfp_field_metadata(const zfp_field* field)

  Return 52-bit compact encoding of the scalar type and array dimensions.

.. c:function:: void zfp_field_set_pointer(zfp_field* field, void* pointer)

  Set pointer to first scalar in the array.

.. c:function:: zfp_type zfp_field_set_type(zfp_field* field, zfp_type type)

  Set array scalar type.

.. c:function:: void zfp_field_set_size_1d(zfp_field* field, uint nx)

  Specify dimensions of 1D array :code:`a[nx]`.

.. c:function:: void zfp_field_set_size_2d(zfp_field* field, uint nx, uint ny)

  Specify dimensions of 2D array :code:`a[ny][nx]`.

.. c:function:: void zfp_field_set_size_3d(zfp_field* field, uint nx, uint ny, uint nz)

  Specify dimensions of 3D array :code:`a[nz][ny][nx]`.

.. c:function:: void zfp_field_set_stride_1d(zfp_field* field, int sx)

  Specify stride for 1D array: :code:`sx = &a[1] - &a[0]`.

.. c:function:: void zfp_field_set_stride_2d(zfp_field* field, int sx, int sy)

  Specify strides for 2D array:
  :code:`sx = &a[0][1] - &a[0][0]; sy = &a[1][0] - &a[0][0]`.

.. c:function:: void zfp_field_set_stride_3d(zfp_field* field, int sx, int sy, int sz)

  Specify strides for 3D array:
  :code:`sx = &a[0][0][1] - &a[0][0][0]; sy = &a[0][1][0] - &a[0][0][0]; sz = &a[1][0][0] - &a[0][0][0]`.

.. c:function:: int zfp_field_set_metadata(zfp_field* field, uint64 meta)

  Specify array scalar type and dimensions from compact 52-bit representation.
  Returns nonzero upon success.  See :c:func:`zfp_field_metadata` for how to
  encode *meta*.

.. _hl-func-codec:

Compression and Decompression
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. c:function:: size_t zfp_compress(zfp_stream* stream, const zfp_field* field)

  Compress the whole array described by *field* using parameters given by
  *stream* and then flush the stream.  The number of bytes of compressed storage
  is returned, if the stream were rewound before compression, and otherwise the
  current byte offset within the bit stream.  Zero is returned if compression
  failed.

.. c:function:: int zfp_decompress(zfp_stream* stream, zfp_field* field)

  Decompress from *stream* to array described by *field* and align the stream on
  the next word boundary.  Nonzero is returned upon success.

.. c:function:: size_t zfp_write_header(zfp_stream* stream, const zfp_field* field, uint mask)

  Write an optional header to the stream that encodes compression parameters,
  array metadata, etc.  The header information written is determined by the
  bit *mask* (see :c:macro:`macros <ZFP_HEADER_MAGIC>`).  The return value is
  the number of bits written, or zero upon failure.

.. c:function:: size_t zfp_read_header(zfp_stream* stream, zfp_field* field, uint mask)

  Read header if one was previously written using :c:func:`zfp_write_header`.
  The return value is the number of bits read, or zero upon failure.  The
  caller must ensure that the bit *mask* agrees between header read and
  write calls.
