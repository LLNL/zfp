.. include:: defs.rst

.. index::
   single: zFORp
.. _zforp:

Fortran Bindings
================

|zfp| |zforprelease| adds |zforp|: a Fortran API providing wrappers around
the :ref:`high-level C API <hl-api>`. Wrappers for
:ref:`compressed-array classes <arrays>` will arrive in a future release.
The |zforp| implementation is based on the standard :code:`iso_c_binding`
module available since Fortran 2003.  The use of :code:`ptrdiff_t` in
the |zfp| |fieldrelease| C API, however, requires the corresponding
:code:`c_ptrdiff_t` available only since Fortran 2018.

Every high-level C API function can be called from a Fortran wrapper function.
C structs are wrapped as Fortran derived types, each containing a single C
pointer to the C struct in memory. The wrapper functions accept and return
these Fortran types, so users should never need to touch the C pointers.
In addition to the high-level C API, two essential functions from the
:ref:`bit stream API <bs-api>` for opening and closing bit streams are
available.

See example code :file:`tests/fortran/testFortran.f` (on the GitHub
`develop branch <https://github.com/LLNL/zfp/tree/develop>`_)
for how the Fortran API is used to compress and decompress data.

.. _zforp_changes:
.. note::

  |zfp| |fieldrelease| simplifies the |zforp| module name from
  ``zforp_module`` to ``zfp``.  This will likely require changing
  associated use statements within existing code when updating
  from prior versions of zFORp.

  Furthermore, as outlined above, the |zfp| |fieldrelease| API requires
  a Fortran 2018 compiler.


Types
-----

.. f:type:: zFORp_bitstream

  :f c_ptr object: A C pointer to the instance of :c:type:`bitstream`

----

.. f:type:: zFORp_stream

  :f c_ptr object: A C pointer to the instance of :c:type:`zfp_stream`

----

.. f:type:: zFORp_field

  :f c_ptr object: A C pointer to the instance of :c:type:`zfp_field`

Constants
---------

Enumerations
^^^^^^^^^^^^

.. _zforp_type:
.. f:variable:: integer zFORp_type_none
.. f:variable:: integer zFORp_type_int32
.. f:variable:: integer zFORp_type_int64
.. f:variable:: integer zFORp_type_float
.. f:variable:: integer zFORp_type_double

  Enums wrapping :c:type:`zfp_type`

----

.. _zforp_mode:
.. f:variable:: integer zFORp_mode_null
.. f:variable:: integer zFORp_mode_expert
.. f:variable:: integer zFORp_mode_fixed_rate
.. f:variable:: integer zFORp_mode_fixed_precision
.. f:variable:: integer zFORp_mode_fixed_accuracy
.. f:variable:: integer zFORp_mode_reversible

  Enums wrapping :c:type:`zfp_mode`

----

.. _zforp_exec:
.. f:variable:: integer zFORp_exec_serial
.. f:variable:: integer zFORp_exec_omp
.. f:variable:: integer zFORp_exec_cuda

  Enums wrapping :c:type:`zfp_exec_policy`

Non-Enum Constants
^^^^^^^^^^^^^^^^^^

.. f:variable:: integer zFORp_version_major

  Wraps :c:macro:`ZFP_VERSION_MAJOR`

----

.. f:variable:: integer zFORp_version_minor

  Wraps :c:macro:`ZFP_VERSION_MINOR`

----

.. f:variable:: integer zFORp_version_patch

  Wraps :c:macro:`ZFP_VERSION_PATCH`

----

.. f:variable:: integer zFORp_version_tweak

  Wraps :c:macro:`ZFP_VERSION_TWEAK`

----

.. f:variable:: integer zFORp_codec_version

  Wraps :c:data:`zfp_codec_version`

----

.. f:variable:: integer zFORp_library_version

  Wraps :c:data:`zfp_library_version`

----

.. f:variable:: character(len=36) zFORp_version_string

  Wraps :c:data:`zfp_version_string`

----

.. f:variable:: integer zFORp_min_bits

  Wraps :c:macro:`ZFP_MIN_BITS`

----

.. f:variable:: integer zFORp_max_bits

  Wraps :c:macro:`ZFP_MAX_BITS`

----

.. f:variable:: integer zFORp_max_prec

  Wraps :c:macro:`ZFP_MAX_PREC`

----

.. f:variable:: integer zFORp_min_exp

  Wraps :c:macro:`ZFP_MIN_EXP`

----

.. _zforp_header:
.. f:variable:: integer zFORp_header_magic

  Wraps :c:macro:`ZFP_HEADER_MAGIC`

----

.. f:variable:: integer zFORp_header_meta

  Wraps :c:macro:`ZFP_HEADER_META`

----

.. f:variable:: integer zFORp_header_mode

  Wraps :c:macro:`ZFP_HEADER_MODE`

----

.. f:variable:: integer zFORp_header_full

  Wraps :c:macro:`ZFP_HEADER_FULL`

----

.. f:variable:: integer zFORp_meta_null

  Wraps :c:macro:`ZFP_META_NULL`

----

.. f:variable:: integer zFORp_magic_bits

  Wraps :c:macro:`ZFP_MAGIC_BITS`

----

.. f:variable:: integer zFORp_meta_bits

  Wraps :c:macro:`ZFP_META_BITS`

----

.. f:variable:: integer zFORp_mode_short_bits

  Wraps :c:macro:`ZFP_MODE_SHORT_BITS`

----

.. f:variable:: integer zFORp_mode_long_bits

  Wraps :c:macro:`ZFP_MODE_LONG_BITS`

----

.. f:variable:: integer zFORp_header_max_bits

  Wraps :c:macro:`ZFP_HEADER_MAX_BITS`

----

.. f:variable:: integer zFORp_mode_short_max

  Wraps :c:macro:`ZFP_MODE_SHORT_MAX`

Functions and Subroutines
-------------------------

Each of the functions included here wraps a corresponding C function.  Please
consult the C documentation for detailed descriptions of the functions, their
parameters, and their return values.

Bit Stream
^^^^^^^^^^

.. f:function:: zFORp_bitstream_stream_open(buffer, bytes)

  Wrapper for :c:func:`stream_open`

  :p c_ptr buffer [in]: Memory buffer
  :p bytes [in]: Buffer size in bytes
  :ptype bytes: integer (kind=8)
  :r bs: Bit stream
  :rtype bs: zFORp_bitstream

----

.. f:subroutine:: zFORp_bitstream_stream_close(bs)

  Wrapper for :c:func:`stream_close`

  :p zFORp_bitstream bs [inout]: Bit stream

Utility Functions
^^^^^^^^^^^^^^^^^

.. f:function:: zFORp_type_size(scalar_type)

  Wrapper for :c:func:`zfp_type_size`

  :p integer scalar_type [in]: :ref:`zFORp_type <zforp_type>` enum
  :r type_size: Size of described :c:type:`zfp_type`, in bytes, from C-language perspective
  :rtype type_size: integer (kind=8)

Compressed Stream
^^^^^^^^^^^^^^^^^

.. f:function:: zFORp_stream_open(bs)

  Wrapper for :c:func:`zfp_stream_open`

  :p zFORp_bitstream bs [in]: Bit stream
  :r stream: Newly allocated compressed stream
  :rtype stream: zFORp_stream

----

.. f:subroutine:: zFORp_stream_close(stream)

  Wrapper for :c:func:`zfp_stream_close`

  :p zFORp_stream stream [inout]: Compressed stream

----

.. f:function:: zFORp_stream_bit_stream(stream)

  Wrapper for :c:func:`zfp_stream_bit_stream`

  :p zFORp_stream stream [in]: Compressed stream
  :r bs: Bit stream
  :rtype bs: zFORp_bitstream

----

.. f:function:: zFORp_stream_compression_mode(stream)

  Wrapper for :c:func:`zfp_stream_compression_mode`

  :p zFORp_stream stream [in]: Compressed stream
  :r mode: :ref:`zFORp_mode <zforp_mode>` enum
  :rtype mode: integer

----

.. f:function:: zFORp_stream_rate(stream, dims)

  Wrapper for :c:func:`zfp_stream_rate`

  :p zFORp_stream stream [in]: Compressed stream
  :p integer dims [in]: Number of dimensions
  :r rate_result: Rate in compressed bits/scalar
  :rtype rate_result: real (kind=8)

----

.. f:function:: zFORp_stream_precision(stream)

  Wrapper for :c:func:`zfp_stream_precision`

  :p zFORp_stream stream [in]: Compressed stream
  :r prec_result: Precision in uncompressed bits/scalar
  :rtype prec_result: integer

----

.. f:function:: zFORp_stream_accuracy(stream)

  Wrapper for :c:func:`zfp_stream_accuracy`

  :p zFORp_stream stream [in]: Compressed stream
  :r tol_result: Absolute error tolerance
  :rtype tol_result: real (kind=8)

----

.. f:function:: zFORp_stream_mode(stream)

  Wrapper for :c:func:`zfp_stream_mode`

  :p zFORp_stream stream [in]: Compressed stream
  :r mode: 64-bit encoded mode
  :rtype mode: integer (kind=8)

----

.. f:subroutine:: zFORp_stream_params(stream, minbits, maxbits, maxprec, minexp)

  Wrapper for :c:func:`zfp_stream_params`

  :p zFORp_stream stream [in]: Compressed stream
  :p integer minbits [inout]: Minimum number of bits per block
  :p integer maxbits [inout]: Maximum number of bits per block
  :p integer maxprec [inout]: Maximum precision
  :p integer minexp [inout]: Minimum bit plane number encoded

----

.. f:function:: zFORp_stream_compressed_size(stream)

  Wrapper for :c:func:`zfp_stream_compressed_size`

  :p zFORp_stream stream [in]: Compressed stream
  :r compressed_size: Compressed size in bytes
  :rtype compressed_size: integer (kind=8)

----

.. f:function:: zFORp_stream_maximum_size(stream, field)

  Wrapper for :c:func:`zfp_stream_maximum_size`

  :p zFORp_stream stream [in]: Compressed stream
  :p zFORp_field field [in]: Field metadata
  :r max_size: Maximum possible compressed size in bytes
  :rtype max_size: integer (kind=8)

----

.. f:subroutine:: zFORp_stream_rewind(stream)

  Wrapper for :c:func:`zfp_stream_rewind`

  :p zFORp_stream stream [in]: Compressed stream

----

.. f:subroutine:: zFORp_stream_set_bit_stream(stream, bs)

  Wrapper for :c:func:`zfp_stream_set_bit_stream`

  :p zFORp_stream stream [in]: Compressed stream
  :p zFORp_bitstream bs [in]: Bit stream


Compression Parameters
^^^^^^^^^^^^^^^^^^^^^^

.. f:subroutine:: zFORp_stream_set_reversible(stream)

  Wrapper for :c:func:`zfp_stream_set_reversible`

  :p zFORp_stream stream [in]: Compressed stream

----

.. f:function:: zFORp_stream_set_rate(stream, rate, scalar_type, dims, align)

  Wrapper for :c:func:`zfp_stream_set_rate`

  :p zFORp_stream stream [in]: Compressed stream
  :p real rate [in]: Desired rate
  :p integer scalar_type [in]: :ref:`zFORp_type <zforp_type>` enum
  :p integer dims [in]: Number of dimensions
  :p integer align [in]: Align blocks on words for write random access?
  :r rate_result: Actual set rate in bits/scalar
  :rtype rate_result: real (kind=8)

----

.. f:function:: zFORp_stream_set_precision(stream, prec)

  Wrapper for :c:func:`zfp_stream_set_precision`

  :p zFORp_stream stream [in]: Compressed stream
  :p integer prec [in]: Desired precision
  :r prec_result: Actual set precision
  :rtype prec_result: integer

----

.. f:function:: zFORp_stream_set_accuracy(stream, tolerance)

  Wrapper for :c:func:`zfp_stream_set_accuracy()`

  :p zFORp_stream stream [in]: Compressed stream
  :p tolerance [in]: Desired error tolerance
  :ptype tolerance: real (kind=8)
  :r tol_result: Actual set tolerance
  :rtype tol_result: real (kind=8)

----

.. f:function:: zFORp_stream_set_mode(stream, mode)

  Wrapper for :c:func:`zfp_stream_set_mode`

  :p zFORp_stream stream [in]: Compressed stream
  :p mode [in]: Compact encoding of compression parameters
  :ptype mode: integer (kind=8)
  :r mode_result: Newly set :ref:`zFORp_mode <zforp_mode>` enum
  :rtype mode_result: integer

----

.. f:function:: zFORp_stream_set_params(stream, minbits, maxbits, maxprec, minexp)

  Wrapper for :c:func:`zfp_stream_set_params`

  :p zFORp_stream stream [in]: Compressed stream
  :p integer minbits [in]: Minimum number of bits per block
  :p integer maxbits [in]: Maximum number of bits per block
  :p integer maxprec [in]: Maximum precision
  :p integer minexp [in]: Minimum bit plane number encoded
  :r is_success: Indicate whether parameters were successfully set (1) or not (0)
  :rtype is_success: integer

Execution Policy
^^^^^^^^^^^^^^^^

.. f:function:: zFORp_stream_execution(stream)

  Wrapper for :c:func:`zfp_stream_execution`

  :p zFORp_stream stream [in]: Compressed stream
  :r execution_policy: :ref:`zFORp_exec <zforp_exec>` enum indicating active execution policy
  :rtype execution_policy: integer

----

.. f:function:: zFORp_stream_omp_threads(stream)

  Wrapper for :c:func:`zfp_stream_omp_threads`

  :p zFORp_stream stream [in]: Compressed stream
  :r thread_count: Number of OpenMP threads to use upon execution
  :rtype thread_count: integer

----

.. f:function:: zFORp_stream_omp_chunk_size(stream)

  Wrapper for :c:func:`zfp_stream_omp_chunk_size`

  :p zFORp_stream stream [in]: Compressed stream
  :r chunk_size_blocks: Specified chunk size, in blocks
  :rtype chunk_size_blocks: integer (kind=8)

----

.. f:function:: zFORp_stream_set_execution(stream, execution_policy)

  Wrapper for :c:func:`zfp_stream_set_execution`

  :p zFORp_stream stream [in]: Compressed stream
  :p integer execution_policy [in]: :ref:`zFORp_exec <zforp_exec>` enum indicating desired execution policy
  :r is_success: Indicate whether execution policy was successfully set (1) or not (0)
  :rtype is_success: integer

----

.. f:function:: zFORp_stream_set_omp_threads(stream, thread_count)

  Wrapper for :c:func:`zfp_stream_set_omp_threads`

  :p zFORp_stream stream [in]: Compressed stream
  :p integer thread_count [in]: Desired number of OpenMP threads
  :r is_success: Indicate whether number of threads was successfully set (1) or not (0)
  :rtype is_success: integer

----

.. f:function:: zFORp_stream_set_omp_chunk_size(stream, chunk_size)

  Wrapper for :c:func:`zfp_stream_set_omp_chunk_size`

  :p zFORp_stream stream [in]: Compressed stream
  :p integer chunk_size [in]: Desired chunk size, in blocks
  :r is_success: Indicate whether chunk size was successfully set (1) or not (0)
  :rtype is_success: integer

Array Metadata
^^^^^^^^^^^^^^

.. f:function:: zFORp_field_alloc()

  Wrapper for :c:func:`zfp_field_alloc`

  :r field: Newly allocated field
  :rtype field: zFORp_field

----

.. f:function:: zFORp_field_1d(uncompressed_ptr, scalar_type, nx)

  Wrapper for :c:func:`zfp_field_1d`

  :p c_ptr uncompressed_ptr [in]: Pointer to uncompressed data
  :p integer scalar_type [in]: :ref:`zFORp_type <zforp_type>` enum describing uncompressed scalar type
  :p integer nx [in]: Number of array elements
  :r field: Newly allocated field
  :rtype field: zFORp_field

----

.. f:function:: zFORp_field_2d(uncompressed_ptr, scalar_type, nx, ny)

  Wrapper for :c:func:`zfp_field_2d`

  :p c_ptr uncompressed_ptr [in]: Pointer to uncompressed data
  :p integer scalar_type [in]: :ref:`zFORp_type <zforp_type>` enum describing uncompressed scalar type
  :p integer nx [in]: Number of array elements in *x* dimension
  :p integer ny [in]: Number of array elements in *y* dimension
  :r field: Newly allocated field
  :rtype field: zFORp_field

----

.. f:function:: zFORp_field_3d(uncompressed_ptr, scalar_type, nx, ny, nz)

  Wrapper for :c:func:`zfp_field_3d`

  :p c_ptr uncompressed_ptr [in]: Pointer to uncompressed data
  :p integer scalar_type [in]: :ref:`zFORp_type <zforp_type>` enum describing uncompressed scalar type
  :p integer nx [in]: Number of array elements in *x* dimension
  :p integer ny [in]: Number of array elements in *y* dimension
  :p integer nz [in]: Number of array elements in *z* dimension
  :r field: Newly allocated field
  :rtype field: zFORp_field

----

.. f:function:: zFORp_field_4d(uncompressed_ptr, scalar_type, nx, ny, nz, nw)

  Wrapper for :c:func:`zfp_field_4d`

  :p c_ptr uncompressed_ptr [in]: Pointer to uncompressed data
  :p integer scalar_type [in]: :ref:`zFORp_type <zforp_type>` enum describing uncompressed scalar type
  :p integer nx [in]: Number of array elements in *x* dimension
  :p integer ny [in]: Number of array elements in *y* dimension
  :p integer nz [in]: Number of array elements in *z* dimension
  :p integer nw [in]: Number of array elements in *w* dimension
  :r field: Newly allocated field
  :rtype field: zFORp_field

----

.. f:subroutine:: zFORp_field_free(field)

  Wrapper for :c:func:`zfp_field_free`

  :p zFORp_field field [inout]: Field metadata

----

.. f:function:: zFORp_field_pointer(field)

  Wrapper for :c:func:`zfp_field_pointer`

  :p zFORp_field field [in]: Field metadata
  :r arr_ptr: Pointer to raw (uncompressed/decompressed) array
  :rtype arr_ptr: c_ptr

----

.. f:function:: zFORp_field_begin(field)

  Wrapper for :c:func:`zfp_field_begin`

  :p zFORp_field field [in]: Field metadata
  :r begin_ptr: Pointer to lowest memory address spanned by field
  :rtype begin_ptr: c_ptr

----

.. f:function:: zFORp_field_type(field)

  Wrapper for :c:func:`zfp_field_type`

  :p zFORp_field field [in]: Field metadata
  :r scalar_type: :ref:`zFORp_type <zforp_type>` enum describing uncompressed scalar type
  :rtype scalar_type: integer

----

.. f:function:: zFORp_field_precision(field)

  Wrapper for :c:func:`zfp_field_precision`

  :p zFORp_field field [in]: Field metadata
  :r prec: Scalar type precision in number of bits
  :rtype prec: integer

----

.. f:function:: zFORp_field_dimensionality(field)

  Wrapper for :c:func:`zfp_field_dimensionality`

  :p zFORp_field field [in]: Field metadata
  :r dims: Dimensionality of array
  :rtype dims: integer

----

.. f:function:: zFORp_field_size(field, size_arr)

  Wrapper for :c:func:`zfp_field_size`

  :p zFORp_field field [in]: Field metadata
  :p size_arr [inout]: Integer array to write field dimensions into
  :ptype size_arr: integer,dimension(4),target
  :r total_size: Total number of array elements
  :rtype total_size: integer (kind=8)

----

.. f:function:: zFORp_field_size_bytes(field)

  Wrapper for :c:func:`zfp_field_size_bytes`

  :p zFORp_field field [in]: Field metadata
  :r byte_size: Number of bytes spanned by field data including gaps (if any)
  :rtype byte_size: integer (kind=8)

----

.. f:function:: zFORp_field_blocks(field)

  Wrapper for :c:func:`zfp_field_blocks`

  :p zFORp_field field [in]: Field metadata
  :r blocks: Total number of blocks spanned by field
  :rtype blocks: integer (kind=8)

----

.. f:function:: zFORp_field_stride(field, stride_arr)

  Wrapper for :c:func:`zfp_field_stride`

  :p zFORp_field field [in]: Field metadata
  :p stride_arr [inout]: Integer array to write strides into
  :ptype stride_arr: integer,dimension(4),target
  :r is_strided: Indicate whether field is strided (1) or not (0)
  :rtype is_strided: integer

----

.. f:function:: zFORp_field_is_contiguous(field)

  Wrapper for :c:func:`zfp_field_is_contiguous`

  :p zFORp_field field [in]: Field metadata
  :r is_contiguous: Indicate whether field is contiguous (1) or not (0)
  :rtype is_contiguous: integer

----

.. f:function:: zFORp_field_metadata(field)

  Wrapper for :c:func:`zfp_field_metadata`

  :p zFORp_field field [in]: Field metadata
  :r encoded_metadata: Compact encoding of metadata
  :rtype encoded_metadata: integer (kind=8)

----

.. f:subroutine:: zFORp_field_set_pointer(field, arr_ptr)

  Wrapper for :c:func:`zfp_field_set_pointer`

  :p zFORp_field field [in]: Field metadata
  :p c_ptr arr_ptr [in]: Pointer to beginning of uncompressed array

----

.. f:function:: zFORp_field_set_type(field, scalar_type)

  Wrapper for :c:func:`zfp_field_set_type`

  :p zFORp_field field [in]: Field metadata
  :p integer scalar_type: :ref:`zFORp_type <zforp_type>` enum indicating desired scalar type
  :r type_result: :ref:`zFORp_type <zforp_type>` enum indicating actual scalar type
  :rtype type_result: integer

----

.. f:subroutine:: zFORp_field_set_size_1d(field, nx)

  Wrapper for :c:func:`zfp_field_set_size_1d`

  :p zFORp_field field [in]: Field metadata
  :p integer nx [in]: Number of array elements

----

.. f:subroutine:: zFORp_field_set_size_2d(field, nx, ny)

  Wrapper for :c:func:`zfp_field_set_size_2d`

  :p zFORp_field field [in]: Field metadata
  :p integer nx [in]: Number of array elements in *x* dimension
  :p integer ny [in]: Number of array elements in *y* dimension

----

.. f:subroutine:: zFORp_field_set_size_3d(field, nx, ny, nz)

  Wrapper for :c:func:`zfp_field_set_size_3d`

  :p zFORp_field field [in]: Field metadata
  :p integer nx [in]: Number of array elements in *x* dimension
  :p integer ny [in]: Number of array elements in *y* dimension
  :p integer nz [in]: Number of array elements in *z* dimension

----

.. f:subroutine:: zFORp_field_set_size_4d(field, nx, ny, nz, nw)

  Wrapper for :c:func:`zfp_field_set_size_4d`

  :p zFORp_field field [in]: Field metadata
  :p integer nx [in]: Number of array elements in *x* dimension
  :p integer ny [in]: Number of array elements in *y* dimension
  :p integer nz [in]: Number of array elements in *z* dimension
  :p integer nw [in]: Number of array elements in *w* dimension

----

.. f:subroutine:: zFORp_field_set_stride_1d(field, sx)

  Wrapper for :c:func:`zfp_field_set_stride_1d`

  :p zFORp_field field [in]: Field metadata
  :p integer sx [in]: Stride in number of scalars

----

.. f:subroutine:: zFORp_field_set_stride_2d(field, sx, sy)

  Wrapper for :c:func:`zfp_field_set_stride_2d`

  :p zFORp_field field [in]: Field metadata
  :p integer sx [in]: Stride in *x* dimension
  :p integer sy [in]: Stride in *y* dimension

----

.. f:subroutine:: zFORp_field_set_stride_3d(field, sx, sy, sz)

  Wrapper for :c:func:`zfp_field_set_stride_3d`

  :p zFORp_field field [in]: Field metadata
  :p integer sx [in]: Stride in *x* dimension
  :p integer sy [in]: Stride in *y* dimension
  :p integer sz [in]: Stride in *z* dimension

----

.. f:subroutine:: zFORp_field_set_stride_4d(field, sx, sy, sz, sw)

  Wrapper for :c:func:`zfp_field_set_stride_4d`

  :p zFORp_field field [in]: Field metadata
  :p integer sx [in]: Stride in *x* dimension
  :p integer sy [in]: Stride in *y* dimension
  :p integer sz [in]: Stride in *z* dimension
  :p integer sw [in]: Stride in *w* dimension

----

.. f:function:: zFORp_field_set_metadata(field, encoded_metadata)

  Wrapper for :c:func:`zfp_field_set_metadata`

  :p zFORp_field field [in]: Field metadata
  :p encoded_metadata [in]: Compact encoding of metadata
  :ptype encoded_metadata: integer (kind=8)
  :r is_success: Indicate whether metadata was successfully set (1) or not (0)
  :rtype is_success: integer

Compression and Decompression
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. f:function:: zFORp_compress(stream, field)

  Wrapper for :c:func:`zfp_compress`

  :p zFORp_stream stream [in]: Compressed stream
  :p zFORp_field field [in]: Field metadata
  :r bitstream_offset_bytes: Bit stream offset after compression, in bytes, or zero on failure
  :rtype bitstream_offset_bytes: integer (kind=8)

----

.. f:function:: zFORp_decompress(stream, field)

  Wrapper for :c:func:`zfp_decompress`

  :p zFORp_stream stream [in]: Compressed stream
  :p zFORp_field field [in]: Field metadata
  :r bitstream_offset_bytes: Bit stream offset after decompression, in bytes, or zero on failure
  :rtype bitstream_offset_bytes: integer (kind=8)

----

.. f:function:: zFORp_write_header(stream, field, mask)

  Wrapper for :c:func:`zfp_write_header`

  :p zFORp_stream stream [in]: Compressed stream
  :p zFORp_field field [in]: Field metadata
  :p integer mask [in]: :ref:`Bit mask <zforp_header>` indicating which parts of header to write
  :r num_bits_written: Number of header bits written or zero on failure
  :rtype num_bits_written: integer (kind=8)

----

.. f:function:: zFORp_read_header(stream, field, mask)

  Wrapper for :c:func:`zfp_read_header`

  :p zFORp_stream stream [in]: Compressed stream
  :p zFORp_field field [in]: Field metadata
  :p integer mask [in]: :ref:`Bit mask <zforp_header>` indicating which parts of header to read
  :r num_bits_read: Number of header bits read or zero on failure
  :rtype num_bits_read: integer (kind=8)
