.. index::
   single: zforp
.. _zforp:

Fortran bindings
----------------

.. cpp:namespace:: zfp

ZFP 0.5.5 adds zFORp: A Fortran API providing wrappers around the high-level C
API. Wrappers for compressed arrays will arrive in a future release.

Every high-level C API function can be called from a Fortran wrapper function.
C structs are wrapped as Fortran derived types, each containing a single C
pointer to the C struct in memory. The wrapper functions accept and return
these Fortran types, so users should never need to touch the C pointers.

In addition to the high-level C API, some other functions are necessary to be
able to completely compress or decompress data. These include opening and
closing a bitstream, and rewinding a bitstream. See example code
tests/fortran/testFortran.f for how the Fortran API is used to compress and
decompress data.

Types
-----

.. f:type:: zFORp_bitstream_type

    :f c_ptr object: A C pointer to the instance of :c:type:`bitstream`

.. f:type:: zFORp_stream_type

    :f c_ptr object: A C pointer to the instance of :c:type:`zfp_stream`

.. f:type:: zFORp_field_type

    :f c_ptr object: A C pointer to the instance of :c:type:`zfp_field`

Constants
---------

Enums wrapping :c:type:`zfp_type`

  .. f:variable:: integer zFORp_type_none
  .. f:variable:: integer zFORp_type_int32
  .. f:variable:: integer zFORp_type_int64
  .. f:variable:: integer zFORp_type_float
  .. f:variable:: integer zFORp_type_double

Enums wrapping :c:type:`zfp_mode`

  .. f:variable:: integer zFORp_mode_null
  .. f:variable:: integer zFORp_mode_expert
  .. f:variable:: integer zFORp_mode_fixed_rate
  .. f:variable:: integer zFORp_mode_fixed_precision
  .. f:variable:: integer zFORp_mode_fixed_accuracy
  .. f:variable:: integer zFORp_mode_reversible

Enums wrapping :c:type:`zfp_exec_policy`

  .. f:variable:: integer zFORp_exec_serial
  .. f:variable:: integer zFORp_exec_omp
  .. f:variable:: integer zFORp_exec_cuda

Non-enum constants

  .. f:variable:: integer zFORp_version_major

    Wraps :c:macro:`ZFP_VERSION_MAJOR`

  .. f:variable:: integer zFORp_version_minor

    Wraps :c:macro:`ZFP_VERSION_MINOR`

  .. f:variable:: integer zFORp_version_patch

    Wraps :c:macro:`ZFP_VERSION_PATCH`

  .. f:variable:: integer zFORp_codec_version

    Wraps :c:data:`zfp_codec_version`

  .. f:variable:: integer zFORp_library_version

    Wraps :c:data:`zfp_library_version`

  .. f:variable:: character(len=36) zFORp_version_string

    Wraps :c:data:`zfp_version_string`

  .. f:variable:: integer zFORp_min_bits

    Wraps :c:macro:`ZFP_MIN_BITS`

  .. f:variable:: integer zFORp_max_bits

    Wraps :c:macro:`ZFP_MAX_BITS`

  .. f:variable:: integer zFORp_max_prec

    Wraps :c:macro:`ZFP_MAX_PREC`

  .. f:variable:: integer zFORp_min_exp

    Wraps :c:macro:`ZFP_MIN_EXP`

  .. f:variable:: integer zFORp_header_magic

    Wraps :c:macro:`ZFP_HEADER_MAGIC`

  .. f:variable:: integer zFORp_header_meta

    Wraps :c:macro:`ZFP_HEADER_META`

  .. f:variable:: integer zFORp_header_mode

    Wraps :c:macro:`ZFP_HEADER_MODE`

  .. f:variable:: integer zFORp_header_full

    Wraps :c:macro:`ZFP_HEADER_FULL`

  .. f:variable:: integer zFORp_meta_null

    Wraps :c:macro:`ZFP_META_NULL`

  .. f:variable:: integer zFORp_magic_bits

    Wraps :c:macro:`ZFP_MAGIC_BITS`

  .. f:variable:: integer zFORp_meta_bits

    Wraps :c:macro:`ZFP_META_BITS`

  .. f:variable:: integer zFORp_mode_short_bits

    Wraps :c:macro:`ZFP_MODE_SHORT_BITS`

  .. f:variable:: integer zFORp_mode_long_bits

    Wraps :c:macro:`ZFP_MODE_LONG_BITS`

  .. f:variable:: integer zFORp_header_max_bits

    Wraps :c:macro:`ZFP_HEADER_MAX_BITS`

  .. f:variable:: integer zFORp_mode_short_max

    Wraps :c:macro:`ZFP_MODE_SHORT_MAX`

Bitstream function wrappers
---------------------------

.. f:function:: zFORp_bitstream_stream_open(buffer, bytes)

    Wrapper for :c:func:`stream_open`

    :p type(c_ptr) buffer [in]: Bitstream buffer
    :p integer (kind=8) bytes [in]: Buffer size, in bytes
    :r bs: Bitstream
    :rtype bs: zFORp_bitstream_type

.. f:subroutine:: zFORp_bitstream_stream_close(bs)

    Wrapper for :c:func:`stream_close`

    :p zFORp_bitstream_type bs [inout]: Bitstream

High-level API utility function wrappers
----------------------------------------

.. f:function:: zFORp_type_size(zfp_type)

    Wrapper for :c:func:`zfp_type_size`

    :p integer zfp_type [in]: zFORp_type enum.
    :r type_size: Size of described zfp_type, in bytes, from C-language perspective.
    :rtype type_size: integer

.. f:function:: zFORp_stream_open(bs)

    Wrapper for :c:func:`zfp_stream_open`

    :p zFORp_bitstream_type bs [in]: Bitstream
    :r stream: Newly allocated zfp_stream
    :rtype stream: zFORp_stream_type

.. f:subroutine:: zFORp_stream_close(stream)

    Wrapper for :c:func:`zfp_stream_close`

    :p zFORp_stream_type stream [inout]: Zfp_stream

.. f:function:: zFORp_stream_bit_stream(stream)

    Wrapper for :c:func:`zfp_stream_bit_stream`

    :p zFORp_stream_type stream [in]: Zfp_stream
    :r bs: Bitstream
    :rtype bs: zFORp_bitstream_type

.. f:function:: zFORp_stream_is_reversible(stream)

    Wrapper for :c:func:`zfp_stream_is_reversible`

    :p zFORp_stream_type stream [in]: Zfp_stream
    :r is_reversible: indicate whether reversible mode active (1) or not (0)
    :rtype is_reversible: integer

.. f:function:: zFORp_stream_compression_mode(stream)

    Wrapper for :c:func:`zfp_stream_compression_mode`

    :p zFORp_stream_type stream [in]: Zfp_stream
    :r zfp_mode: zFORp_mode enum
    :rtype zfp_mode: integer

.. f:function:: zFORp_stream_mode(stream)

    Wrapper for :c:func:`zfp_stream_mode`

    :p zFORp_stream_type stream [in]: Zfp_stream
    :r encoded_mode: 64 bit encoded mode
    :rtype encoded_mode: integer (kind=8)

.. f:subroutine:: zFORp_stream_params(stream, minbits, maxbits, maxprec, minexp)

    Wrapper for :c:func:`zfp_stream_params`

    :p zFORp_stream_type stream [in]: Zfp_stream
    :p integer (kind=8) minbits [inout]: minbits
    :p integer (kind=8) maxbits [inout]: maxbits
    :p integer (kind=8) maxprec [inout]: maxprec
    :p integer (kind=8) minexp [inout]: minexp

.. f:function:: zFORp_stream_compressed_size(stream)

    Wrapper for :c:func:`zfp_stream_compressed_size`

    :p zFORp_stream_type stream [in]: Zfp_stream
    :r compressed_size: compressed size
    :rtype compressed_size: integer (kind=8)

.. f:function:: zFORp_stream_maximum_size(stream, field)

    Wrapper for :c:func:`zfp_stream_maximum_size`

    :p zFORp_stream_type stream [in]: Zfp_stream
    :p zFORp_field_type field [in]: Zfp_field
    :r max_size: maximum size
    :rtype max_size: integer (kind=8)

.. f:subroutine:: zFORp_stream_set_bit_stream(stream, bs)

    Wrapper for :c:func:`zfp_stream_set_bit_stream`

    :p zFORp_stream_type stream [in]: Zfp_stream
    :p zFORp_bitstream_type bs [in]: bitstream

.. f:subroutine:: zFORp_stream_set_reversible(stream)

    Wrapper for :c:func:`zfp_stream_set_reversible`

    :p zFORp_stream_type stream [in]: Zfp_stream

.. f:function:: zFORp_stream_set_rate(stream, rate, zfp_type, dims, wra)

    Wrapper for :c:func:`zfp_stream_set_rate`

    :p zFORp_stream_type stream [in]: Zfp_stream
    :p real rate [in]: desired rate
    :p integer zfp_type [in]: enum zfp_type
    :p integer dims [in]: dimensions
    :p integer wra [in]: use write random access?
    :r rate_result: actual set rate
    :rtype rate_result: real

.. f:function:: zFORp_stream_set_precision(stream, prec)

    Wrapper for :c:func:`zfp_stream_set_precision`

    :p zFORp_stream_type stream [in]: Zfp_stream
    :p integer prec [in]: desired precision
    :r prec_result: actual set precision
    :rtype prec_result: integer

.. f:function:: zFORp_stream_set_accuracy(stream, acc)

    Wrapper for :c:func:`zfp_stream_set_accuracy()`

    :p zFORp_stream_type stream [in]: Zfp_stream
    :p real acc: desired accuracy (kind=8)
    :r acc_result: actual set accuracy
    :rtype acc_result: real (kind=8)

.. f:function:: zFORp_stream_set_mode(stream, encoded_mode)

    Wrapper for :c:func:`zfp_stream_set_mode`

    :p zFORp_stream_type stream [in]: Zfp_stream
    :p integer encoded_mode [in]: encoded mode parameter
    :r mode_result: newly set zfp_mode enum on zfp_stream
    :rtype mode_result: integer

.. f:function:: zFORp_stream_set_params(stream, minbits, maxbits, maxprec, minexp)

    Wrapper for :c:func:`zfp_stream_set_params`

    :p zFORp_stream_type stream [in]: Zfp_stream
    :p integer minbits [in]: min num of bits
    :p integer maxbits [in]: max num of bits
    :p integer maxprec [in]: max precision
    :p integer minexp [in]: min exponent
    :r is_success: indicate whether parameters were successfully set (1) or not (0)
    :rtype is_success: integer

High-level API: execution policy function wrappers
--------------------------------------------------

  .. f:function:: zFORp_stream_execution(stream)

      Wrapper for :c:func:`zfp_stream_execution`

      :p zFORp_stream_type stream [in]: Zfp_stream
      :r execution_policy: enum of active execution policy
      :rtype execution_policy: integer

  .. f:function:: zFORp_stream_omp_threads(stream)

      Wrapper for :c:func:`zfp_stream_omp_threads`

      :p zFORp_stream_type stream [in]: Zfp_stream
      :r thread_count: number of threads to use upon execution
      :rtype thread_count: integer

  .. f:function:: zFORp_stream_omp_chunk_size(stream)

      Wrapper for :c:func:`zfp_stream_omp_chunk_size`

      :p zFORp_stream_type stream [in]: Zfp_stream
      :r chunk_size_blocks: specified chunk size, in blocks
      :rtype chunk_size_blocks: integer (kind=8)

  .. f:function:: zFORp_stream_set_execution(stream, execution_policy)

      Wrapper for :c:func:`zfp_stream_set_execution`

      :p zFORp_stream_type stream [in]: Zfp_stream
      :p integer execution_policy [in]: desired execution policy (enum)
      :r is_success: indicate whether execution policy was successfully set or not
      :rtype is_success: integer

  .. f:function:: zFORp_stream_set_omp_threads(stream, thread_count)

      Wrapper for :c:func:`zfp_stream_set_omp_threads`

      :p zFORp_stream_type stream [in]: Zfp_stream
      :p integer thread_count [in]: desired number of threads
      :r is_success: indicate whether number of threads successfully set or not
      :rtype is_success: integer

  .. f:function:: zFORp_stream_set_omp_chunk_size(stream, chunk_size)

      Wrapper for :c:func:`zfp_stream_set_omp_chunk_size`

      :p zFORp_stream_type stream [in]: Zfp_stream
      :p integer chunk_size [in]: desired chunk size, in blocks
      :r is_success: indicate whether chunk size successfully set or not
      :rtype is_success: integer

High-level API: zfp_field function wrappers
-------------------------------------------

  .. f:function:: zFORp_field_alloc()

      Wrapper for :c:func:`zfp_field_alloc`

      :r field: newly allocated zfp_field
      :rtype field: zFORp_field_type

  .. f:function:: zFORp_field_1d(uncompressed_ptr, zfp_type, nx)

      Wrapper for :c:func:`zfp_field_1d`

      :p type(c_ptr) uncompressed_ptr [in]: pointer to uncompressed data
      :p integer zfp_type [in]: zfp_type enum describing uncompressed data type
      :p integer nx [in]: number of elements in uncompressed data array
      :r field: newly allocated zfp_field
      :rtype field: zFORp_field_type

  .. f:function:: zFORp_field_2d(uncompressed_ptr, zfp_type, nx, ny)

      Wrapper for :c:func:`zfp_field_2d`

      :p type(c_ptr) uncompressed_ptr [in]: pointer to uncompressed data
      :p integer zfp_type [in]: zfp_type enum describing uncompressed data type
      :p integer nx [in]: number of elements in uncompressed data array's x dimension
      :p integer ny [in]: number of elements in uncompressed data array's y dimension
      :r field: newly allocated zfp_field
      :rtype field: zFORp_field_type

  .. f:function:: zFORp_field_3d(uncompressed_ptr, zfp_type, nx, ny, nz)

      Wrapper for :c:func:`zfp_field_3d`

      :p type(c_ptr) uncompressed_ptr [in]: pointer to uncompressed data
      :p integer zfp_type [in]: zfp_type enum describing uncompressed data type
      :p integer nx [in]: number of elements in uncompressed data array's x dimension
      :p integer ny [in]: number of elements in uncompressed data array's y dimension
      :p integer nz [in]: number of elements in uncompressed data array's z dimension
      :r field: newly allocated zfp_field
      :rtype field: zFORp_field_type

  .. f:function:: zFORp_field_4d(uncompressed_ptr, zfp_type, nx, ny, nz, nw)

      Wrapper for :c:func:`zfp_field_4d`

      :p type(c_ptr) uncompressed_ptr [in]: pointer to uncompressed data
      :p integer zfp_type [in]: zfp_type enum describing uncompressed data type
      :p integer nx [in]: number of elements in uncompressed data array's x dimension
      :p integer ny [in]: number of elements in uncompressed data array's y dimension
      :p integer nz [in]: number of elements in uncompressed data array's z dimension
      :p integer nw [in]: number of elements in uncompressed data array's w dimension
      :r field: newly allocated zfp_field
      :rtype field: zFORp_field_type

  .. f:subroutine:: zFORp_field_free(field)

      Wrapper for :c:func:`zfp_field_free`

      :p zFORp_field_type field [inout]: Zfp_field

  .. f:function:: zFORp_field_pointer(field)

      Wrapper for :c:func:`zfp_field_pointer`

      :p zFORp_field_type field [in]: Zfp_field
      :r arr_ptr: pointer to raw (uncompressed/decompressed) array
      :rtype arr_ptr: type(c_ptr)

  .. f:function:: zFORp_field_scalar_type(field)

      Wrapper for :c:func:`zfp_field_type`

      :p zFORp_field_type field [in]: Zfp_field
      :r zfp_type: zfp_type enum describing field data
      :rtype zfp_type: integer

  .. f:function:: zFORp_field_precision(field)

      Wrapper for :c:func:`zfp_field_precision`

      :p zFORp_field_type field [in]: Zfp_field
      :r prec: type precision describing field data
      :rtype prec: integer

  .. f:function:: zFORp_field_dimensionality(field)

      Wrapper for :c:func:`zfp_field_dimensionality`

      :p zFORp_field_type field [in]: Zfp_field
      :r dims: dimensionality of field data
      :rtype dims: integer

  .. f:function:: zFORp_field_size(field, size_arr)

      Wrapper for :c:func:`zfp_field_size`

      :p zFORp_field_type field [in]: Zfp_field
      :p integer size_arr [inout]: integer array to write field dimensions into
      :r total_size: total number of elements in field
      :rtype total_size: integer (kind=8)

  .. f:function:: zFORp_field_stride(field, stride_arr)

      Wrapper for :c:func:`zfp_field_stride`

      :p zFORp_field_type field [in]: Zfp_field
      :p integer stride_arr [inout]: integer array to write strides into
      :r is_strided: indicate whether field is strided or not
      :rtype is_strided: integer

  .. f:function:: zFORp_field_metadata(field)

      Wrapper for :c:func:`zfp_field_metadata`

      :p zFORp_field_type field [in]: Zfp_field
      :r encoded_metadata: encoded metadata of field
      :rtype encoded_metadata: integer (kind=8)

  .. f:subroutine:: zFORp_field_set_pointer(field, arr_ptr)

      Wrapper for :c:func:`zfp_field_set_pointer`

      :p zFORp_field_type field [in]: Zfp_field
      :p type(c_ptr) arr_ptr [in]: pointer to raw array

  .. f:function:: zFORp_field_set_type(field, zfp_type)

      Wrapper for :c:func:`zfp_field_set_type`

      :p zFORp_field_type field [in]: Zfp_field
      :p integer zfp_type: desired zfp_type enum
      :r zfp_type_result: new zfp_type on the field
      :rtype zfp_type_result: integer

  .. f:subroutine:: zFORp_field_set_size_1d(field, nx)

      Wrapper for :c:func:`zfp_field_set_size_1d`

      :p zFORp_field_type field [in]: Zfp_field
      :p integer nx [in]: number of elements in data array

  .. f:subroutine:: zFORp_field_set_size_2d(field, nx, ny)

      Wrapper for :c:func:`zfp_field_set_size_2d`

      :p zFORp_field_type field [in]: Zfp_field
      :p integer nx [in]: number of elements in data array's x dimension
      :p integer ny [in]: number of elements in data array's y dimension

  .. f:subroutine:: zFORp_field_set_size_3d(field, nx, ny, nz)

      Wrapper for :c:func:`zfp_field_set_size_3d`

      :p zFORp_field_type field [in]: Zfp_field
      :p integer nx [in]: number of elements in data array's x dimension
      :p integer ny [in]: number of elements in data array's y dimension
      :p integer nz [in]: number of elements in data array's z dimension

  .. f:subroutine:: zFORp_field_set_size_4d(field, nx, ny, nz, nw)

      Wrapper for :c:func:`zfp_field_set_size_4d`

      :p zFORp_field_type field [in]: Zfp_field
      :p integer nx [in]: number of elements in data array's x dimension
      :p integer ny [in]: number of elements in data array's y dimension
      :p integer nz [in]: number of elements in data array's z dimension
      :p integer nw [in]: number of elements in data array's w dimension

  .. f:subroutine:: zFORp_field_set_stride_1d(field, sx)

      Wrapper for :c:func:`zfp_field_set_stride_1d`

      :p zFORp_field_type field [in]: Zfp_field
      :p integer sx [in]: stride of data array's x dimension

  .. f:subroutine:: zFORp_field_set_stride_2d(field, sx, sy)

      Wrapper for :c:func:`zfp_field_set_stride_2d`

      :p zFORp_field_type field [in]: Zfp_field
      :p integer sx [in]: stride of data array's x dimension
      :p integer sy [in]: stride of data array's y dimension

  .. f:subroutine:: zFORp_field_set_stride_3d(field, sx, sy, sz)

      Wrapper for :c:func:`zfp_field_set_stride_3d`

      :p zFORp_field_type field [in]: Zfp_field
      :p integer sx [in]: stride of data array's x dimension
      :p integer sy [in]: stride of data array's y dimension
      :p integer sz [in]: stride of data array's z dimension

  .. f:subroutine:: zFORp_field_set_stride_4d(field, sx, sy, sz, sw)

      Wrapper for :c:func:`zfp_field_set_stride_4d`

      :p zFORp_field_type field [in]: Zfp_field
      :p integer sx [in]: stride of data array's x dimension
      :p integer sy [in]: stride of data array's y dimension
      :p integer sz [in]: stride of data array's z dimension
      :p integer sw [in]: stride of data array's w dimension

  .. f:function:: zFORp_field_set_metadata(field, encoded_metadata)

      Wrapper for :c:func:`zfp_field_set_metadata`

      :p zFORp_field_type field [in]: Zfp_field
      :p integer encoded_metadata [in]: encoded metadata (kind=8)
      :r is_success: indicate whether metadata successfully set on field or not
      :rtype is_success: integer

High-level API: compression, decompression, header wrappers
-----------------------------------------------------------

  .. f:function:: zFORp_compress(stream, field)

      Wrapper for :c:func:`zfp_compress`

      :p zFORp_stream_type stream [in]: Zfp_stream
      :p zFORp_field_type field [in]: Zfp_field
      :r bitstream_offset_bytes: bitstream offset after compression, in bytes
      :rtype bitstream_offset_bytes: integer (kind=8)

  .. f:function:: zFORp_decompress(stream, field)

      Wrapper for :c:func:`zfp_decompress`

      :p zFORp_stream_type stream [in]: Zfp_stream
      :p zFORp_field_type field [in]: Zfp_field
      :r bitstream_offset_bytes: bitstream offset after decompression, in bytes
      :rtype bitstream_offset_bytes: integer (kind=8)

  .. f:function:: zFORp_write_header(stream, field, mask)

      Wrapper for :c:func:`zfp_write_header`

      :p zFORp_stream_type stream [in]: Zfp_stream
      :p zFORp_field_type field [in]: Zfp_field
      :p integer mask [in]: indicates header level of detail
      :r num_bits_written: number of bits successfully written in header
      :rtype num_bits_written: integer (kind=8)

  .. f:function:: zFORp_read_header(stream, field, mask)

      Wrapper for :c:func:`zfp_read_header`

      :p zFORp_stream_type stream [in]: Zfp_stream
      :p zFORp_field_type field [in]: Zfp_field
      :p integer mask [in]: indicates header level of detail
      :r num_bits_read: number of bits successfully read in header
      :rtype num_bits_read: integer (kind=8)

Low-level API: stream manipulation wrappers
-------------------------------------------

  .. f:subroutine:: zFORp_stream_rewind(stream)

      Wrapper for :c:func:`zfp_stream_rewind`

      :p zFORp_stream_type stream [in]: Zfp_stream
