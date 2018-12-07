module zFORp_module

  use, intrinsic :: iso_c_binding, only: c_int, c_int64_t, c_size_t, c_double, c_ptr, c_null_ptr, c_loc
  implicit none
  private

  type zFORp_bitstream_type
    private
    type(c_ptr) :: object = c_null_ptr
  end type zFORp_bitstream_type

  type zFORp_stream_type
    private
    type(c_ptr) :: object = c_null_ptr
  end type zFORp_stream_type

  type zFORp_field_type
    private
    type(c_ptr) :: object = c_null_ptr
  end type zFORp_field_type

  enum, bind(c)
    enumerator :: zFORp_type_none = 0, &
                  zFORp_type_int32 = 1, &
                  zFORp_type_int64 = 2, &
                  zFORp_type_float = 3, &
                  zFORp_type_double = 4
  end enum

  enum, bind(c)
    enumerator :: zFORp_mode_null = 0, &
                  zFORp_mode_expert = 1, &
                  zFORp_mode_fixed_rate = 2, &
                  zFORp_mode_fixed_precision = 3, &
                  zFORp_mode_fixed_accuracy = 4
  end enum

  interface

    ! minimal bitstream API

    function zfp_bitstream_stream_open(buffer, bytes) result(bitstream) bind(c, name="stream_open")
      import
      type(c_ptr), value :: buffer
      integer(c_int), value :: bytes
      type(c_ptr) :: bitstream
    end function zfp_bitstream_stream_open

    subroutine zfp_bitstream_stream_close(bitstream) bind(c, name="stream_close")
      import
      type(c_ptr), value :: bitstream
    end subroutine

    ! high-level API: utility functions

    function zfp_type_size(zfp_type) result(type_size) bind(c, name="zfp_type_size")
      import
      integer(c_int) zfp_type
      integer(c_size_t) type_size
    end function

    ! high-level API: zfp_stream functions

    function zfp_stream_open(bitstream) result(zfp_stream) bind(c, name="zfp_stream_open")
      import
      type(c_ptr), value :: bitstream
      type(c_ptr) :: zfp_stream
    end function zfp_stream_open

    subroutine zfp_stream_close(zfp_stream) bind(c, name="zfp_stream_close")
      import
      type(c_ptr), value :: zfp_stream
    end subroutine

    function zfp_stream_bit_stream(zfp_stream) result(bitstream) bind(c, name="zfp_stream_bit_stream")
      import
      type(c_ptr), value :: zfp_stream
      type(c_ptr) :: bitstream
    end function

    function zfp_stream_compression_mode(zfp_stream) result(zfp_mode) bind(c, name="zfp_stream_compression_mode")
      import
      type(c_ptr), value :: zfp_stream
      integer(c_int) :: zfp_mode
    end function

    function zfp_stream_mode(zfp_stream) result(encoded_mode) bind(c, name="zfp_stream_mode")
      import
      type(c_ptr), value :: zfp_stream
      integer(c_int64_t) encoded_mode
    end function

    subroutine zfp_stream_params(zfp_stream, minbits, maxbits, maxprec, minexp) bind(c, name="zfp_stream_params")
      import
      type(c_ptr), value :: zfp_stream
      integer(c_int) :: minbits, maxbits, maxprec, minexp
    end subroutine

    function zfp_stream_compressed_size(zfp_stream) result(compressed_size) bind(c, name="zfp_stream_compressed_size")
      import
      type(c_ptr), value :: zfp_stream
      integer(c_size_t) compressed_size
    end function

    function zfp_stream_maximum_size(zfp_stream, zfp_field) result(max_size) bind(c, name="zfp_stream_maximum_size")
      import
      type(c_ptr), value :: zfp_stream, zfp_field
      integer(c_size_t) max_size
    end function

    subroutine zfp_stream_set_bit_stream(zfp_stream, bitstream) bind(c, name="zfp_stream_set_bit_stream")
      import
      type(c_ptr), value :: zfp_stream, bitstream
    end subroutine

    function zfp_stream_set_rate(zfp_stream, rate, zfp_type, dims, wra) result(rate_result) bind(c, name="zfp_stream_set_rate")
      import
      type(c_ptr), value :: zfp_stream
      real(c_double), value :: rate
      integer(c_int), value :: zfp_type
      ! no unsigned int in Fortran
      integer(c_int), value :: dims, wra
      real(c_double) :: rate_result
    end function

    function zfp_stream_set_precision(zfp_stream, prec) result(prec_result) bind(c, name="zfp_stream_set_precision")
      import
      type(c_ptr), value :: zfp_stream
      integer(c_int), value :: prec
      integer(c_int) prec_result
    end function

    function zfp_stream_set_accuracy(zfp_stream, acc) result(acc_result) bind(c, name="zfp_stream_set_accuracy")
      import
      type(c_ptr), value :: zfp_stream
      real(c_double), value :: acc
      real(c_double) acc_result
    end function

    function zfp_stream_set_mode(zfp_stream, encoded_mode) result(mode_result) bind(c, name="zfp_stream_set_mode")
      import
      type(c_ptr), value :: zfp_stream
      integer(c_int64_t), value :: encoded_mode
      integer(c_int) mode_result
    end function

    function zfp_stream_set_params(zfp_stream, minbits, maxbits, maxprec, minexp) &
        result(is_success) bind(c, name="zfp_stream_set_params")
      import
      type(c_ptr), value :: zfp_stream
      integer(c_int), value :: minbits, maxbits, maxprec, minexp
      integer(c_int) is_success
    end function

    ! high-level API: execution policy functions

    function zfp_stream_execution(zfp_stream) result(execution_policy) bind(c, name="zfp_stream_execution")
      import
      type(c_ptr), value :: zfp_stream
      integer(c_int) execution_policy
    end function

    function zfp_stream_omp_threads(zfp_stream) result(num_threads) bind(c, name="zfp_stream_omp_threads")
      import
      type(c_ptr), value :: zfp_stream
      integer(c_int) num_threads
    end function

    function zfp_stream_omp_chunk_size(zfp_stream) result(chunk_size_blocks) bind(c, name="zfp_stream_omp_chunk_size")
      import
      type(c_ptr), value :: zfp_stream
      integer(c_int) chunk_size_blocks
    end function

    function zfp_stream_set_execution(zfp_stream, execution_policy) result(is_success) bind(c, name="zfp_stream_set_execution")
      import
      type(c_ptr), value :: zfp_stream
      integer(c_int) execution_policy, is_success
    end function

    function zfp_stream_set_omp_threads(zfp_stream, threads) result(is_success) bind(c, name="zfp_stream_set_omp_threads")
      import
      type(c_ptr), value :: zfp_stream
      integer(c_int) threads, is_success
    end function

    function zfp_stream_set_omp_chunk_size(zfp_stream, chunk_size) result(is_success) bind(c, name="zfp_stream_set_omp_chunk_size")
      import
      type(c_ptr), value :: zfp_stream
      integer(c_int) chunk_size, is_success
    end function

    ! high-level API: zfp_field functions

    function zfp_field_alloc() result(zfp_field) bind(c, name="zfp_field_alloc")
      import
      type(c_ptr) :: zfp_field
    end function

    function zfp_field_1d(uncompressed_ptr, zfp_type, nx) result(zfp_field) bind(c, name="zfp_field_1d")
      import
      type(c_ptr), value :: uncompressed_ptr
      type(c_ptr) :: zfp_field
      integer(c_int), value :: zfp_type, nx
    end function

    function zfp_field_2d(uncompressed_ptr, zfp_type, nx, ny) result(zfp_field) bind(c, name="zfp_field_2d")
      import
      type(c_ptr), value :: uncompressed_ptr
      type(c_ptr) :: zfp_field
      integer(c_int), value :: zfp_type, nx, ny
    end function

    function zfp_field_3d(uncompressed_ptr, zfp_type, nx, ny, nz) result(zfp_field) bind(c, name="zfp_field_3d")
      import
      type(c_ptr), value :: uncompressed_ptr
      type(c_ptr) :: zfp_field
      integer(c_int), value :: zfp_type, nx, ny, nz
    end function

    function zfp_field_4d(uncompressed_ptr, zfp_type, nx, ny, nz, nw) result(zfp_field) bind(c, name="zfp_field_4d")
      import
      type(c_ptr), value :: uncompressed_ptr
      type(c_ptr) :: zfp_field
      integer(c_int), value :: zfp_type, nx, ny, nz, nw
    end function

    subroutine zfp_field_free(zfp_field) bind(c, name="zfp_field_free")
      import
      type(c_ptr), value :: zfp_field
    end subroutine

    function zfp_field_pointer(zfp_field) result(arr_ptr) bind(c, name="zfp_field_pointer")
      import
      type(c_ptr), value :: zfp_field
      type(c_ptr) :: arr_ptr
    end function

    function zfp_field_scalar_type(zfp_field) result(zfp_type) bind(c, name="zfp_field_type")
      import
      type(c_ptr), value :: zfp_field
      integer(c_int) zfp_type
    end function

    function zfp_field_precision(zfp_field) result(prec) bind(c, name="zfp_field_precision")
      import
      type(c_ptr), value :: zfp_field
      integer(c_int) prec
    end function

    function zfp_field_dimensionality(zfp_field) result(dims) bind(c, name="zfp_field_dimensionality")
      import
      type(c_ptr), value :: zfp_field
      integer(c_int) dims
    end function

    function zfp_field_size(zfp_field, size_arr) result(total_size) bind(c, name="zfp_field_size")
      import
      type(c_ptr), value :: zfp_field, size_arr
      integer(c_size_t) total_size
    end function

    function zfp_field_stride(zfp_field, stride_arr) result(is_strided) bind(c, name="zfp_field_stride")
      import
      type(c_ptr), value :: zfp_field, stride_arr
      integer(c_int) is_strided
    end function

    function zfp_field_metadata(zfp_field) result(encoded_metadata) bind(c, name="zfp_field_metadata")
      import
      type(c_ptr), value :: zfp_field
      integer(c_int64_t) encoded_metadata
    end function

    subroutine zfp_field_set_pointer(zfp_field, arr_ptr) bind(c, name="zfp_field_set_pointer")
      import
      type(c_ptr), value :: zfp_field, arr_ptr
    end subroutine

    function zfp_field_set_type(zfp_field, zfp_type) result(zfp_type_result) bind(c, name="zfp_field_set_type")
      import
      type(c_ptr), value :: zfp_field
      integer(c_int) zfp_type, zfp_type_result
    end function

    subroutine zfp_field_set_size_1d(zfp_field, nx) bind(c, name="zfp_field_set_size_1d")
      import
      type(c_ptr), value :: zfp_field
      integer(c_int) nx
    end subroutine

    subroutine zfp_field_set_size_2d(zfp_field, nx, ny) bind(c, name="zfp_field_set_size_2d")
      import
      type(c_ptr), value :: zfp_field
      integer(c_int) nx, ny
    end subroutine

    subroutine zfp_field_set_size_3d(zfp_field, nx, ny, nz) bind(c, name="zfp_field_set_size_3d")
      import
      type(c_ptr), value :: zfp_field
      integer(c_int) nx, ny, nz
    end subroutine

    subroutine zfp_field_set_size_4d(zfp_field, nx, ny, nz, nw) bind(c, name="zfp_field_set_size_4d")
      import
      type(c_ptr), value :: zfp_field
      integer(c_int) nx, ny, nz, nw
    end subroutine

    subroutine zfp_field_set_stride_1d(zfp_field, sx) bind(c, name="zfp_field_set_stride_1d")
      import
      type(c_ptr), value :: zfp_field
      integer(c_int) sx
    end subroutine

    subroutine zfp_field_set_stride_2d(zfp_field, sx, sy) bind(c, name="zfp_field_set_stride_2d")
      import
      type(c_ptr), value :: zfp_field
      integer(c_int) sx, sy
    end subroutine

    subroutine zfp_field_set_stride_3d(zfp_field, sx, sy, sz) bind(c, name="zfp_field_set_stride_3d")
      import
      type(c_ptr), value :: zfp_field
      integer(c_int) sx, sy, sz
    end subroutine

    subroutine zfp_field_set_stride_4d(zfp_field, sx, sy, sz, sw) bind(c, name="zfp_field_set_stride_4d")
      import
      type(c_ptr), value :: zfp_field
      integer(c_int) sx, sy, sz, sw
    end subroutine

    function zfp_field_set_metadata(zfp_field, encoded_metadata) result(is_success) bind(c, name="zfp_field_set_metadata")
      import
      type(c_ptr), value :: zfp_field
      integer(c_int64_t) :: encoded_metadata
      integer(c_int) is_success
    end function

    ! high-level API: compression and decompression

    function zfp_compress(zfp_stream, zfp_field) result(bitstream_offset_bytes) bind(c, name="zfp_compress")
      import
      type(c_ptr), value :: zfp_stream, zfp_field
      integer(c_size_t) :: bitstream_offset_bytes
    end function

    function zfp_decompress(zfp_stream, zfp_field) result(bitstream_offset_bytes) bind(c, name="zfp_decompress")
      import
      type(c_ptr), value :: zfp_stream, zfp_field
      integer(c_size_t) :: bitstream_offset_bytes
    end function

    function zfp_write_header(zfp_stream, zfp_field, mask) result(num_bits_written) bind(c, name="zfp_write_header")
      import
      type(c_ptr), value :: zfp_stream, zfp_field
      integer(c_int) mask
      integer(c_size_t) num_bits_written
    end function

    function zfp_read_header(zfp_stream, zfp_field, mask) result(num_bits_read) bind(c, name="zfp_read_header")
      import
      type(c_ptr), value :: zfp_stream, zfp_field
      integer(c_int) mask
      integer(c_size_t) num_bits_read
    end function

  end interface

  ! types

  public :: zFORp_bitstream_type, &
            zFORp_stream_type, &
            zFORp_field_type

  ! enums

  public :: zFORp_type_none, &
            zFORp_type_int32, &
            zFORp_type_int64, &
            zFORp_type_float, &
            zFORp_type_double

  public :: zFORp_mode_null, &
            zFORp_mode_expert, &
            zFORp_mode_fixed_rate, &
            zFORp_mode_fixed_precision, &
            zFORp_mode_fixed_accuracy

  ! minimal bitstream API

  public :: zFORp_type_size

  public :: zFORp_bitstream_stream_open, &
            zFORp_bitstream_stream_close

  ! high-level API: zfp_stream functions

  public :: zFORp_stream_open, &
            zFORp_stream_close, &
            zFORp_stream_bit_stream, &
            zFORp_stream_compression_mode, &
            zFORp_stream_mode, &
            zFORp_stream_params, &
            zFORp_stream_compressed_size, &
            zFORp_stream_maximum_size, &
            zFORp_stream_set_bit_stream, &
            zFORp_stream_set_rate, &
            zFORp_stream_set_precision, &
            zFORp_stream_set_accuracy, &
            zFORp_stream_set_mode, &
            zFORp_stream_set_params

  ! high-level API: execution policy functions
  public :: zFORp_stream_execution, &
            zFORp_stream_omp_threads, &
            zFORp_stream_omp_chunk_size, &
            zFORp_stream_set_execution, &
            zFORp_stream_set_omp_threads, &
            zFORp_stream_set_omp_chunk_size

  ! high-level API: zfp_field functions

  public :: zFORp_field_alloc, &
            zFORp_field_1d, &
            zFORp_field_2d, &
            zFORp_field_3d, &
            zFORp_field_4d, &
            zFORp_field_free, &
            zFORp_field_pointer, &
            zFORp_field_scalar_type, &
            zFORp_field_precision, &
            zFORp_field_dimensionality, &
            zFORp_field_size, &
            zFORp_field_stride, &
            zFORp_field_metadata, &
            zFORp_field_set_pointer, &
            zFORp_field_set_type, &
            zFORp_field_set_size_1d, &
            zFORp_field_set_size_2d, &
            zFORp_field_set_size_3d, &
            zFORp_field_set_size_4d, &
            zFORp_field_set_stride_1d, &
            zFORp_field_set_stride_2d, &
            zFORp_field_set_stride_3d, &
            zFORp_field_set_stride_4d, &
            zFORp_field_set_metadata

  ! high-level API: compression and decompression

  public :: zFORp_compress, &
            zFORp_decompress, &
            zFORp_write_header, &
            zFORp_read_header

contains

  ! minimal bitstream API

  function zFORp_bitstream_stream_open(buffer, bytes) result(bitstream)
    implicit none
    type(zFORp_bitstream_type) :: bitstream
    type(c_ptr), intent(in) :: buffer
    integer, intent(in) :: bytes
    bitstream%object = zfp_bitstream_stream_open(buffer, int(bytes, c_int))
  end function zFORp_bitstream_stream_open

  subroutine zFORp_bitstream_stream_close(bitstream)
    type(zFORp_bitstream_type), intent(inout) :: bitstream
    call zfp_bitstream_stream_close(bitstream%object)
    bitstream%object = c_null_ptr
  end subroutine zFORp_bitstream_stream_close

  ! high-level API: utility functions

  function zFORp_type_size(zfp_type) result(type_size)
    implicit none
    integer, intent(in) :: zfp_type
    integer type_size
    type_size = zfp_type_size(int(zfp_type, c_int))
  end function zFORp_type_size

  ! high-level API: zfp_stream functions

  function zFORp_stream_open(bitstream) result(zfp_stream)
    implicit none
    type(zFORp_bitstream_type), intent(in) :: bitstream
    type(zFORp_stream_type) :: zfp_stream
    zfp_stream%object = zfp_stream_open(bitstream%object)
  end function zFORp_stream_open

  subroutine zFORp_stream_close(zfp_stream)
    type(zFORp_stream_type), intent(inout) :: zfp_stream
    call zfp_stream_close(zfp_stream%object)
    zfp_stream%object = c_null_ptr
  end subroutine zFORp_stream_close

  function zFORp_stream_bit_stream(zfp_stream) result(bitstream)
    implicit none
    type(zFORp_stream_type), intent(in) :: zfp_stream
    type(zFORp_bitstream_type) :: bitstream
    bitstream%object = zfp_stream_bit_stream(zfp_stream%object)
  end function zFORp_stream_bit_stream

  function zFORp_stream_compression_mode(zfp_stream) result(zfp_mode)
    implicit none
    type(zFORp_stream_type), intent(in) :: zfp_stream
    integer zfp_mode
    zfp_mode = zfp_stream_compression_mode(zfp_stream%object)
  end function zFORp_stream_compression_mode

  function zFORp_stream_mode(zfp_stream) result(encoded_mode)
    implicit none
    type(zFORp_stream_type), intent(in) :: zfp_stream
    integer (kind=8) encoded_mode
    encoded_mode = zfp_stream_mode(zfp_stream%object)
  end function zFORp_stream_mode

  subroutine zFORp_stream_params(zfp_stream, minbits, maxbits, maxprec, minexp)
    type(zFORp_stream_type), intent(in) :: zfp_stream
    integer, intent(inout) :: minbits, maxbits, maxprec, minexp
    call zfp_stream_params(zfp_stream%object, &
                           int(minbits, c_int), &
                           int(maxbits, c_int), &
                           int(maxprec, c_int), &
                           int(minexp, c_int))
  end subroutine zFORp_stream_params

  function zFORp_stream_compressed_size(zfp_stream) result(compressed_size)
    implicit none
    type(zFORp_stream_type), intent(in) :: zfp_stream
    integer compressed_size
    compressed_size = zfp_stream_compressed_size(zfp_stream%object)
  end function zFORp_stream_compressed_size

  function zFORp_stream_maximum_size(zfp_stream, zfp_field) result(max_size)
    implicit none
    type(zFORp_stream_type), intent(in) :: zfp_stream
    type(zFORp_field_type), intent(in) :: zfp_field
    integer max_size
    max_size = zfp_stream_maximum_size(zfp_stream%object, zfp_field%object)
  end function zFORp_stream_maximum_size

  subroutine zFORp_stream_set_bit_stream(zfp_stream, bitstream)
    type(zFORp_stream_type), intent(in) :: zfp_stream
    type(zFORp_bitstream_type), intent(in) :: bitstream
    call zfp_stream_set_bit_stream(zfp_stream%object, bitstream%object)
  end subroutine zFORp_stream_set_bit_stream

  function zFORp_stream_set_rate(zfp_stream, rate, zfp_type, dims, wra) result(rate_result)
    implicit none
    type(zFORp_stream_type), intent(in) :: zfp_stream
    real, intent(in) :: rate
    integer, intent(in) :: zfp_type
    integer, intent(in) :: dims, wra
    real :: rate_result
    rate_result = zfp_stream_set_rate(zfp_stream%object, real(rate, c_double), &
      int(zfp_type, c_int), int(dims, c_int), int(wra, c_int))
  end function zFORp_stream_set_rate

  function zFORp_stream_set_precision(zfp_stream, prec) result(prec_result)
    implicit none
    type(zFORp_stream_type), intent(in) :: zfp_stream
    integer, intent(in) :: prec
    integer prec_result
    prec_result = zfp_stream_set_precision(zfp_stream%object, int(prec, c_int))
  end function zFORp_stream_set_precision

  function zFORp_stream_set_accuracy(zfp_stream, acc) result(acc_result)
    implicit none
    type(zFORp_stream_type), intent(in) :: zfp_stream
    real (kind=8), intent(in) :: acc
    real (kind=8) acc_result
    acc_result = zfp_stream_set_accuracy(zfp_stream%object, real(acc, c_double))
  end function zFORp_stream_set_accuracy

  function zFORp_stream_set_mode(zfp_stream, encoded_mode) result(mode_result)
    implicit none
    type(zFORp_stream_type), intent(in) :: zfp_stream
    integer, intent(in) :: encoded_mode
    integer mode_result
    mode_result = zfp_stream_set_mode(zfp_stream%object, int(encoded_mode, c_int64_t))
  end function zFORp_stream_set_mode

  function zFORp_stream_set_params(zfp_stream, minbits, maxbits, maxprec, minexp) result(is_success)
    implicit none
    type(zFORp_stream_type), intent(in) :: zfp_stream
    integer, intent(in) :: minbits, maxbits, maxprec, minexp
    integer is_success
    is_success = zfp_stream_set_params(zfp_stream%object, &
                                       int(minbits, c_int), &
                                       int(maxbits, c_int), &
                                       int(maxprec, c_int), &
                                       int(minexp, c_int))
  end function zFORp_stream_set_params

  ! high-level API: execution policy functions

  function zFORp_stream_execution(zfp_stream) result(execution_policy)
    implicit none
    type(zFORp_stream_type), intent(in) :: zfp_stream
    integer execution_policy
    execution_policy = zfp_stream_execution(zfp_stream%object)
  end function zFORp_stream_execution

  function zFORp_stream_omp_threads(zfp_stream) result(thread_count)
    implicit none
    type(zFORp_stream_type), intent(in) :: zfp_stream
    integer thread_count
    thread_count = zfp_stream_omp_threads(zfp_stream%object)
  end function zFORp_stream_omp_threads

  function zFORp_stream_omp_chunk_size(zfp_stream) result(chunk_size_blocks)
    implicit none
    type(zFORp_stream_type), intent(in) :: zfp_stream
    integer chunk_size_blocks
    chunk_size_blocks = zfp_stream_omp_chunk_size(zfp_stream%object)
  end function zFORp_stream_omp_chunk_size

  function zFORp_stream_set_execution(zfp_stream, execution_policy) result(is_success)
    implicit none
    type(zFORp_stream_type), intent(in) :: zfp_stream
    integer, intent(in) :: execution_policy
    integer is_success
    is_success = zfp_stream_set_execution(zfp_stream%object, int(execution_policy, c_int))
  end function zFORp_stream_set_execution

  function zFORp_stream_set_omp_threads(zfp_stream, thread_count) result(is_success)
    implicit none
    type(zFORp_stream_type), intent(in) :: zfp_stream
    integer, intent(in) :: thread_count
    integer is_success
    is_success = zfp_stream_set_omp_threads(zfp_stream%object, int(thread_count, c_int))
  end function zFORp_stream_set_omp_threads

  function zFORp_stream_set_omp_chunk_size(zfp_stream, chunk_size) result(is_success)
    implicit none
    type(zFORp_stream_type), intent(in) :: zfp_stream
    integer, intent(in) :: chunk_size
    integer is_success
    is_success = zfp_stream_set_omp_chunk_size(zfp_stream%object, int(chunk_size, c_int))
  end function zFORp_stream_set_omp_chunk_size

  ! high-level API: zfp_field functions

  function zFORp_field_alloc() result(zfp_field)
    implicit none
    type(zFORp_field_type) zfp_field
    zfp_field%object = zfp_field_alloc()
  end function zFORp_field_alloc

  function zFORp_field_1d(uncompressed_ptr, zfp_type, nx) result(zfp_field)
    implicit none
    type(c_ptr), intent(in) :: uncompressed_ptr
    integer, intent(in) :: zfp_type, nx
    type(zFORp_field_type) zfp_field
    zfp_field%object = zfp_field_1d(uncompressed_ptr, int(zfp_type, c_int), &
                                    int(nx, c_int))
  end function zFORp_field_1d

  function zFORp_field_2d(uncompressed_ptr, zfp_type, nx, ny) result(zfp_field)
    implicit none
    type(c_ptr), intent(in) :: uncompressed_ptr
    integer, intent(in) :: zfp_type, nx, ny
    type(zFORp_field_type) zfp_field
    zfp_field%object = zfp_field_2d(uncompressed_ptr, int(zfp_type, c_int), &
                                    int(nx, c_int), int(ny, c_int))
  end function zFORp_field_2d

  function zFORp_field_3d(uncompressed_ptr, zfp_type, nx, ny, nz) result(zfp_field)
    implicit none
    type(c_ptr), intent(in) :: uncompressed_ptr
    integer, intent(in) :: zfp_type, nx, ny, nz
    type(zFORp_field_type) zfp_field
    zfp_field%object = zfp_field_3d(uncompressed_ptr, int(zfp_type, c_int), &
                                    int(nx, c_int), int(ny, c_int), &
                                    int(nz, c_int))
  end function zFORp_field_3d

  function zFORp_field_4d(uncompressed_ptr, zfp_type, nx, ny, nz, nw) result(zfp_field)
    implicit none
    type(c_ptr), intent(in) :: uncompressed_ptr
    integer, intent(in) :: zfp_type, nx, ny, nz, nw
    type(zFORp_field_type) zfp_field
    zfp_field%object = zfp_field_4d(uncompressed_ptr, int(zfp_type, c_int), &
                                    int(nx, c_int), int(ny, c_int), &
                                    int(nz, c_int), int(nw, c_int))
  end function zFORp_field_4d

  subroutine zFORp_field_free(zfp_field)
    type(zFORp_field_type), intent(inout) :: zfp_field
    call zfp_field_free(zfp_field%object)
    zfp_field%object = c_null_ptr
  end subroutine zFORp_field_free

  function zFORp_field_pointer(zfp_field) result(arr_ptr)
    implicit none
    type(zFORp_field_type), intent(in) :: zfp_field
    type(c_ptr) arr_ptr
    arr_ptr = zfp_field_pointer(zfp_field%object)
  end function zFORp_field_pointer

  ! added "scalar" to name to avoid clash with zfp_field_type
  function zFORp_field_scalar_type(zfp_field) result(zfp_type)
    implicit none
    type(zFORp_field_type), intent(in) :: zfp_field
    integer zfp_type
    zfp_type = zfp_field_scalar_type(zfp_field%object)
  end function zFORp_field_scalar_type

  function zFORp_field_precision(zfp_field) result(prec)
    implicit none
    type(zFORp_field_type), intent(in) :: zfp_field
    integer prec
    prec = zfp_field_precision(zfp_field%object)
  end function zFORp_field_precision

  function zFORp_field_dimensionality(zfp_field) result(dims)
    implicit none
    type(zFORp_field_type), intent(in) :: zfp_field
    integer dims
    dims = zfp_field_dimensionality(zfp_field%object)
  end function zFORp_field_dimensionality

  function zFORp_field_size(zfp_field, size_arr) result(total_size)
    implicit none
    type(zFORp_field_type), intent(in) :: zfp_field
    integer, dimension(:), target, intent(inout) :: size_arr
    integer total_size
    total_size = zfp_field_size(zfp_field%object, c_loc(size_arr))
  end function zFORp_field_size

  function zFORp_field_stride(zfp_field, stride_arr) result(is_strided)
    implicit none
    type(zFORp_field_type), intent(in) :: zfp_field
    integer, dimension(:), target, intent(inout) :: stride_arr
    integer is_strided
    is_strided = zfp_field_stride(zfp_field%object, c_loc(stride_arr))
  end function zFORp_field_stride

  function zFORp_field_metadata(zfp_field) result(encoded_metadata)
    implicit none
    type(zFORp_field_type), intent(in) :: zfp_field
    integer (kind=8) encoded_metadata
    encoded_metadata = zfp_field_metadata(zfp_field%object)
  end function zFORp_field_metadata

  subroutine zFORp_field_set_pointer(zfp_field, arr_ptr)
    type(zFORp_field_type), intent(in) :: zfp_field
    type(c_ptr), intent(in) :: arr_ptr
    call zfp_field_set_pointer(zfp_field%object, arr_ptr)
  end subroutine zFORp_field_set_pointer

  function zFORp_field_set_type(zfp_field, zfp_type) result(zfp_type_result)
    implicit none
    type(zFORp_field_type), intent(in) :: zfp_field
    integer, intent(in) :: zfp_type
    integer zfp_type_result
    zfp_type_result = zfp_field_set_type(zfp_field%object, int(zfp_type, c_int))
  end function zFORp_field_set_type

  subroutine zFORp_field_set_size_1d(zfp_field, nx)
    type(zFORp_field_type), intent(in) :: zfp_field
    integer, intent(in) :: nx
    call zfp_field_set_size_1d(zfp_field%object, int(nx, c_int))
  end subroutine zFORp_field_set_size_1d

  subroutine zFORp_field_set_size_2d(zfp_field, nx, ny)
    type(zFORp_field_type), intent(in) :: zfp_field
    integer, intent(in) :: nx, ny
    call zfp_field_set_size_2d(zfp_field%object, int(nx, c_int), int(ny, c_int))
  end subroutine zFORp_field_set_size_2d

  subroutine zFORp_field_set_size_3d(zfp_field, nx, ny, nz)
    type(zFORp_field_type), intent(in) :: zfp_field
    integer, intent(in) :: nx, ny, nz
    call zfp_field_set_size_3d(zfp_field%object, int(nx, c_int), int(ny, c_int), int(nz, c_int))
  end subroutine zFORp_field_set_size_3d

  subroutine zFORp_field_set_size_4d(zfp_field, nx, ny, nz, nw)
    type(zFORp_field_type), intent(in) :: zfp_field
    integer, intent(in) :: nx, ny, nz, nw
    call zfp_field_set_size_4d(zfp_field%object, int(nx, c_int), int(ny, c_int), int(nz, c_int), int(nw, c_int))
  end subroutine zFORp_field_set_size_4d

  subroutine zFORp_field_set_stride_1d(zfp_field, sx)
    type(zFORp_field_type), intent(in) :: zfp_field
    integer, intent(in) :: sx
    call zfp_field_set_stride_1d(zfp_field%object, int(sx, c_int))
  end subroutine zFORp_field_set_stride_1d

  subroutine zFORp_field_set_stride_2d(zfp_field, sx, sy)
    type(zFORp_field_type), intent(in) :: zfp_field
    integer, intent(in) :: sx, sy
    call zfp_field_set_stride_2d(zfp_field%object, int(sx, c_int), int(sy, c_int))
  end subroutine zFORp_field_set_stride_2d

  subroutine zFORp_field_set_stride_3d(zfp_field, sx, sy, sz)
    type(zFORp_field_type), intent(in) :: zfp_field
    integer, intent(in) :: sx, sy, sz
    call zfp_field_set_stride_3d(zfp_field%object, int(sx, c_int), int(sy, c_int), int(sz, c_int))
  end subroutine zFORp_field_set_stride_3d

  subroutine zFORp_field_set_stride_4d(zfp_field, sx, sy, sz, sw)
    type(zFORp_field_type), intent(in) :: zfp_field
    integer, intent(in) :: sx, sy, sz, sw
    call zfp_field_set_stride_4d(zfp_field%object, int(sx, c_int), int(sy, c_int), int(sz, c_int), int(sw, c_int))
  end subroutine zFORp_field_set_stride_4d

  function zFORp_field_set_metadata(zfp_field, encoded_metadata) result(is_success)
    implicit none
    type(zFORp_field_type), intent(in) :: zfp_field
    integer (kind=8), intent(in) :: encoded_metadata
    integer is_success
    is_success = zfp_field_set_metadata(zfp_field%object, int(encoded_metadata, c_int64_t))
  end function zFORp_field_set_metadata

  ! high-level API: compression and decompression

  function zFORp_compress(zfp_stream, zfp_field) result(bitstream_offset_bytes)
    implicit none
    type(zFORp_stream_type), intent(in) :: zfp_stream
    type(zFORp_field_type), intent(in) :: zfp_field
    integer bitstream_offset_bytes
    bitstream_offset_bytes = zfp_compress(zfp_stream%object, zfp_field%object)
  end function zFORp_compress

  function zFORp_decompress(zfp_stream, zfp_field) result(bitstream_offset_bytes)
    implicit none
    type(zFORp_stream_type), intent(in) :: zfp_stream
    type(zFORp_field_type), intent(in) :: zfp_field
    integer bitstream_offset_bytes
    bitstream_offset_bytes = zfp_decompress(zfp_stream%object, zfp_field%object)
  end function zFORp_decompress

  function zFORp_write_header(zfp_stream, zfp_field, mask) result(num_bits_written)
    implicit none
    type(zFORp_stream_type), intent(in) :: zfp_stream
    type(zFORp_field_type), intent(in) :: zfp_field
    integer, intent(in) :: mask
    integer num_bits_written
    num_bits_written = zfp_write_header(zfp_stream%object, zfp_field%object, int(mask, c_int))
  end function zFORp_write_header

  function zFORp_read_header(zfp_stream, zfp_field, mask) result(num_bits_read)
    implicit none
    type(zFORp_stream_type), intent(in) :: zfp_stream
    type(zFORp_field_type), intent(in) :: zfp_field
    integer, intent(in) :: mask
    integer num_bits_read
    num_bits_read = zfp_read_header(zfp_stream%object, zfp_field%object, int(mask, c_int))
  end function zFORp_read_header

end module zFORp_module
