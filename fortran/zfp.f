module zFORp_module

  use, intrinsic :: iso_c_binding, only: c_int, c_int64_t, c_size_t, c_double, c_ptr, c_null_ptr
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

  end interface

  ! types

  public :: zFORp_bitstream_type, &
            zFORp_stream_type

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
            zFORp_stream_set_bit_stream, &
            zFORp_stream_set_rate, &
            zFORp_stream_set_precision, &
            zFORp_stream_set_accuracy, &
            zFORp_stream_set_mode, &
            zFORp_stream_set_params

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

end module zFORp_module
