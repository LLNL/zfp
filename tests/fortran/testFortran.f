program main
  use zfp
  use iso_c_binding

  ! loop counters
  integer i, j

  ! input/decompressed arrays
  integer xLen, yLen
  integer, dimension(:, :), allocatable, target :: input_array
  integer, dimension(:, :), allocatable, target :: decompressed_array
  type(c_ptr) :: array_c_ptr
  integer error, max_abs_error

  ! zfp_field
  type(zFORp_field) :: field

  ! bitstream
  character, dimension(:), allocatable, target :: buffer
  type(c_ptr) :: buffer_c_ptr
  integer (kind=8) buffer_size_bytes, bitstream_offset_bytes
  type(zFORp_bitstream) :: bitstream, queried_bitstream

  ! zfp_stream
  type(zFORp_stream) :: stream
  real (kind=8) :: desired_rate, rate_result
  integer :: dims, wra
  integer :: zfp_type

  ! initialize input and decompressed arrays
  xLen = 8
  yLen = 8
  allocate(input_array(xLen, yLen))
  do i = 1, xLen
    do j = 1, yLen
      input_array(i, j) = i * i + j * (j + 1)
    enddo
  enddo

  allocate(decompressed_array(xLen, yLen))

  ! setup zfp_field
  array_c_ptr = c_loc(input_array)
  zfp_type = zFORp_type_int32
  field = zFORp_field_2d(array_c_ptr, zfp_type, xLen, yLen)

  ! setup bitstream
  buffer_size_bytes = 256
  allocate(buffer(buffer_size_bytes))
  buffer_c_ptr = c_loc(buffer)
  bitstream = zFORp_bitstream_stream_open(buffer_c_ptr, buffer_size_bytes)

  ! setup zfp_stream
  stream = zFORp_stream_open(bitstream)

  desired_rate = 8.0
  dims = 2
  wra = 0
  zfp_type = zFORp_type_float
  rate_result = zFORp_stream_set_rate(stream, desired_rate, zfp_type, dims, wra)

  queried_bitstream = zFORp_stream_bit_stream(stream)

  ! compress
  bitstream_offset_bytes = zFORp_compress(stream, field)
  write(*, *) "After compression, bitstream offset at "
  write(*, *) bitstream_offset_bytes

  ! decompress
  call zFORp_stream_rewind(stream)
  array_c_ptr = c_loc(decompressed_array)
  call zFORp_field_set_pointer(field, array_c_ptr)

  bitstream_offset_bytes = zFORp_decompress(stream, field)
  write(*, *) "After decompression, bitstream offset at "
  write(*, *) bitstream_offset_bytes

  max_abs_error = 0
  do i = 1, xLen
    do j = 1, yLen
      error = abs(decompressed_array(i, j) - input_array(i, j))
      max_abs_error = max(error, max_abs_error)
    enddo
  enddo
  write(*, *) "Max absolute error: "
  write(*, *) max_abs_error

  write(*, *) "Absolute errors: "
  write(*, *) abs(input_array - decompressed_array)

  ! zfp library info
  write(*, *) zFORp_version_string
  write(*, *) zFORp_meta_null

  ! deallocations
  call zFORp_stream_close(stream)
  call zFORp_bitstream_stream_close(queried_bitstream)
  call zFORp_field_free(field)

  deallocate(buffer)
  deallocate(input_array)
  deallocate(decompressed_array)
end program main
