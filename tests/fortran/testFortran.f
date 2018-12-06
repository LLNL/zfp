program main
  use zFORp_module
  use iso_c_binding

  ! bitstream
  character, dimension(:), allocatable, target :: buffer
  type(c_ptr) :: buffer_c_ptr
  integer bytes
  type(zFORp_bitstream_type) :: bitstream, queried_bitstream

  ! zfp_stream
  type(zFORp_stream_type) :: zfp_stream
  real :: desired_rate, rate_result
  integer :: dims, wra
  integer :: zfp_type

  ! bitstream
  bytes = 256
  allocate(buffer(bytes))
  buffer_c_ptr = c_loc(buffer)
  bitstream = zFORp_bitstream_stream_open(buffer_c_ptr, bytes)

  ! zfp_stream
  zfp_stream = zFORp_stream_open(bitstream)

  desired_rate = 16.0
  dims = 2
  wra = 0
  zfp_type = zFORp_type_float
  rate_result = zFORp_stream_set_rate(zfp_stream, desired_rate, zfp_type, dims, wra)

  queried_bitstream = zFORp_stream_bit_stream(zfp_stream)

  ! deallocations
  call zFORp_stream_close(zfp_stream)
  call zFORp_bitstream_stream_close(queried_bitstream)
  deallocate(buffer)
end program main
