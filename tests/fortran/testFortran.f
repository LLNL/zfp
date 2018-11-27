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

  ! bitstream
  bytes = 256
  allocate(buffer(bytes))
  buffer_c_ptr = c_loc(buffer)
  bitstream = zFORp_bitstream_stream_open(buffer_c_ptr, bytes)

  ! zfp_stream
  zfp_stream = zFORp_stream_open(bitstream)

  queried_bitstream = zFORp_stream_bit_stream(zfp_stream)

  ! deallocations
  call zFORp_stream_close(zfp_stream)
  call zFORp_bitstream_stream_close(queried_bitstream)
  deallocate(buffer)
end program main
