# compile tests
add_subdirectory(inline)
add_subdirectory(misc)
add_subdirectory(encode)
add_subdirectory(decode)
add_subdirectory(endtoend)
if(ZFP_WITH_OPENMP)
  add_subdirectory(share)
endif()
