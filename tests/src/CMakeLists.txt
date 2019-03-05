# compile tests
if(NOT DEFINED ZFP_OMP_TESTS_ONLY)
  add_subdirectory(inline)
  add_subdirectory(misc)
  add_subdirectory(encode)
  add_subdirectory(decode)
endif()

add_subdirectory(endtoend)
add_subdirectory(execPolicy)
