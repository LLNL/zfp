# compile tests
if(NOT DEFINED ZFP_ENDTOEND_TESTS_ONLY)
  add_subdirectory(inline)
  add_subdirectory(misc)
  add_subdirectory(encode)
  add_subdirectory(decode)
  if(ZFP_WITH_OPENMP)
    add_subdirectory(share)
  endif()
endif()

add_subdirectory(endtoend)
