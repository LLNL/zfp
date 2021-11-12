function(zfp_add_cpp_tests dims type bits)

  # test compressed const array class
  set(test_name testConstArray${dims}${type})
  add_executable(${test_name} ${test_name}.cpp)
  target_link_libraries(${test_name}
    gtest gtest_main zfp zfpHashLib genSmoothRandNumsLib zfpChecksumsLib)
  target_compile_definitions(${test_name} PRIVATE ${zfp_compressed_array_defs})
  add_test(NAME ${test_name} COMMAND ${test_name})

endfunction()

zfp_add_cpp_tests(1 f 32)
zfp_add_cpp_tests(2 f 32)
zfp_add_cpp_tests(3 f 32)
zfp_add_cpp_tests(4 f 32)
zfp_add_cpp_tests(1 d 64)
zfp_add_cpp_tests(2 d 64)
zfp_add_cpp_tests(3 d 64)
zfp_add_cpp_tests(4 d 64)
