function(zfp_add_cpp_tests dims type bits)
  # test compressed array class
  set(test_name testArray${dims}${type})
  add_executable(${test_name} ${test_name}.cpp)
  target_link_libraries(${test_name}
    gtest gtest_main zfp zfpHashLib genSmoothRandNumsLib zfpChecksumsLib)
  add_test(NAME ${test_name} COMMAND ${test_name})

  # test class's references
  set(test_name testArray${dims}${type}Refs)
  add_executable(${test_name} ${test_name}.cpp)
  target_link_libraries(${test_name}
    gtest gtest_main zfp zfpHashLib rand${bits}Lib)
  add_test(NAME ${test_name} COMMAND ${test_name})

  # test class's pointers
  set(test_name testArray${dims}${type}Ptrs)
  add_executable(${test_name} ${test_name}.cpp)
  target_link_libraries(${test_name}
    gtest gtest_main zfp)
  add_test(NAME ${test_name} COMMAND ${test_name})

  # test class's iterators
  set(test_name testArray${dims}${type}Iters)
  add_executable(${test_name} ${test_name}.cpp)
  target_link_libraries(${test_name}
    gtest gtest_main zfp)
  add_test(NAME ${test_name} COMMAND ${test_name})

  # test class's views
  set(test_name testArray${dims}${type}Views)
  add_executable(${test_name} ${test_name}.cpp)
  target_link_libraries(${test_name}
    gtest gtest_main zfp)
  add_test(NAME ${test_name} COMMAND ${test_name})

endfunction()

zfp_add_cpp_tests(1 f 32)
zfp_add_cpp_tests(2 f 32)
zfp_add_cpp_tests(3 f 32)
zfp_add_cpp_tests(1 d 64)
zfp_add_cpp_tests(2 d 64)
zfp_add_cpp_tests(3 d 64)

add_subdirectory(zfp)
