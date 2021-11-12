function(zfp_add_cpp_tests dims type bits)
  # test compressed array class
  set(test_name testArray${dims}${type})
  add_executable(${test_name} ${test_name}.cpp)
  target_link_libraries(${test_name}
    gtest gtest_main zfp zfpHashLib genSmoothRandNumsLib zfpChecksumsLib)
  target_compile_definitions(${test_name} PRIVATE ${zfp_compressed_array_defs})
  add_test(NAME ${test_name} COMMAND ${test_name})

  # test class's references
  set(test_name testArray${dims}${type}Refs)
  add_executable(${test_name} ${test_name}.cpp)
  target_link_libraries(${test_name}
    gtest gtest_main zfp zfpHashLib rand${bits}Lib)
  target_compile_definitions(${test_name} PRIVATE ${zfp_compressed_array_defs})
  add_test(NAME ${test_name} COMMAND ${test_name})

  # test class's pointers
  set(test_name testArray${dims}${type}Ptrs)
  add_executable(${test_name} ${test_name}.cpp)
  target_link_libraries(${test_name}
    gtest gtest_main zfp)
  target_compile_definitions(${test_name} PRIVATE ${zfp_compressed_array_defs})
  add_test(NAME ${test_name} COMMAND ${test_name})

  # test class's iterators
  set(test_name testArray${dims}${type}Iters)
  add_executable(${test_name} ${test_name}.cpp)
  target_link_libraries(${test_name}
    gtest gtest_main zfp)
  target_compile_definitions(${test_name} PRIVATE ${zfp_compressed_array_defs})
  add_test(NAME ${test_name} COMMAND ${test_name})

  # test class's views
  set(test_name testArray${dims}${type}Views)
  add_executable(${test_name} ${test_name}.cpp)
  target_link_libraries(${test_name}
    gtest gtest_main zfp)
  target_compile_definitions(${test_name} PRIVATE ${zfp_compressed_array_defs})
  add_test(NAME ${test_name} COMMAND ${test_name})

  # test class's view pointers
  set(test_name testArray${dims}${type}ViewPtrs)
  add_executable(${test_name} ${test_name}.cpp)
  target_link_libraries(${test_name}
    gtest gtest_main zfp)
  target_compile_definitions(${test_name} PRIVATE ${zfp_compressed_array_defs})
  add_test(NAME ${test_name} COMMAND ${test_name})

  # test class's view iterators
  set(test_name testArray${dims}${type}ViewIters)
  add_executable(${test_name} ${test_name}.cpp)
  target_link_libraries(${test_name}
    gtest gtest_main zfp)
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

# test zfp::array::construct() invalid cases
set(test_name testConstruct)
add_executable(testConstruct testConstruct.cpp)
target_link_libraries(testConstruct gtest gtest_main zfp)
target_compile_definitions(testConstruct PRIVATE ${zfp_compressed_array_defs})
add_test(NAME testConstruct COMMAND testConstruct)
