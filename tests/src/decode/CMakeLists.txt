function(zfp_add_block_tests dims type bits)
  set(block_test_name testZfpDecodeBlock${dims}d${type})
  add_executable(${block_test_name} ${block_test_name}.c)
  target_link_libraries(${block_test_name}
    cmocka zfp rand${bits}Lib zfpHashLib zfpChecksumsLib)
  if(HAVE_LIBM_MATH)
    target_link_libraries(${block_test_name} m)
  endif()

  target_compile_definitions(${block_test_name} PRIVATE ${zfp_private_defs})
  if(PRINT_CHECKSUMS)
    target_compile_definitions(${block_test_name} PUBLIC PRINT_CHECKSUMS)
  endif()

  add_test(NAME ${block_test_name} COMMAND ${block_test_name})

  set(strided_block_test_name testZfpDecodeBlockStrided${dims}d${type})
  add_executable(${strided_block_test_name} ${strided_block_test_name}.c)
  target_link_libraries(${strided_block_test_name}
    cmocka zfp rand${bits}Lib zfpHashLib zfpChecksumsLib)
  if(HAVE_LIBM_MATH)
    target_link_libraries(${strided_block_test_name} m)
  endif()

  target_compile_definitions(${strided_block_test_name} PRIVATE ${zfp_private_defs})
  if(PRINT_CHECKSUMS)
    target_compile_definitions(${strided_block_test_name} PUBLIC PRINT_CHECKSUMS)
  endif()

  add_test(NAME ${strided_block_test_name} COMMAND ${strided_block_test_name})
endfunction()

zfp_add_block_tests(1 Int32 32)
zfp_add_block_tests(1 Int64 64)
zfp_add_block_tests(2 Int32 32)
zfp_add_block_tests(2 Int64 64)
zfp_add_block_tests(3 Int32 32)
zfp_add_block_tests(3 Int64 64)
zfp_add_block_tests(4 Int32 32)
zfp_add_block_tests(4 Int64 64)

zfp_add_block_tests(1 Float 32)
zfp_add_block_tests(1 Double 64)
zfp_add_block_tests(2 Float 32)
zfp_add_block_tests(2 Double 64)
zfp_add_block_tests(3 Float 32)
zfp_add_block_tests(3 Double 64)
zfp_add_block_tests(4 Float 32)
zfp_add_block_tests(4 Double 64)
