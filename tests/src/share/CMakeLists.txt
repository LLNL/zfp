add_executable(testOmpInternal testOmpInternal.c)
target_compile_options(testOmpInternal PRIVATE ${OpenMP_C_FLAGS})
target_link_libraries(testOmpInternal
  cmocka zfp ${OpenMP_C_FLAGS} ${OpenMP_C_LIBRARIES})
add_test(NAME testOmpInternal COMMAND testOmpInternal)
