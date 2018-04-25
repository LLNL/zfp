add_executable(testOmp testOmp.c)
target_compile_options(testOmp PRIVATE ${OpenMP_C_FLAGS})
target_link_libraries(testOmp
  cmocka zfp ${OpenMP_C_FLAGS} ${OpenMP_C_LIBRARIES})
add_test(NAME testOmp COMMAND testOmp)
