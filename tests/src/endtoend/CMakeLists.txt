add_executable(testZfpEndtoend1dInt32 testZfpEndtoend1dInt32.c ../utils/fixedpoint96.c ../utils/genSmoothRandInts.c)
target_link_libraries(testZfpEndtoend1dInt32 cmocka zfp)
add_test(NAME testZfpEndtoend1dInt32 COMMAND testZfpEndtoend1dInt32)

add_executable(testZfpEndtoend1dInt64 testZfpEndtoend1dInt64.c ../utils/fixedpoint96.c ../utils/genSmoothRandInts.c)
target_link_libraries(testZfpEndtoend1dInt64 cmocka zfp)
add_test(NAME testZfpEndtoend1dInt64 COMMAND testZfpEndtoend1dInt64)
