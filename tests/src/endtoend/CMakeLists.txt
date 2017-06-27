add_executable(testZfpEndtoend1dInt32 testZfpEndtoend1dInt32.c ../utils/fixedpoint96.c ../utils/genSmoothRandNums.c)
target_link_libraries(testZfpEndtoend1dInt32 cmocka zfp)
add_test(NAME testZfpEndtoend1dInt32 COMMAND testZfpEndtoend1dInt32)

add_executable(testZfpEndtoend1dInt64 testZfpEndtoend1dInt64.c ../utils/fixedpoint96.c ../utils/genSmoothRandNums.c)
target_link_libraries(testZfpEndtoend1dInt64 cmocka zfp)
add_test(NAME testZfpEndtoend1dInt64 COMMAND testZfpEndtoend1dInt64)

add_executable(testZfpEndtoend2dInt32 testZfpEndtoend2dInt32.c ../utils/fixedpoint96.c ../utils/genSmoothRandNums.c)
target_link_libraries(testZfpEndtoend2dInt32 cmocka zfp)
add_test(NAME testZfpEndtoend2dInt32 COMMAND testZfpEndtoend2dInt32)

add_executable(testZfpEndtoend2dInt64 testZfpEndtoend2dInt64.c ../utils/fixedpoint96.c ../utils/genSmoothRandNums.c)
target_link_libraries(testZfpEndtoend2dInt64 cmocka zfp)
add_test(NAME testZfpEndtoend2dInt64 COMMAND testZfpEndtoend2dInt64)

add_executable(testZfpEndtoend3dInt32 testZfpEndtoend3dInt32.c ../utils/fixedpoint96.c ../utils/genSmoothRandNums.c)
target_link_libraries(testZfpEndtoend3dInt32 cmocka zfp)
add_test(NAME testZfpEndtoend3dInt32 COMMAND testZfpEndtoend3dInt32)

add_executable(testZfpEndtoend3dInt64 testZfpEndtoend3dInt64.c ../utils/fixedpoint96.c ../utils/genSmoothRandNums.c)
target_link_libraries(testZfpEndtoend3dInt64 cmocka zfp)
add_test(NAME testZfpEndtoend3dInt64 COMMAND testZfpEndtoend3dInt64)
