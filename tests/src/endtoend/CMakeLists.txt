# serial exec
# int
add_executable(testZfpSerial1dInt32 testZfpSerial1dInt32.c)
target_link_libraries(testZfpSerial1dInt32 cmocka zfp hash32Lib genSmoothRandNumsLib)
add_test(NAME testZfpSerial1dInt32 COMMAND testZfpSerial1dInt32)

add_executable(testZfpSerial1dInt64 testZfpSerial1dInt64.c)
target_link_libraries(testZfpSerial1dInt64 cmocka zfp hash64Lib genSmoothRandNumsLib)
add_test(NAME testZfpSerial1dInt64 COMMAND testZfpSerial1dInt64)

add_executable(testZfpSerial2dInt32 testZfpSerial2dInt32.c)
target_link_libraries(testZfpSerial2dInt32 cmocka zfp hash32Lib genSmoothRandNumsLib)
add_test(NAME testZfpSerial2dInt32 COMMAND testZfpSerial2dInt32)

add_executable(testZfpSerial2dInt64 testZfpSerial2dInt64.c)
target_link_libraries(testZfpSerial2dInt64 cmocka zfp hash64Lib genSmoothRandNumsLib)
add_test(NAME testZfpSerial2dInt64 COMMAND testZfpSerial2dInt64)

add_executable(testZfpSerial3dInt32 testZfpSerial3dInt32.c)
target_link_libraries(testZfpSerial3dInt32 cmocka zfp hash32Lib genSmoothRandNumsLib)
add_test(NAME testZfpSerial3dInt32 COMMAND testZfpSerial3dInt32)

add_executable(testZfpSerial3dInt64 testZfpSerial3dInt64.c)
target_link_libraries(testZfpSerial3dInt64 cmocka zfp hash64Lib genSmoothRandNumsLib)
add_test(NAME testZfpSerial3dInt64 COMMAND testZfpSerial3dInt64)

# floating point
add_executable(testZfpSerial1dFloat testZfpSerial1dFloat.c)
target_link_libraries(testZfpSerial1dFloat cmocka zfp hash32Lib genSmoothRandNumsLib)
add_test(NAME testZfpSerial1dFloat COMMAND testZfpSerial1dFloat)

add_executable(testZfpSerial1dDouble testZfpSerial1dDouble.c)
target_link_libraries(testZfpSerial1dDouble cmocka zfp hash64Lib genSmoothRandNumsLib)
add_test(NAME testZfpSerial1dDouble COMMAND testZfpSerial1dDouble)

add_executable(testZfpSerial2dFloat testZfpSerial2dFloat.c)
target_link_libraries(testZfpSerial2dFloat cmocka zfp hash32Lib genSmoothRandNumsLib)
add_test(NAME testZfpSerial2dFloat COMMAND testZfpSerial2dFloat)

add_executable(testZfpSerial2dDouble testZfpSerial2dDouble.c)
target_link_libraries(testZfpSerial2dDouble cmocka zfp hash64Lib genSmoothRandNumsLib)
add_test(NAME testZfpSerial2dDouble COMMAND testZfpSerial2dDouble)

add_executable(testZfpSerial3dFloat testZfpSerial3dFloat.c)
target_link_libraries(testZfpSerial3dFloat cmocka zfp hash32Lib genSmoothRandNumsLib)
add_test(NAME testZfpSerial3dFloat COMMAND testZfpSerial3dFloat)

add_executable(testZfpSerial3dDouble testZfpSerial3dDouble.c)
target_link_libraries(testZfpSerial3dDouble cmocka zfp hash64Lib genSmoothRandNumsLib)
add_test(NAME testZfpSerial3dDouble COMMAND testZfpSerial3dDouble)

if (OPENMP_FOUND)
  # omp exec
  # int
  add_executable(testZfpOmp1dInt32 testZfpOmp1dInt32.c)
  target_link_libraries(testZfpOmp1dInt32 cmocka zfp hash32Lib genSmoothRandNumsLib)
  add_test(NAME testZfpOmp1dInt32 COMMAND testZfpOmp1dInt32)

  add_executable(testZfpOmp1dInt64 testZfpOmp1dInt64.c)
  target_link_libraries(testZfpOmp1dInt64 cmocka zfp hash64Lib genSmoothRandNumsLib)
  add_test(NAME testZfpOmp1dInt64 COMMAND testZfpOmp1dInt64)

  add_executable(testZfpOmp2dInt32 testZfpOmp2dInt32.c)
  target_link_libraries(testZfpOmp2dInt32 cmocka zfp hash32Lib genSmoothRandNumsLib)
  add_test(NAME testZfpOmp2dInt32 COMMAND testZfpOmp2dInt32)

  add_executable(testZfpOmp2dInt64 testZfpOmp2dInt64.c)
  target_link_libraries(testZfpOmp2dInt64 cmocka zfp hash64Lib genSmoothRandNumsLib)
  add_test(NAME testZfpOmp2dInt64 COMMAND testZfpOmp2dInt64)

  add_executable(testZfpOmp3dInt32 testZfpOmp3dInt32.c)
  target_link_libraries(testZfpOmp3dInt32 cmocka zfp hash32Lib genSmoothRandNumsLib)
  add_test(NAME testZfpOmp3dInt32 COMMAND testZfpOmp3dInt32)

  add_executable(testZfpOmp3dInt64 testZfpOmp3dInt64.c)
  target_link_libraries(testZfpOmp3dInt64 cmocka zfp hash64Lib genSmoothRandNumsLib)
  add_test(NAME testZfpOmp3dInt64 COMMAND testZfpOmp3dInt64)

  # floating point
  add_executable(testZfpOmp1dFloat testZfpOmp1dFloat.c)
  target_link_libraries(testZfpOmp1dFloat cmocka zfp hash32Lib genSmoothRandNumsLib)
  add_test(NAME testZfpOmp1dFloat COMMAND testZfpOmp1dFloat)

  add_executable(testZfpOmp1dDouble testZfpOmp1dDouble.c)
  target_link_libraries(testZfpOmp1dDouble cmocka zfp hash64Lib genSmoothRandNumsLib)
  add_test(NAME testZfpOmp1dDouble COMMAND testZfpOmp1dDouble)

  add_executable(testZfpOmp2dFloat testZfpOmp2dFloat.c)
  target_link_libraries(testZfpOmp2dFloat cmocka zfp hash32Lib genSmoothRandNumsLib)
  add_test(NAME testZfpOmp2dFloat COMMAND testZfpOmp2dFloat)

  add_executable(testZfpOmp2dDouble testZfpOmp2dDouble.c)
  target_link_libraries(testZfpOmp2dDouble cmocka zfp hash64Lib genSmoothRandNumsLib)
  add_test(NAME testZfpOmp2dDouble COMMAND testZfpOmp2dDouble)

  add_executable(testZfpOmp3dFloat testZfpOmp3dFloat.c)
  target_link_libraries(testZfpOmp3dFloat cmocka zfp hash32Lib genSmoothRandNumsLib)
  add_test(NAME testZfpOmp3dFloat COMMAND testZfpOmp3dFloat)

  add_executable(testZfpOmp3dDouble testZfpOmp3dDouble.c)
  target_link_libraries(testZfpOmp3dDouble cmocka zfp hash64Lib genSmoothRandNumsLib)
  add_test(NAME testZfpOmp3dDouble COMMAND testZfpOmp3dDouble)
endif()
