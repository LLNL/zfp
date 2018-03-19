add_executable(testZfpHeader testZfpHeader.c)
target_link_libraries(testZfpHeader cmocka zfp)
add_test(NAME testZfpHeader COMMAND testZfpHeader)

# preprocessor has different tests if compiled with[out] OpenMP
add_executable(testOmpParams testOmpParams.c)
target_link_libraries(testOmpParams cmocka zfp)
add_test(NAME testOmpParams COMMAND testOmpParams)
