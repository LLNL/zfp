add_executable(testZfpHeader testZfpHeader.c)
target_link_libraries(testZfpHeader cmocka zfp)
add_test(NAME testZfpHeader COMMAND testZfpHeader)
