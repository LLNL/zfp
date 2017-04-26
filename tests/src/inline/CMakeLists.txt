add_executable(testBitstream testBitstream.c)
target_link_libraries(testBitstream cmocka)
add_test(NAME testBitstream COMMAND testBitstream)
