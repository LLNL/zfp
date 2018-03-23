add_executable(testOmp testOmp.c)
target_link_libraries(testOmp cmocka zfp)
add_test(NAME testOmp COMMAND testOmp)
