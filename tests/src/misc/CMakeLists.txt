add_executable(testZfpHeader testZfpHeader.c)
target_link_libraries(testZfpHeader cmocka zfp)
add_test(NAME testZfpHeader COMMAND testZfpHeader)

add_executable(testZfpStream testZfpStream.c)
target_link_libraries(testZfpStream cmocka zfp)
add_test(NAME testZfpStream COMMAND testZfpStream)

add_executable(testZfpPromote testZfpPromote.c)
target_link_libraries(testZfpPromote cmocka zfp)
add_test(NAME testZfpPromote COMMAND testZfpPromote)

if(HAVE_LIBM_MATH)
  target_link_libraries(testZfpHeader m)
  target_link_libraries(testZfpStream m)
endif()
