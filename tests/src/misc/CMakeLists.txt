add_executable(testZfpHeader testZfpHeader.c)
target_link_libraries(testZfpHeader cmocka zfp)
add_test(NAME testZfpHeader COMMAND testZfpHeader)

add_executable(testZfpStream testZfpStream.c)
target_link_libraries(testZfpStream cmocka zfp)
add_test(NAME testZfpStream COMMAND testZfpStream)

if(HAVE_LIBM_MATH)
  target_link_libraries(testZfpHeader m)
  target_link_libraries(testZfpStream m)
endif()
