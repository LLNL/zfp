add_executable(testZfpHeader testZfpHeader.c)
target_link_libraries(testZfpHeader cmocka zfp)
add_test(NAME testZfpHeader COMMAND testZfpHeader)

add_executable(testZfpStream testZfpStream.c)
target_link_libraries(testZfpStream cmocka zfp)
add_test(NAME testZfpStream COMMAND testZfpStream)

add_executable(testZfpPromote testZfpPromote.c)
target_link_libraries(testZfpPromote cmocka zfp)
add_test(NAME testZfpPromote COMMAND testZfpPromote)

add_executable(testZfpField1f testZfpField1f.c)
target_link_libraries(testZfpField1f cmocka zfp)
add_test(NAME testZfpField1f COMMAND testZfpField1f)

add_executable(testZfpField2f testZfpField2f.c)
target_link_libraries(testZfpField2f cmocka zfp)
add_test(NAME testZfpField2f COMMAND testZfpField2f)

add_executable(testZfpField3f testZfpField3f.c)
target_link_libraries(testZfpField3f cmocka zfp)
add_test(NAME testZfpField3f COMMAND testZfpField3f)

add_executable(testZfpField4f testZfpField4f.c)
target_link_libraries(testZfpField4f cmocka zfp)
add_test(NAME testZfpField4f COMMAND testZfpField4f)

add_executable(testZfpField1d testZfpField1d.c)
target_link_libraries(testZfpField1d cmocka zfp)
add_test(NAME testZfpField1d COMMAND testZfpField1d)

add_executable(testZfpField2d testZfpField2d.c)
target_link_libraries(testZfpField2d cmocka zfp)
add_test(NAME testZfpField2d COMMAND testZfpField2d)

add_executable(testZfpField3d testZfpField3d.c)
target_link_libraries(testZfpField3d cmocka zfp)
add_test(NAME testZfpField3d COMMAND testZfpField3d)

add_executable(testZfpField4d testZfpField4d.c)
target_link_libraries(testZfpField4d cmocka zfp)
add_test(NAME testZfpField4d COMMAND testZfpField4d)

if(HAVE_LIBM_MATH)
  target_link_libraries(testZfpHeader m)
  target_link_libraries(testZfpStream m)
endif()
