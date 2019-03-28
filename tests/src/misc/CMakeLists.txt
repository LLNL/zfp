add_executable(testZfpHeader testZfpHeader.c)
target_link_libraries(testZfpHeader cmocka zfp)
add_test(NAME testZfpHeader COMMAND testZfpHeader)

add_executable(testZfpStream testZfpStream.c)
target_link_libraries(testZfpStream cmocka zfp)
add_test(NAME testZfpStream COMMAND testZfpStream)

add_executable(rose_testZfpHeader rose_testZfpHeader.c)
target_link_libraries(rose_testZfpHeader cmocka zfp zFORp)
add_test(NAME rose_testZfpHeader COMMAND rose_testZfpHeader)

add_executable(rose_testZfpStream rose_testZfpStream.c)
target_link_libraries(rose_testZfpStream cmocka zfp zFORp)
add_test(NAME rose_testZfpStream COMMAND rose_testZfpStream)

if(HAVE_LIBM_MATH)
  target_link_libraries(testZfpHeader m)
  target_link_libraries(testZfpStream m)
endif()


add_executable(testOmp testOmp.c)
target_link_libraries(testOmp cmocka zfp)
add_test(NAME testOmp COMMAND testOmp)
if(ZFP_WITH_OPENMP)
  target_compile_options(testOmp PRIVATE ${OpenMP_C_FLAGS})
  target_link_libraries(testOmp ${OpenMP_C_LIBRARIES})
  set_property(TEST testOmp PROPERTY RUN_SERIAL TRUE)
endif()

if(ZFP_WITH_CUDA)
  add_executable(testCuda testCuda.c)
  target_link_libraries(testCuda cmocka zfp)
  add_test(NAME testCuda COMMAND testCuda)
endif()
