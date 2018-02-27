add_executable(testArray1d testArray1d.cpp)
target_link_libraries(testArray1d gtest gtest_main zfp hash64Lib genSmoothRandNumsLib)
add_test(NAME testArray1d COMMAND testArray1d)
