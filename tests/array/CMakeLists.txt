add_executable(testArray1d testArray1d.cpp)
target_link_libraries(testArray1d gtest gtest_main zfp hash64Lib genSmoothRandNumsLib)
add_test(NAME testArray1d COMMAND testArray1d)

add_executable(testArray1dRefs testArray1dRefs.cpp)
target_link_libraries(testArray1dRefs gtest gtest_main zfp hash64Lib)
add_test(NAME testArray1dRefs COMMAND testArray1dRefs)

add_executable(testArray1dPtrs testArray1dPtrs.cpp)
target_link_libraries(testArray1dPtrs gtest gtest_main zfp)
add_test(NAME testArray1dPtrs COMMAND testArray1dPtrs)

add_executable(testArray1dIters testArray1dIters.cpp)
target_link_libraries(testArray1dIters gtest gtest_main zfp)
add_test(NAME testArray1dIters COMMAND testArray1dIters)
