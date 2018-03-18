add_executable(testArray1d testArray1d.cpp)
target_link_libraries(testArray1d gtest gtest_main zfp hash64Lib genSmoothRandNumsLib)
add_test(NAME testArray1d COMMAND testArray1d)

add_executable(testArray2d testArray2d.cpp)
target_link_libraries(testArray2d gtest gtest_main zfp hash64Lib genSmoothRandNumsLib)
add_test(NAME testArray2d COMMAND testArray2d)

add_executable(testArray3d testArray3d.cpp)
target_link_libraries(testArray3d gtest gtest_main zfp hash64Lib genSmoothRandNumsLib)
add_test(NAME testArray3d COMMAND testArray3d)

add_executable(testArray1dRefs testArray1dRefs.cpp)
target_link_libraries(testArray1dRefs gtest gtest_main zfp hash64Lib rand64Lib)
add_test(NAME testArray1dRefs COMMAND testArray1dRefs)

add_executable(testArray2dRefs testArray2dRefs.cpp)
target_link_libraries(testArray2dRefs gtest gtest_main zfp hash64Lib rand64Lib)
add_test(NAME testArray2dRefs COMMAND testArray2dRefs)

add_executable(testArray3dRefs testArray3dRefs.cpp)
target_link_libraries(testArray3dRefs gtest gtest_main zfp hash64Lib rand64Lib)
add_test(NAME testArray3dRefs COMMAND testArray3dRefs)

add_executable(testArray1dPtrs testArray1dPtrs.cpp)
target_link_libraries(testArray1dPtrs gtest gtest_main zfp)
add_test(NAME testArray1dPtrs COMMAND testArray1dPtrs)

add_executable(testArray2dPtrs testArray2dPtrs.cpp)
target_link_libraries(testArray2dPtrs gtest gtest_main zfp)
add_test(NAME testArray2dPtrs COMMAND testArray2dPtrs)

add_executable(testArray3dPtrs testArray3dPtrs.cpp)
target_link_libraries(testArray3dPtrs gtest gtest_main zfp)
add_test(NAME testArray3dPtrs COMMAND testArray3dPtrs)

add_executable(testArray1dIters testArray1dIters.cpp)
target_link_libraries(testArray1dIters gtest gtest_main zfp)
add_test(NAME testArray1dIters COMMAND testArray1dIters)

add_executable(testArray2dIters testArray2dIters.cpp)
target_link_libraries(testArray2dIters gtest gtest_main zfp)
add_test(NAME testArray2dIters COMMAND testArray2dIters)

add_executable(testArray3dIters testArray3dIters.cpp)
target_link_libraries(testArray3dIters gtest gtest_main zfp)
add_test(NAME testArray3dIters COMMAND testArray3dIters)

# float
add_executable(testArray1f testArray1f.cpp)
target_link_libraries(testArray1f gtest gtest_main zfp hash32Lib genSmoothRandNumsLib)
add_test(NAME testArray1f COMMAND testArray1f)

add_executable(testArray2f testArray2f.cpp)
target_link_libraries(testArray2f gtest gtest_main zfp hash32Lib genSmoothRandNumsLib)
add_test(NAME testArray2f COMMAND testArray2f)

add_executable(testArray3f testArray3f.cpp)
target_link_libraries(testArray3f gtest gtest_main zfp hash32Lib genSmoothRandNumsLib)
add_test(NAME testArray3f COMMAND testArray3f)

add_executable(testArray1fRefs testArray1fRefs.cpp)
target_link_libraries(testArray1fRefs gtest gtest_main zfp hash32Lib rand32Lib)
add_test(NAME testArray1fRefs COMMAND testArray1fRefs)

add_executable(testArray2fRefs testArray2fRefs.cpp)
target_link_libraries(testArray2fRefs gtest gtest_main zfp hash32Lib rand32Lib)
add_test(NAME testArray2fRefs COMMAND testArray2fRefs)

add_executable(testArray3fRefs testArray3fRefs.cpp)
target_link_libraries(testArray3fRefs gtest gtest_main zfp hash32Lib rand32Lib)
add_test(NAME testArray3fRefs COMMAND testArray3fRefs)

add_executable(testArray1fPtrs testArray1fPtrs.cpp)
target_link_libraries(testArray1fPtrs gtest gtest_main zfp)
add_test(NAME testArray1fPtrs COMMAND testArray1fPtrs)

add_executable(testArray2fPtrs testArray2fPtrs.cpp)
target_link_libraries(testArray2fPtrs gtest gtest_main zfp)
add_test(NAME testArray2fPtrs COMMAND testArray2fPtrs)

add_executable(testArray3fPtrs testArray3fPtrs.cpp)
target_link_libraries(testArray3fPtrs gtest gtest_main zfp)
add_test(NAME testArray3fPtrs COMMAND testArray3fPtrs)

add_executable(testArray1fIters testArray1fIters.cpp)
target_link_libraries(testArray1fIters gtest gtest_main zfp)
add_test(NAME testArray1fIters COMMAND testArray1fIters)

add_executable(testArray2fIters testArray2fIters.cpp)
target_link_libraries(testArray2fIters gtest gtest_main zfp)
add_test(NAME testArray2fIters COMMAND testArray2fIters)

add_executable(testArray3fIters testArray3fIters.cpp)
target_link_libraries(testArray3fIters gtest gtest_main zfp)
add_test(NAME testArray3fIters COMMAND testArray3fIters)
