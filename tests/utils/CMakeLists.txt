# libraries used by tests

# seeded random number generators
add_library(rand32Lib rand32.c rand32.h)

add_library(rand64Lib rand64.c rand64.h)

# hashing
add_library(zfpHashLib zfpHash.c zfpHash.h)
if(CMAKE_C_COMPILER_ID STREQUAL "GNU")
  target_compile_options(zfpHashLib
    PUBLIC $<$<COMPILE_LANGUAGE:C>:-Wno-pointer-sign>)
endif()

# fixed point
add_library(fixedpoint96Lib fixedpoint96.c fixedpoint96.h)

# [seeded] random correlated array generator
add_library(genSmoothRandNumsLib genSmoothRandNums.c genSmoothRandNums.h)
target_link_libraries(genSmoothRandNumsLib PRIVATE rand64Lib fixedpoint96Lib)

# strided array operations
add_library(stridedOperationsLib stridedOperations.c stridedOperations.h)
target_link_libraries(stridedOperationsLib PRIVATE zfp)

# compute zfp compression parameters
add_library(zfpCompressionParamsLib zfpCompressionParams.c zfpCompressionParams.h)

# timer
add_library(zfpTimerLib zfpTimer.c zfpTimer.h)

# checksums API
add_library(zfpChecksumsLib zfpChecksums.c zfpChecksums.h)
if(PRINT_CHECKSUMS)
  target_compile_definitions(zfpChecksumsLib PUBLIC PRINT_CHECKSUMS)
endif()

if(HAVE_LIBM_MATH)
  target_link_libraries(rand32Lib PRIVATE m)
  target_link_libraries(rand64Lib PRIVATE m)
  target_link_libraries(genSmoothRandNumsLib PRIVATE m)
  target_link_libraries(zfpCompressionParamsLib PRIVATE m)
endif()
