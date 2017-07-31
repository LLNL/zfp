# libraries used by tests

# seeded random number generators
add_library(rand32Lib rand32.c rand32.h)

add_library(rand64Lib rand64.c rand64.h)

# hashing
add_library(hashBaseLib hashBase.c hashBase.h)

add_library(hash32Lib hash32.c hash32.h)
target_link_libraries(hash32Lib hashBaseLib)

add_library(hash64Lib hash64.c hash64.h)
target_link_libraries(hash64Lib hashBaseLib)

# fixed point
add_library(fixedpoint96Lib fixedpoint96.c fixedpoint96.h)

# [seeded] random correlated array generator
add_library(genSmoothRandNumsLib genSmoothRandNums.c genSmoothRandNums.h)
target_link_libraries(genSmoothRandNumsLib rand64Lib fixedpoint96Lib)
