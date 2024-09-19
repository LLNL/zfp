###############################################################################
#
#  file: src/dpcpp_zfp/CMakeLists.txt 
#
###############################################################################

set(syclZFP_sources
    syclZFP.dp.cpp         # main entry point
    decode.dp.hpp
    decode1.dp.hpp
    decode2.dp.hpp
    decode3.dp.hpp
    encode.dp.hpp
    encode1.dp.hpp
    encode2.dp.hpp
    encode3.dp.hpp
    pointers.dp.hpp
    type_info.dp.hpp)

set(syclZFP_headers
    shared.h
    syclZFP.h
    ErrorCheck.h)

#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 --std=c++11")


#add_library(cuZFP
#                 ${cuZFP_sources}
#                 ${cuZFP_headers})

#install(TARGETS cuZFP DESTINATION lib64)
#install(FILES cuZFP.h zfp_structs.h DESTINATION include)	
