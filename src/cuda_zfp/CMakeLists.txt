###############################################################################
#
#  file: src/cuZFP/CMakeLists.txt 
#
###############################################################################

set(cuZFP_sources
    cuZFP.cu         # main entry point
    decode.cuh
    decode1.cuh
    decode2.cuh
    decode3.cuh
    encode.cuh
    encode1.cuh
    encode2.cuh
    encode3.cuh
    pointers.cuh
    type_info.cuh)

set(cuZFP_headers
    constant_setup.cuh
    shared.h
    cuZFP.h
    ErrorCheck.h)

