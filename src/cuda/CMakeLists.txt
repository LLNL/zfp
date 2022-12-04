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
    error.cuh
    pointers.cuh
    shared.cuh
    traits.cuh)

set(cuZFP_headers
    cuZFP.h)
