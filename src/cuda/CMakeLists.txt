###############################################################################
#
#  file: src/cuda/CMakeLists.txt
#
###############################################################################

set(zfp_cuda_sources
    interface.cu         # main entry point
    decode.cuh
    decode1.cuh
    decode2.cuh
    decode3.cuh
    reader.cuh
    encode.cuh
    encode1.cuh
    encode2.cuh
    encode3.cuh
    writer.cuh
    constants.cuh
    device.cuh
    error.cuh
    shared.cuh
    timer.cuh
    traits.cuh
    variable.cuh)

set(zfp_cuda_headers
    interface.h)
