###############################################################################
#
#  file: src/hip/CMakeLists.txt
#
###############################################################################

set(zfp_hip_sources
    interface.cpp         # main entry point
    decode.h
    decode1.h
    decode2.h
    decode3.h
    reader.h
    encode.h
    encode1.h
    encode2.h
    encode3.h
    writer.h
    constants.h
    device.h
    error.h
    shared.h
    timer.h
    traits.h
    variable.h)

set(zfp_hip_headers
    interface.h)
