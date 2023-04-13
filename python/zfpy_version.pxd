import cython

from libcpp.string cimport string
cimport libc.stdint as stdint

cdef extern from "zfp/version.h":
    cython.uint ZFP_VERSION_MAJOR
    cython.uint ZFP_VERSION_MINOR
    cython.uint ZFP_VERSION_PATCH
    cython.uint ZFP_VERSION_TWEAK

    cython.uint ZFP_CODEC

cdef extern from "zfp.h":
    cdef const char* const c_zfp_version_string "zfp_version_string"
