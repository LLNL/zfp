import cython

from libcpp cimport bool as cybool
cimport libc.stdint as stdint

cdef extern from "zfp/version.h":
    """
    #ifndef ZFP_VERSION_DEVELOP
        #define ZFP_VERSION_DEVELOP 0
    #endif
    """
    cython.uint ZFP_VERSION_MAJOR
    cython.uint ZFP_VERSION_MINOR
    cython.uint ZFP_VERSION_PATCH
    cython.uint ZFP_VERSION_TWEAK

    cython.uint ZFP_CODEC
    
    cybool ZFP_VERSION_DEVELOP

cdef extern from "zfp.h":
    cdef const char* const c_zfp_version_string "zfp_version_string"
