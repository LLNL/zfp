from zfpy_version cimport *

# constants
major = ZFP_VERSION_MAJOR
minor = ZFP_VERSION_MINOR
patch = ZFP_VERSION_PATCH
tweak = ZFP_VERSION_TWEAK

codec = ZFP_CODEC

version = "{}.{}.{}".format(major, minor, patch)

full_version = "{}.{}.{}.{}".format(major, minor, patch, tweak)

version_string = c_zfp_version_string.decode()

# zfpy specific calls
cpdef geq(major, minor, patch, tweak=ZFP_VERSION_TWEAK):
    return (ZFP_VERSION_MAJOR, ZFP_VERSION_MINOR, ZFP_VERSION_PATCH, ZFP_VERSION_TWEAK) >= (major, minor, patch, tweak)
