from zfpy_version cimport *

# constants
major = ZFP_VERSION_MAJOR
minor = ZFP_VERSION_MINOR
patch = ZFP_VERSION_PATCH
tweak = ZFP_VERSION_TWEAK

codec = ZFP_CODEC

# zfpy specific calls
cpdef version(full = False):
    vstr = "{}.{}.{}".format(major, minor, patch)
    if full:
        vstr += ".{}".format(tweak)
    return vstr

cpdef full_version():
    return version(True)

cpdef version_string():
    return c_zfp_version_string.decode()
