#ifndef ZFP_VERSION_H
#define ZFP_VERSION_H

/* library version information */
#define ZFP_VERSION_MAJOR 1   /* library major version number */
#define ZFP_VERSION_MINOR 0   /* library minor version number */
#define ZFP_VERSION_PATCH 0   /* library patch version number */
#define ZFP_VERSION_TWEAK 0   /* library tweak version number */

/* defined for work in progress (indicates unofficial release) */
#define ZFP_VERSION_DEVELOP 1

/* codec version number (see also zfp_codec_version) */
#define ZFP_CODEC 5

/* stringification */
#define _zfp_str_(x) # x
#define _zfp_str(x) _zfp_str_(x)

/* macro for generating an integer version identifier */
#define ZFP_MAKE_VERSION(major, minor, patch, tweak) \
  (((major) << 12) + \
   ((minor) << 8) + \
   ((patch) << 4) + \
   ((tweak) << 0))

/* macros for generating a version string */
#define ZFP_MAKE_VERSION_STRING(major, minor, patch) \
  _zfp_str(major) "." \
  _zfp_str(minor) "." \
  _zfp_str(patch)

#define ZFP_MAKE_FULLVERSION_STRING(major, minor, patch, tweak) \
  _zfp_str(major) "." \
  _zfp_str(minor) "." \
  _zfp_str(patch) "." \
  _zfp_str(tweak)

/* library version number (see also zfp_library_version) */
#define ZFP_VERSION \
  ZFP_MAKE_VERSION(ZFP_VERSION_MAJOR, ZFP_VERSION_MINOR, ZFP_VERSION_PATCH, ZFP_VERSION_TWEAK)

/* library version string (see also zfp_version_string) */
#if ZFP_VERSION_TWEAK == 0
  #define ZFP_VERSION_STRING \
    ZFP_MAKE_VERSION_STRING(ZFP_VERSION_MAJOR, ZFP_VERSION_MINOR, ZFP_VERSION_PATCH)
#else
  #define ZFP_VERSION_STRING \
    ZFP_MAKE_FULLVERSION_STRING(ZFP_VERSION_MAJOR, ZFP_VERSION_MINOR, ZFP_VERSION_PATCH, ZFP_VERSION_TWEAK)
#endif

#endif
