#ifndef ZFP_VERSION_H
#define ZFP_VERSION_H

/* stringification */
#define _zfp_str_(x) # x
#define _zfp_str(x) _zfp_str_(x)

/* macro for generating an integer version identifier */
#define ZFP_MAKE_VERSION(major, minor, patch) \
  (((major) << 8) + \
   ((minor) << 4) + \
   ((patch) << 0))

/* macro for generating a version string */
#define ZFP_MAKE_VERSION_STRING(major, minor, patch) \
  _zfp_str(major) "." \
  _zfp_str(minor) "." \
  _zfp_str(patch)

/* library version information */
#define ZFP_VERSION_MAJOR 0 /* library major version number */
#define ZFP_VERSION_MINOR 5 /* library minor version number */
#define ZFP_VERSION_PATCH 5 /* library patch version number */
#define ZFP_VERSION_RELEASE ZFP_VERSION_PATCH

/* codec version number (see also zfp_codec_version) */
#define ZFP_CODEC 5

/* library version number (see also zfp_library_version) */
#define ZFP_VERSION \
  ZFP_MAKE_VERSION(ZFP_VERSION_MAJOR, ZFP_VERSION_MINOR, ZFP_VERSION_PATCH)

/* library version string (see also zfp_version_string) */
#define ZFP_VERSION_STRING \
  ZFP_MAKE_VERSION_STRING(ZFP_VERSION_MAJOR, ZFP_VERSION_MINOR, ZFP_VERSION_PATCH)

#endif
