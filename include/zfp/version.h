#ifndef ZFP_VERSION_H
#define ZFP_VERSION_H

/* stringification */
#define _zfp_str_(x) # x
#define _zfp_str(x) _zfp_str_(x)

/* library version information */
#define ZFP_VERSION_MAJOR 0 /* library major version number */
#define ZFP_VERSION_MINOR 5 /* library minor version number */
#define ZFP_VERSION_PATCH 5 /* library patch version number */
#define ZFP_VERSION_RELEASE ZFP_VERSION_PATCH

/* codec version number (see also zfp_codec_version) */
#define ZFP_CODEC 5

/* library version number (see also zfp_library_version) */
#define ZFP_VERSION \
  ((ZFP_VERSION_MAJOR << 8) + \
   (ZFP_VERSION_MINOR << 4) + \
   (ZFP_VERSION_PATCH << 0))

/* library version string (see also zfp_version_string) */
#define ZFP_VERSION_STRING \
  _zfp_str(ZFP_VERSION_MAJOR) "." \
  _zfp_str(ZFP_VERSION_MINOR) "." \
  _zfp_str(ZFP_VERSION_PATCH)

#endif
