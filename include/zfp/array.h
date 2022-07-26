#ifndef CFP_ARRAY_H
#define CFP_ARRAY_H

#include <stddef.h>
#include "zfp/internal/cfp/header.h"
#include "zfp/internal/cfp/array1f.h"
#include "zfp/internal/cfp/array1d.h"
#include "zfp/internal/cfp/array2f.h"
#include "zfp/internal/cfp/array2d.h"
#include "zfp/internal/cfp/array3f.h"
#include "zfp/internal/cfp/array3d.h"
#include "zfp/internal/cfp/array4f.h"
#include "zfp/internal/cfp/array4d.h"

typedef struct {
  cfp_array1f_api array1f;
  cfp_array1d_api array1d;
  cfp_array2f_api array2f;
  cfp_array2d_api array2d;
  cfp_array3f_api array3f;
  cfp_array3d_api array3d;
  cfp_array4f_api array4f;
  cfp_array4d_api array4d;
} cfp_api;

#ifndef CFP_NAMESPACE
  #define CFP_NAMESPACE cfp
#endif

extern_ const cfp_api CFP_NAMESPACE;

#endif
