#ifndef CFP_ARRAYS
#define CFP_ARRAYS

#include "cfparray1f.h"
#include "cfparray1d.h"
#include "cfparray2f.h"
#include "cfparray2d.h"
#include "cfparray3f.h"
#include "cfparray3d.h"

#include "zfp/system.h"

typedef struct {
  Cfp_array1f_api array1f;
  Cfp_array1d_api array1d;
  Cfp_array2f_api array2f;
  Cfp_array2d_api array2d;
  Cfp_array3f_api array3f;
  Cfp_array3d_api array3d;
} Cfp_api;

extern_ const Cfp_api cfp_api;

#endif
