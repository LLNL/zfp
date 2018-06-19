#ifndef CFP_ARRAYS
#define CFP_ARRAYS

#include "cfparray1f.h"

#include "zfp/system.h"

typedef struct {
  Cfp_array1f_api array1f;
} Cfp_api;

extern_ const Cfp_api cfp_api;

#endif
