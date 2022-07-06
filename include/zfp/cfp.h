#ifndef CFP_ARRAY
#define CFP_ARRAY

#include <stddef.h>
#include "cfpheader.h"
#include "cfparray1f.h"
#include "cfparray1d.h"
#include "cfparray2f.h"
#include "cfparray2d.h"
#include "cfparray3f.h"
#include "cfparray3d.h"
#include "cfparray4f.h"
#include "cfparray4d.h"

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
