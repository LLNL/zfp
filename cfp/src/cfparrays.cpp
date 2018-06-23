#include "cfparrays.h"

#include "cfparray1f.cpp"
#include "cfparray1d.cpp"
#include "cfparray2f.cpp"
#include "cfparray2d.cpp"
#include "cfparray3f.cpp"
#include "cfparray3d.cpp"

const Cfp_api cfp_api = {
  cfp_array1f_api,
  cfp_array1d_api,
  cfp_array2f_api,
  cfp_array2d_api,
  cfp_array3f_api,
  cfp_array3d_api,
};
