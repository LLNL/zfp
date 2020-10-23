#ifndef CFP_HEADER
#define CFP_HEADER

#include <stddef.h>
#include "cfparray1f.h"
#include "cfparray1d.h"
#include "cfparray2f.h"
#include "cfparray2d.h"
#include "cfparray3f.h"
#include "cfparray3d.h"
#include "cfparray4f.h"
#include "cfparray4d.h"

struct cfp_header {
  void* object;
};
typedef struct cfp_header cfp_header;

typedef struct {
  cfp_header (*ctor_buffer)(uint dim, zfp_type scalar_type, const void* bytes, size_t n);
  cfp_header (*ctor_array1f)(cfp_array1f a);
  cfp_header (*ctor_array1d)(cfp_array1d a);
  cfp_header (*ctor_array2f)(cfp_array2f a);
  cfp_header (*ctor_array2d)(cfp_array2d a);
  cfp_header (*ctor_array3f)(cfp_array3f a);
  cfp_header (*ctor_array3d)(cfp_array3d a);
  cfp_header (*ctor_array4f)(cfp_array4f a);
  cfp_header (*ctor_array4d)(cfp_array4d a);
  void (*dtor)(cfp_header self);

  zfp_type (*scalar_type)(cfp_header self);
  uint (*dimensionality)(cfp_header self);
  size_t (*size_x)(cfp_header self);
  size_t (*size_y)(cfp_header self);
  size_t (*size_z)(cfp_header self);
  size_t (*size_w)(cfp_header self);
  double (*rate)(cfp_header self);

  const void* (*data)(cfp_header self);
  size_t (*size)(cfp_header self);
} cfp_header_api;

#endif
