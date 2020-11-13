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
  /* header constructor/destructor */
  cfp_header (*ctor_buffer)(uint dims, zfp_type scalar_type, const void* bytes, size_t n);
  cfp_header (*ctor_array1f)(const cfp_array1f a);
  cfp_header (*ctor_array1d)(const cfp_array1d a);
  cfp_header (*ctor_array2f)(const cfp_array2f a);
  cfp_header (*ctor_array2d)(const cfp_array2d a);
  cfp_header (*ctor_array3f)(const cfp_array3f a);
  cfp_header (*ctor_array3d)(const cfp_array3d a);
  cfp_header (*ctor_array4f)(const cfp_array4f a);
  cfp_header (*ctor_array4d)(const cfp_array4d a);
  void (*dtor)(cfp_header self);

  /* array metadata */
  zfp_type (*scalar_type)(const cfp_header self);
  uint (*dimensionality)(const cfp_header self);
  size_t (*size_x)(const cfp_header self);
  size_t (*size_y)(const cfp_header self);
  size_t (*size_z)(const cfp_header self);
  size_t (*size_w)(const cfp_header self);
  double (*rate)(const cfp_header self);

  /* header payload: data pointer and byte size */
  const void* (*data)(const cfp_header self);
  size_t (*size)(const cfp_header self);
} cfp_header_api;

#endif
