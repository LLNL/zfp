#ifndef CFP_HEADER
#define CFP_HEADER

#include <stddef.h>
#include "cfptypes.h"

struct cfp_header {
  void* object;
};

typedef struct {
  cfp_header (*ctor_default)(void);
  cfp_header (*ctor_buffer)(const void* bytes, size_t n);
  cfp_header (*ctor_array1f)(cfp_array1f a);
  cfp_header (*ctor_array1d)(cfp_array1d a);
  cfp_header (*ctor_array2f)(cfp_array2f a);
  cfp_header (*ctor_array2d)(cfp_array2d a);
  cfp_header (*ctor_array3f)(cfp_array3f a);
  cfp_header (*ctor_array3d)(cfp_array3d a);
//  cfp_header (*ctor_array4f)(cfp_array4f a);
//  cfp_header (*ctor_array4d)(cfp_array4d a);
  void (*dtor)(cfp_header self);

  double (*rate)(cfp_header self);
  const void* (*data)(cfp_header self);
  size_t (*size)(cfp_header self);
} cfp_header_api;

#endif
