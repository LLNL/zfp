#ifndef CFP_ARRAY_2F
#define CFP_ARRAY_2F

#include <stddef.h>
#include "zfp/types.h"

struct cfp_array2f;
typedef struct cfp_array2f cfp_array2f;

typedef struct {
  cfp_array2f* (*ctor_default)();
  cfp_array2f* (*ctor)(uint nx, uint ny, double rate, const float* p, size_t csize);
  cfp_array2f* (*ctor_copy)(const cfp_array2f* src);
  void (*dtor)(cfp_array2f* self);

  void (*deep_copy)(cfp_array2f* self, const cfp_array2f* src);

  double (*rate)(const cfp_array2f* self);
  double (*set_rate)(cfp_array2f* self, double rate);
  size_t (*cache_size)(const cfp_array2f* self);
  void (*set_cache_size)(cfp_array2f* self, size_t csize);
  void (*clear_cache)(const cfp_array2f* self);
  void (*flush_cache)(const cfp_array2f* self);
  size_t (*compressed_size)(const cfp_array2f* self);
  uchar* (*compressed_data)(const cfp_array2f* self);
  size_t (*size)(const cfp_array2f* self);
  uint (*size_x)(const cfp_array2f* self);
  uint (*size_y)(const cfp_array2f* self);
  void (*resize)(cfp_array2f* self, uint nx, uint ny, int clear);

  void (*get_array)(const cfp_array2f* self, float* p);
  void (*set_array)(cfp_array2f* self, const float* p);
  float (*get_flat)(const cfp_array2f* self, uint i);
  void (*set_flat)(cfp_array2f* self, uint i, float val);
  float (*get)(const cfp_array2f* self, uint i, uint j);
  void (*set)(cfp_array2f* self, uint i, uint j, float val);
} cfp_array2f_api;

#endif
