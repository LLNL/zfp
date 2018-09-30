#ifndef CFP_ARRAY_1F
#define CFP_ARRAY_1F

#include <stddef.h>
#include "zfp/types.h"

struct cfp_array1f;
typedef struct cfp_array1f cfp_array1f;

typedef struct {
  cfp_array1f* (*ctor_default)();
  cfp_array1f* (*ctor)(uint n, double rate, const float* p, size_t csize);
  cfp_array1f* (*ctor_copy)(const cfp_array1f* src);
  void (*dtor)(cfp_array1f* self);

  void (*deep_copy)(cfp_array1f* self, const cfp_array1f* src);

  double (*rate)(const cfp_array1f* self);
  double (*set_rate)(cfp_array1f* self, double rate);
  size_t (*cache_size)(const cfp_array1f* self);
  void (*set_cache_size)(cfp_array1f* self, size_t csize);
  void (*clear_cache)(const cfp_array1f* self);
  void (*flush_cache)(const cfp_array1f* self);
  size_t (*compressed_size)(const cfp_array1f* self);
  uchar* (*compressed_data)(const cfp_array1f* self);
  size_t (*size)(const cfp_array1f* self);
  void (*resize)(cfp_array1f* self, uint n, int clear);

  void (*get_array)(const cfp_array1f* self, float* p);
  void (*set_array)(cfp_array1f* self, const float* p);
  float (*get_flat)(const cfp_array1f* self, uint i);
  void (*set_flat)(cfp_array1f* self, uint i, float val);
  float (*get)(const cfp_array1f* self, uint i);
  void (*set)(cfp_array1f* self, uint i, float val);
} cfp_array1f_api;

#endif
