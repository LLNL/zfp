#ifndef CFP_ARRAY_3F
#define CFP_ARRAY_3F

#include <stddef.h>
#include "zfp/types.h"

struct cfp_array3f;
typedef struct cfp_array3f cfp_array3f;

typedef struct {
  cfp_array3f* (*ctor_default)();
  cfp_array3f* (*ctor)(uint nx, uint ny, uint nz, double rate, const float* p, size_t csize);
  cfp_array3f* (*ctor_copy)(const cfp_array3f* src);
  void (*dtor)(cfp_array3f* self);

  void (*deep_copy)(cfp_array3f* self, const cfp_array3f* src);

  double (*rate)(const cfp_array3f* self);
  double (*set_rate)(cfp_array3f* self, double rate);
  size_t (*cache_size)(const cfp_array3f* self);
  void (*set_cache_size)(cfp_array3f* self, size_t csize);
  void (*clear_cache)(const cfp_array3f* self);
  void (*flush_cache)(const cfp_array3f* self);
  size_t (*compressed_size)(const cfp_array3f* self);
  uchar* (*compressed_data)(const cfp_array3f* self);
  size_t (*size)(const cfp_array3f* self);
  uint (*size_x)(const cfp_array3f* self);
  uint (*size_y)(const cfp_array3f* self);
  uint (*size_z)(const cfp_array3f* self);
  void (*resize)(cfp_array3f* self, uint nx, uint ny, uint nz, int clear);

  void (*get_array)(const cfp_array3f* self, float* p);
  void (*set_array)(cfp_array3f* self, const float* p);
  float (*get_flat)(const cfp_array3f* self, uint i);
  void (*set_flat)(cfp_array3f* self, uint i, float val);
  float (*get)(const cfp_array3f* self, uint i, uint j, uint k);
  void (*set)(cfp_array3f* self, uint i, uint j, uint k, float val);
} cfp_array3f_api;

#endif
