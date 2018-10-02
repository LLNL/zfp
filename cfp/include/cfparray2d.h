#ifndef CFP_ARRAY_2D
#define CFP_ARRAY_2D

#include <stddef.h>
#include "zfp/types.h"

struct cfp_array2d;
typedef struct cfp_array2d cfp_array2d;

typedef struct {
  cfp_array2d* (*ctor_default)();
  cfp_array2d* (*ctor)(uint nx, uint ny, double rate, const double* p, size_t csize);
  cfp_array2d* (*ctor_copy)(const cfp_array2d* src);
  void (*dtor)(cfp_array2d* self);

  void (*deep_copy)(cfp_array2d* self, const cfp_array2d* src);

  double (*rate)(const cfp_array2d* self);
  double (*set_rate)(cfp_array2d* self, double rate);
  size_t (*cache_size)(const cfp_array2d* self);
  void (*set_cache_size)(cfp_array2d* self, size_t csize);
  void (*clear_cache)(const cfp_array2d* self);
  void (*flush_cache)(const cfp_array2d* self);
  size_t (*compressed_size)(const cfp_array2d* self);
  uchar* (*compressed_data)(const cfp_array2d* self);
  size_t (*size)(const cfp_array2d* self);
  uint (*size_x)(const cfp_array2d* self);
  uint (*size_y)(const cfp_array2d* self);
  void (*resize)(cfp_array2d* self, uint nx, uint ny, int clear);

  void (*get_array)(const cfp_array2d* self, double* p);
  void (*set_array)(cfp_array2d* self, const double* p);
  double (*get_flat)(const cfp_array2d* self, uint i);
  void (*set_flat)(cfp_array2d* self, uint i, double val);
  double (*get)(const cfp_array2d* self, uint i, uint j);
  void (*set)(cfp_array2d* self, uint i, uint j, double val);
} cfp_array2d_api;

#endif
