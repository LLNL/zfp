#ifndef CFP_ARRAY_3D
#define CFP_ARRAY_3D

#include <stddef.h>
#include "zfp/types.h"

struct cfp_array3d;
typedef struct cfp_array3d cfp_array3d;

typedef struct {
  cfp_array3d* (*ctor_default)();
  cfp_array3d* (*ctor)(uint nx, uint ny, uint nz, double rate, const double* p, size_t csize);
  cfp_array3d* (*ctor_copy)(const cfp_array3d* src);
  void (*dtor)(cfp_array3d* self);

  void (*deep_copy)(cfp_array3d* self, const cfp_array3d* src);

  double (*rate)(const cfp_array3d* self);
  double (*set_rate)(cfp_array3d* self, double rate);
  size_t (*cache_size)(const cfp_array3d* self);
  void (*set_cache_size)(cfp_array3d* self, size_t csize);
  void (*clear_cache)(const cfp_array3d* self);
  void (*flush_cache)(const cfp_array3d* self);
  size_t (*compressed_size)(const cfp_array3d* self);
  uchar* (*compressed_data)(const cfp_array3d* self);
  size_t (*size)(const cfp_array3d* self);
  uint (*size_x)(const cfp_array3d* self);
  uint (*size_y)(const cfp_array3d* self);
  uint (*size_z)(const cfp_array3d* self);
  void (*resize)(cfp_array3d* self, uint nx, uint ny, uint nz, int clear);

  void (*get_array)(const cfp_array3d* self, double* p);
  void (*set_array)(cfp_array3d* self, const double* p);
  double (*get_flat)(const cfp_array3d* self, uint i);
  void (*set_flat)(cfp_array3d* self, uint i, double val);
  double (*get)(const cfp_array3d* self, uint i, uint j, uint k);
  void (*set)(cfp_array3d* self, uint i, uint j, uint k, double val);
} cfp_array3d_api;

#endif
