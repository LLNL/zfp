#ifndef CFP_ARRAY_3D
#define CFP_ARRAY_3D

#include <stddef.h>
#include "zfp/types.h"

typedef struct {
  void* object;
} cfp_array3d;

typedef struct {
  uint i;
  uint j;
  uint k;
  cfp_array3d array;
} cfp_ref3d;

typedef struct {
  cfp_ref3d reference;
} cfp_ptr3d;

typedef struct {
  uint i;
  uint j;
  uint k;
  cfp_array3d array;
} cfp_iter3d;

typedef struct {
  double (*get)(cfp_ref3d self);
  void (*set)(cfp_ref3d self, double val);
  void (*copy)(cfp_ref3d self, cfp_ref3d src);
  cfp_ptr3d (*ptr)(cfp_ref3d self);
} cfp_ref3d_api;

typedef struct {
  cfp_ref3d (*ref)(cfp_ptr3d self);
  cfp_ref3d (*offset_ref)(cfp_ptr3d self, int i);
  int (*eq)(cfp_ptr3d self, cfp_ptr3d src);
  int (*diff)(cfp_ptr3d self, cfp_ptr3d src);
  cfp_ptr3d (*shift)(cfp_ptr3d self, int i);
  cfp_ptr3d (*inc)(cfp_ptr3d self);
  cfp_ptr3d (*dec)(cfp_ptr3d self);
} cfp_ptr3d_api;

typedef struct {
  cfp_ref3d (*ref)(cfp_iter3d self);
  cfp_iter3d (*inc)(cfp_iter3d self);
  int (*eq)(cfp_iter3d self, cfp_iter3d src);
  uint (*i)(cfp_iter3d self);
  uint (*j)(cfp_iter3d self);
  uint (*k)(cfp_iter3d self);
} cfp_iter3d_api;

typedef struct {
  cfp_array3d (*ctor_default)();
  cfp_array3d (*ctor)(uint nx, uint ny, uint nz, double rate, const double* p, size_t csize);
  cfp_array3d (*ctor_copy)(const cfp_array3d src);
  void (*dtor)(cfp_array3d self);

  void (*deep_copy)(cfp_array3d self, const cfp_array3d src);

  double (*rate)(const cfp_array3d self);
  double (*set_rate)(cfp_array3d self, double rate);
  size_t (*cache_size)(const cfp_array3d self);
  void (*set_cache_size)(cfp_array3d self, size_t csize);
  void (*clear_cache)(const cfp_array3d self);
  void (*flush_cache)(const cfp_array3d self);
  size_t (*compressed_size)(const cfp_array3d self);
  uchar* (*compressed_data)(const cfp_array3d self);
  size_t (*size)(const cfp_array3d self);
  uint (*size_x)(const cfp_array3d self);
  uint (*size_y)(const cfp_array3d self);
  uint (*size_z)(const cfp_array3d self);
  void (*resize)(cfp_array3d self, uint nx, uint ny, uint nz, int clear);

  void (*get_array)(const cfp_array3d self, double* p);
  void (*set_array)(cfp_array3d self, const double* p);
  double (*get_flat)(const cfp_array3d self, uint i);
  void (*set_flat)(cfp_array3d self, uint i, double val);
  double (*get)(const cfp_array3d self, uint i, uint j, uint k);
  void (*set)(cfp_array3d self, uint i, uint j, uint k, double val);

  cfp_ref3d (*ref)(cfp_array3d self, uint i, uint j, uint k);
  cfp_ref3d (*flat_ref)(cfp_array3d self, uint i);

  cfp_ptr3d (*ptr)(cfp_array3d self, uint i, uint j, uint k);
  cfp_ptr3d (*flat_ptr)(cfp_array3d self, uint i);

  cfp_iter3d (*begin)(cfp_array3d self);
  cfp_iter3d (*end)(cfp_array3d self);

  cfp_ref3d_api reference;
  cfp_ptr3d_api pointer;
  cfp_iter3d_api iterator;
} cfp_array3d_api;

#endif
