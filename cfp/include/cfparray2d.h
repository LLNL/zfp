#ifndef CFP_ARRAY_2D
#define CFP_ARRAY_2D

#include <stddef.h>
#include "zfp/types.h"

typedef struct {
  void* object;
} cfp_array2d;

typedef struct {
  uint i;
  uint j;
  cfp_array2d array;
} cfp_ref2d;

typedef struct {
  cfp_ref2d reference;
} cfp_ptr2d;

typedef struct {
  double (*get)(cfp_ref2d self);
  void (*set)(cfp_ref2d self, double val);
  void (*copy)(cfp_ref2d self, cfp_ref2d src);
  cfp_ptr2d (*get_ptr)(cfp_ref2d self);
} cfp_ref2d_api;

typedef struct {
  cfp_ref2d (*get_ref)(cfp_ptr2d self);
  //cfp_ref2d (*get_offset_ref)(cfp_ptr2d self, int i);
  int (*is_equal)(cfp_ptr2d self, cfp_ptr2d src);
  //int (*diff)(cfp_ptr2d self, cfp_ptr2d src);
  //cfp_ptr2d (*shift)(cfp_ptr2d self, int i);
  //cfp_ptr2d (*inc)(cfp_ptr2d self);
  //cfp_ptr2d (*dec)(cfp_ptr2d self);
  //cfp_ptr2d (*post_inc)(cfp_ptr2d self);
  //cfp_ptr2d (*post_dec)(cfp_ptr2d self);
} cfp_ptr2d_api;

typedef struct {
  cfp_array2d (*ctor_default)();
  cfp_array2d (*ctor)(uint nx, uint ny, double rate, const double* p, size_t csize);
  cfp_array2d (*ctor_copy)(const cfp_array2d src);
  void (*dtor)(cfp_array2d self);

  void (*deep_copy)(cfp_array2d self, const cfp_array2d src);

  double (*rate)(const cfp_array2d self);
  double (*set_rate)(cfp_array2d self, double rate);
  size_t (*cache_size)(const cfp_array2d self);
  void (*set_cache_size)(cfp_array2d self, size_t csize);
  void (*clear_cache)(const cfp_array2d self);
  void (*flush_cache)(const cfp_array2d self);
  size_t (*compressed_size)(const cfp_array2d self);
  uchar* (*compressed_data)(const cfp_array2d self);
  size_t (*size)(const cfp_array2d self);
  uint (*size_x)(const cfp_array2d self);
  uint (*size_y)(const cfp_array2d self);
  void (*resize)(cfp_array2d self, uint nx, uint ny, int clear);

  void (*get_array)(const cfp_array2d self, double* p);
  void (*set_array)(cfp_array2d self, const double* p);
  double (*get_flat)(const cfp_array2d self, uint i);
  void (*set_flat)(cfp_array2d self, uint i, double val);
  double (*get)(const cfp_array2d self, uint i, uint j);
  void (*set)(cfp_array2d self, uint i, uint j, double val);

  cfp_ref2d (*get_ref)(cfp_array2d self, uint i, uint j);
  cfp_ptr2d (*get_ptr)(cfp_array2d self, uint i, uint j);

  cfp_ref2d_api ref;
  cfp_ptr2d_api ptr;
} cfp_array2d_api;

#endif
