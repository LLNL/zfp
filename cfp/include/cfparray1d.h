#ifndef CFP_ARRAY_1D
#define CFP_ARRAY_1D

#include <stddef.h>
#include "zfp/types.h"

typedef struct {
  void* object;
} cfp_array1d;

typedef struct {
  uint i;
  cfp_array1d array;
} cfp_ref1d;

typedef struct {
  double (*get)(cfp_ref1d self);
} cfp_ref1d_api;

typedef struct {
  cfp_array1d (*ctor_default)();
  cfp_array1d (*ctor)(uint n, double rate, const double* p, size_t csize);
  cfp_array1d (*ctor_copy)(const cfp_array1d src);
  void (*dtor)(cfp_array1d self);

  void (*deep_copy)(cfp_array1d self, const cfp_array1d src);

  double (*rate)(const cfp_array1d self);
  double (*set_rate)(cfp_array1d self, double rate);
  size_t (*cache_size)(const cfp_array1d self);
  void (*set_cache_size)(cfp_array1d self, size_t csize);
  void (*clear_cache)(const cfp_array1d self);
  void (*flush_cache)(const cfp_array1d self);
  size_t (*compressed_size)(const cfp_array1d self);
  uchar* (*compressed_data)(const cfp_array1d self);
  size_t (*size)(const cfp_array1d self);
  void (*resize)(cfp_array1d self, uint n, int clear);

  void (*get_array)(const cfp_array1d self, double* p);
  void (*set_array)(cfp_array1d self, const double* p);
  double (*get_flat)(const cfp_array1d self, uint i);
  void (*set_flat)(cfp_array1d self, uint i, double val);
  double (*get)(const cfp_array1d self, uint i);
  void (*set)(cfp_array1d self, uint i, double val);

  cfp_ref1d (*get_ref)(cfp_array1d self, uint i);

  cfp_ref1d_api ref;
} cfp_array1d_api;

#endif
