#ifndef CFP_ARRAY_1D
#define CFP_ARRAY_1D

#include <stddef.h>
#include "zfp/types.h"

struct cfp_array1d;
typedef struct cfp_array1d cfp_array1d;

typedef struct {
  cfp_array1d* (*ctor_default)();
  cfp_array1d* (*ctor)(uint n, double rate, const double* p, size_t csize);
  cfp_array1d* (*ctor_copy)(const cfp_array1d* src);
  void (*dtor)(cfp_array1d* self);

  void (*deep_copy)(cfp_array1d* self, const cfp_array1d* src);

  double (*rate)(const cfp_array1d* self);
  double (*set_rate)(cfp_array1d* self, double rate);
  size_t (*cache_size)(const cfp_array1d* self);
  void (*set_cache_size)(cfp_array1d* self, size_t csize);
  void (*clear_cache)(const cfp_array1d* self);
  void (*flush_cache)(const cfp_array1d* self);
  size_t (*compressed_size)(const cfp_array1d* self);
  uchar* (*compressed_data)(const cfp_array1d* self);
  size_t (*size)(const cfp_array1d* self);
  void (*resize)(cfp_array1d* self, uint n, int clear);

  void (*get_array)(const cfp_array1d* self, double* p);
  void (*set_array)(cfp_array1d* self, const double* p);
  double (*get_flat)(const cfp_array1d* self, uint i);
  void (*set_flat)(cfp_array1d* self, uint i, double val);
  double (*get)(const cfp_array1d* self, uint i);
  void (*set)(cfp_array1d* self, uint i, double val);
} cfp_array1d_api;

#endif
