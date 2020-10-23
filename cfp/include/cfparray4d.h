#ifndef CFP_ARRAY_4D
#define CFP_ARRAY_4D

#include <stddef.h>
#include "zfp.h"
#include "zfp/types.h"

typedef struct {
  void* object;
} cfp_array4d;

typedef struct {
  cfp_array4d array;
  size_t x, y, z, w;
} cfp_ref4d;

typedef struct {
  cfp_ref4d reference;
} cfp_ptr4d;

typedef struct {
  cfp_array4d array;
  size_t x, y, z, w;
} cfp_iter4d;

struct cfp_header;

typedef struct {
  double (*get)(cfp_ref4d self);
  void (*set)(cfp_ref4d self, double val);
  void (*copy)(cfp_ref4d self, cfp_ref4d src);
  cfp_ptr4d (*ptr)(cfp_ref4d self);
} cfp_ref4d_api;

typedef struct {
  void (*set)(cfp_ptr4d self, double val);
  void (*set_at)(cfp_ptr4d self, double val, ptrdiff_t d);
  double (*get)(cfp_ptr4d self);
  double (*get_at)(cfp_ptr4d self, ptrdiff_t d);
  cfp_ref4d (*ref)(cfp_ptr4d self);
  cfp_ref4d (*ref_at)(cfp_ptr4d self, ptrdiff_t d);
  zfp_bool (*lt)(cfp_ptr4d self, cfp_ptr4d src);
  zfp_bool (*gt)(cfp_ptr4d self, cfp_ptr4d src);
  zfp_bool (*leq)(cfp_ptr4d self, cfp_ptr4d src);
  zfp_bool (*geq)(cfp_ptr4d self, cfp_ptr4d src);
  zfp_bool (*eq)(cfp_ptr4d self, cfp_ptr4d src);
  zfp_bool (*neq)(cfp_ptr4d self, cfp_ptr4d src);
  ptrdiff_t (*distance)(cfp_ptr4d self, cfp_ptr4d src);
  cfp_ptr4d (*next)(cfp_ptr4d self, ptrdiff_t d);
  cfp_ptr4d (*prev)(cfp_ptr4d self, ptrdiff_t d);
  cfp_ptr4d (*inc)(cfp_ptr4d self);
  cfp_ptr4d (*dec)(cfp_ptr4d self);
} cfp_ptr4d_api;

typedef struct {
  void (*set)(cfp_iter4d self, double val);
  double (*get)(cfp_iter4d self);
  cfp_ref4d (*ref)(cfp_iter4d self);
  cfp_ptr4d (*ptr)(cfp_iter4d self);
  cfp_iter4d (*inc)(cfp_iter4d self);
  zfp_bool (*eq)(cfp_iter4d self, cfp_iter4d src);
  zfp_bool (*neq)(cfp_iter4d self, cfp_iter4d src);
  size_t (*i)(cfp_iter4d self);
  size_t (*j)(cfp_iter4d self);
  size_t (*k)(cfp_iter4d self);
  size_t (*l)(cfp_iter4d self);
} cfp_iter4d_api;

typedef struct {
  cfp_array4d (*ctor_default)();
  cfp_array4d (*ctor)(size_t nx, size_t ny, size_t nz, size_t nw, double rate, const double* p, size_t csize);
  cfp_array4d (*ctor_copy)(const cfp_array4d src);
  cfp_array4d (*ctor_header)(const struct cfp_header h, const void* buffer, size_t buffer_size_bytes);
  void (*dtor)(cfp_array4d self);

  void (*deep_copy)(cfp_array4d self, const cfp_array4d src);

  double (*rate)(const cfp_array4d self);
  double (*set_rate)(cfp_array4d self, double rate);
  size_t (*cache_size)(const cfp_array4d self);
  void (*set_cache_size)(cfp_array4d self, size_t csize);
  void (*clear_cache)(const cfp_array4d self);
  void (*flush_cache)(const cfp_array4d self);
  size_t (*compressed_size)(const cfp_array4d self);
  void* (*compressed_data)(const cfp_array4d self);
  size_t (*size)(const cfp_array4d self);
  size_t (*size_x)(const cfp_array4d self);
  size_t (*size_y)(const cfp_array4d self);
  size_t (*size_z)(const cfp_array4d self);
  size_t (*size_w)(const cfp_array4d self);
  void (*resize)(cfp_array4d self, size_t nx, size_t ny, size_t nz, size_t nw, zfp_bool clear);

  void (*get_array)(const cfp_array4d self, double* p);
  void (*set_array)(cfp_array4d self, const double* p);
  double (*get_flat)(const cfp_array4d self, size_t i);
  void (*set_flat)(cfp_array4d self, size_t i, double val);
  double (*get)(const cfp_array4d self, size_t i, size_t j, size_t k, size_t l);
  void (*set)(cfp_array4d self, size_t i, size_t j, size_t k, size_t l, double val);

  cfp_ref4d (*ref)(cfp_array4d self, size_t i, size_t j, size_t k, size_t l);
  cfp_ref4d (*ref_flat)(cfp_array4d self, size_t i);

  cfp_ptr4d (*ptr)(cfp_array4d self, size_t i, size_t j, size_t k, size_t l);
  cfp_ptr4d (*ptr_flat)(cfp_array4d self, size_t i);

  cfp_iter4d (*begin)(cfp_array4d self);
  cfp_iter4d (*end)(cfp_array4d self);

  cfp_ref4d_api reference;
  cfp_ptr4d_api pointer;
  cfp_iter4d_api iterator;
} cfp_array4d_api;

#endif
