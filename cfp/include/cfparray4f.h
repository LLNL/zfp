#ifndef CFP_ARRAY_4F
#define CFP_ARRAY_4F

#include <stddef.h>
#include "zfp.h"
#include "zfp/types.h"

typedef struct {
  void* object;
} cfp_array4f;

typedef struct {
  cfp_array4f array;
  size_t x, y, z, w;
} cfp_ref4f;

typedef struct {
  cfp_ref4f reference;
} cfp_ptr4f;

typedef struct {
  cfp_array4f array;
  size_t x, y, z, w;
} cfp_iter4f;

struct cfp_header;

typedef struct {
  float (*get)(cfp_ref4f self);
  void (*set)(cfp_ref4f self, float val);
  void (*copy)(cfp_ref4f self, cfp_ref4f src);
  cfp_ptr4f (*ptr)(cfp_ref4f self);
} cfp_ref4f_api;

typedef struct {
  void (*set)(cfp_ptr4f self, float val);
  void (*set_at)(cfp_ptr4f self, float val, ptrdiff_t d);
  float (*get)(cfp_ptr4f self);
  float (*get_at)(cfp_ptr4f self, ptrdiff_t d);
  cfp_ref4f (*ref)(cfp_ptr4f self);
  cfp_ref4f (*ref_at)(cfp_ptr4f self, ptrdiff_t d);
  zfp_bool (*lt)(cfp_ptr4f self, cfp_ptr4f src);
  zfp_bool (*gt)(cfp_ptr4f self, cfp_ptr4f src);
  zfp_bool (*leq)(cfp_ptr4f self, cfp_ptr4f src);
  zfp_bool (*geq)(cfp_ptr4f self, cfp_ptr4f src);
  zfp_bool (*eq)(cfp_ptr4f self, cfp_ptr4f src);
  zfp_bool (*neq)(cfp_ptr4f self, cfp_ptr4f src);
  ptrdiff_t (*distance)(cfp_ptr4f self, cfp_ptr4f src);
  cfp_ptr4f (*next)(cfp_ptr4f self, ptrdiff_t d);
  cfp_ptr4f (*prev)(cfp_ptr4f self, ptrdiff_t d);
  cfp_ptr4f (*inc)(cfp_ptr4f self);
  cfp_ptr4f (*dec)(cfp_ptr4f self);
} cfp_ptr4f_api;

typedef struct {
  void (*set)(cfp_iter4f self, float val);
  float (*get)(cfp_iter4f self);
  cfp_ref4f (*ref)(cfp_iter4f self);
  cfp_ptr4f (*ptr)(cfp_iter4f self);
  cfp_iter4f (*inc)(cfp_iter4f self);
  zfp_bool (*eq)(cfp_iter4f self, cfp_iter4f src);
  zfp_bool (*neq)(cfp_iter4f self, cfp_iter4f src);
  size_t (*i)(cfp_iter4f self);
  size_t (*j)(cfp_iter4f self);
  size_t (*k)(cfp_iter4f self);
  size_t (*l)(cfp_iter4f self);
} cfp_iter4f_api;

typedef struct {
  cfp_array4f (*ctor_default)();
  cfp_array4f (*ctor)(size_t nx, size_t ny, size_t nz, size_t nw, double rate, const float* p, size_t csize);
  cfp_array4f (*ctor_copy)(const cfp_array4f src);
  cfp_array4f (*ctor_header)(const struct cfp_header h, const void* buffer, size_t buffer_size_bytes);
  void (*dtor)(cfp_array4f self);

  void (*deep_copy)(cfp_array4f self, const cfp_array4f src);

  double (*rate)(const cfp_array4f self);
  double (*set_rate)(cfp_array4f self, double rate);
  size_t (*cache_size)(const cfp_array4f self);
  void (*set_cache_size)(cfp_array4f self, size_t csize);
  void (*clear_cache)(const cfp_array4f self);
  void (*flush_cache)(const cfp_array4f self);
  size_t (*compressed_size)(const cfp_array4f self);
  void* (*compressed_data)(const cfp_array4f self);
  size_t (*size)(const cfp_array4f self);
  size_t (*size_x)(const cfp_array4f self);
  size_t (*size_y)(const cfp_array4f self);
  size_t (*size_z)(const cfp_array4f self);
  size_t (*size_w)(const cfp_array4f self);
  void (*resize)(cfp_array4f self, size_t nx, size_t ny, size_t nz, size_t nw, zfp_bool clear);

  void (*get_array)(const cfp_array4f self, float* p);
  void (*set_array)(cfp_array4f self, const float* p);
  float (*get_flat)(const cfp_array4f self, size_t i);
  void (*set_flat)(cfp_array4f self, size_t i, float val);
  float (*get)(const cfp_array4f self, size_t i, size_t j, size_t k, size_t l);
  void (*set)(cfp_array4f self, size_t i, size_t j, size_t k, size_t l, float val);

  cfp_ref4f (*ref)(cfp_array4f self, size_t i, size_t j, size_t k, size_t l);
  cfp_ref4f (*ref_flat)(cfp_array4f self, size_t i);

  cfp_ptr4f (*ptr)(cfp_array4f self, size_t i, size_t j, size_t k, size_t l);
  cfp_ptr4f (*ptr_flat)(cfp_array4f self, size_t i);

  cfp_iter4f (*begin)(cfp_array4f self);
  cfp_iter4f (*end)(cfp_array4f self);

  cfp_ref4f_api reference;
  cfp_ptr4f_api pointer;
  cfp_iter4f_api iterator;
} cfp_array4f_api;

#endif
