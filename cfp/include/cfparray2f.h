#ifndef CFP_ARRAY_2F
#define CFP_ARRAY_2F

#include <stddef.h>
#include "zfp.h"
#include "zfp/types.h"

typedef struct {
  void* object;
} cfp_array2f;

typedef struct {
  size_t i;
  size_t j;
  cfp_array2f array;
} cfp_ref2f;

typedef struct {
  cfp_ref2f reference;
} cfp_ptr2f;

typedef struct {
  size_t i;
  size_t j;
  cfp_array2f array;
} cfp_iter2f;

struct cfp_header;

typedef struct {
  float (*get)(cfp_ref2f self);
  void (*set)(cfp_ref2f self, float val);
  void (*copy)(cfp_ref2f self, cfp_ref2f src);
  cfp_ptr2f (*ptr)(cfp_ref2f self);
} cfp_ref2f_api;

typedef struct {
  void (*set)(cfp_ptr2f self, float val);
  void (*set_at)(cfp_ptr2f self, float val, ptrdiff_t d);
  float (*get)(cfp_ptr2f self);
  float (*get_at)(cfp_ptr2f self, ptrdiff_t d);
  cfp_ref2f (*ref)(cfp_ptr2f self);
  cfp_ref2f (*ref_at)(cfp_ptr2f self, ptrdiff_t d);
  zfp_bool (*lt)(cfp_ptr2f self, cfp_ptr2f src);
  zfp_bool (*gt)(cfp_ptr2f self, cfp_ptr2f src);
  zfp_bool (*leq)(cfp_ptr2f self, cfp_ptr2f src);
  zfp_bool (*geq)(cfp_ptr2f self, cfp_ptr2f src);
  zfp_bool (*eq)(cfp_ptr2f self, cfp_ptr2f src);
  zfp_bool (*neq)(cfp_ptr2f self, cfp_ptr2f src);
  ptrdiff_t (*distance)(cfp_ptr2f self, cfp_ptr2f src);
  cfp_ptr2f (*next)(cfp_ptr2f self, ptrdiff_t);
  cfp_ptr2f (*prev)(cfp_ptr2f self, ptrdiff_t);
  cfp_ptr2f (*inc)(cfp_ptr2f self);
  cfp_ptr2f (*dec)(cfp_ptr2f self);
} cfp_ptr2f_api;

typedef struct {
  void (*set)(cfp_iter2f self, float value);
  float (*get)(cfp_iter2f self);
  cfp_ref2f (*ref)(cfp_iter2f self);
  cfp_ptr2f (*ptr)(cfp_iter2f self);
  cfp_iter2f (*inc)(cfp_iter2f self);
  zfp_bool (*eq)(cfp_iter2f self, cfp_iter2f src);
  zfp_bool (*neq)(cfp_iter2f self, cfp_iter2f src);
  size_t (*i)(cfp_iter2f self);
  size_t (*j)(cfp_iter2f self);
} cfp_iter2f_api;

typedef struct {
  cfp_array2f (*ctor_default)();
  cfp_array2f (*ctor)(size_t nx, size_t ny, double rate, const float* p, size_t csize);
  cfp_array2f (*ctor_copy)(const cfp_array2f src);
  cfp_array2f (*ctor_header)(const struct cfp_header h, const void* buffer, size_t buffer_size_bytes);
  void (*dtor)(cfp_array2f self);

  void (*deep_copy)(cfp_array2f self, const cfp_array2f src);

  double (*rate)(const cfp_array2f self);
  double (*set_rate)(cfp_array2f self, double rate);
  size_t (*cache_size)(const cfp_array2f self);
  void (*set_cache_size)(cfp_array2f self, size_t csize);
  void (*clear_cache)(const cfp_array2f self);
  void (*flush_cache)(const cfp_array2f self);
  size_t (*compressed_size)(const cfp_array2f self);
  void* (*compressed_data)(const cfp_array2f self);
  size_t (*size)(const cfp_array2f self);
  size_t (*size_x)(const cfp_array2f self);
  size_t (*size_y)(const cfp_array2f self);
  void (*resize)(cfp_array2f self, size_t nx, size_t ny, zfp_bool clear);

  void (*get_array)(const cfp_array2f self, float* p);
  void (*set_array)(cfp_array2f self, const float* p);
  float (*get_flat)(const cfp_array2f self, size_t i);
  void (*set_flat)(cfp_array2f self, size_t i, float val);
  float (*get)(const cfp_array2f self, size_t i, size_t j);
  void (*set)(cfp_array2f self, size_t i, size_t j, float val);

  cfp_ref2f (*ref)(cfp_array2f self, size_t i, size_t j);
  cfp_ref2f (*ref_flat)(cfp_array2f self, size_t i);

  cfp_ptr2f (*ptr)(cfp_array2f self, size_t i, size_t j);
  cfp_ptr2f (*ptr_flat)(cfp_array2f self, size_t i);

  cfp_iter2f (*begin)(cfp_array2f self);
  cfp_iter2f (*end)(cfp_array2f self);

  cfp_ref2f_api reference;
  cfp_ptr2f_api pointer;
  cfp_iter2f_api iterator;
} cfp_array2f_api;

#endif
