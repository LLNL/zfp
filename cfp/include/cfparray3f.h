#ifndef CFP_ARRAY_3F
#define CFP_ARRAY_3F

#include <stddef.h>
#include "zfp/types.h"
#include "cfptypes.h"

struct cfp_array3f {
  void* object;
};

typedef struct {
  size_t i;
  size_t j;
  size_t k;
  cfp_array3f array;
} cfp_ref3f;

typedef struct {
  cfp_ref3f reference;
} cfp_ptr3f;

typedef struct {
  size_t i;
  size_t j;
  size_t k;
  cfp_array3f array;
} cfp_iter3f;

typedef struct {
  float (*get)(cfp_ref3f self);
  void (*set)(cfp_ref3f self, float val);
  void (*copy)(cfp_ref3f self, cfp_ref3f src);
  cfp_ptr3f (*ptr)(cfp_ref3f self);
} cfp_ref3f_api;

typedef struct {
  void (*set)(cfp_ptr3f self, float val);
  void (*set_at)(cfp_ptr3f self, float val, ptrdiff_t d);
  float (*get)(cfp_ptr3f self);
  float (*get_at)(cfp_ptr3f self, ptrdiff_t d);
  cfp_ref3f (*ref)(cfp_ptr3f self);
  cfp_ref3f (*ref_at)(cfp_ptr3f self, ptrdiff_t d);
  int (*lt)(cfp_ptr3f self, cfp_ptr3f src);
  int (*gt)(cfp_ptr3f self, cfp_ptr3f src);
  int (*leq)(cfp_ptr3f self, cfp_ptr3f src);
  int (*geq)(cfp_ptr3f self, cfp_ptr3f src);
  int (*eq)(cfp_ptr3f self, cfp_ptr3f src);
  int (*neq)(cfp_ptr3f self, cfp_ptr3f src);
  ptrdiff_t (*distance)(cfp_ptr3f self, cfp_ptr3f src);
  cfp_ptr3f (*next)(cfp_ptr3f self, ptrdiff_t d);
  cfp_ptr3f (*prev)(cfp_ptr3f self, ptrdiff_t d);
  cfp_ptr3f (*inc)(cfp_ptr3f self);
  cfp_ptr3f (*dec)(cfp_ptr3f self);
} cfp_ptr3f_api;

typedef struct {
  void (*set)(cfp_iter3f self, float val);
  float (*get)(cfp_iter3f self);
  cfp_ref3f (*ref)(cfp_iter3f self);
  cfp_ptr3f (*ptr)(cfp_iter3f self);
  cfp_iter3f (*inc)(cfp_iter3f self);
  int (*eq)(cfp_iter3f self, cfp_iter3f src);
  int (*neq)(cfp_iter3f self, cfp_iter3f src);
  size_t (*i)(cfp_iter3f self);
  size_t (*j)(cfp_iter3f self);
  size_t (*k)(cfp_iter3f self);
} cfp_iter3f_api;

typedef struct {
  cfp_array3f (*ctor_default)();
  cfp_array3f (*ctor)(size_t nx, size_t ny, size_t nz, double rate, const float* p, size_t csize);
  cfp_array3f (*ctor_copy)(const cfp_array3f src);
  cfp_array3f (*ctor_header)(const cfp_header h, const void* buffer, size_t buffer_size_bytes);
  void (*dtor)(cfp_array3f self);

  void (*deep_copy)(cfp_array3f self, const cfp_array3f src);

  double (*rate)(const cfp_array3f self);
  double (*set_rate)(cfp_array3f self, double rate);
  size_t (*cache_size)(const cfp_array3f self);
  void (*set_cache_size)(cfp_array3f self, size_t csize);
  void (*clear_cache)(const cfp_array3f self);
  void (*flush_cache)(const cfp_array3f self);
  size_t (*compressed_size)(const cfp_array3f self);
  void* (*compressed_data)(const cfp_array3f self);
  size_t (*size)(const cfp_array3f self);
  size_t (*size_x)(const cfp_array3f self);
  size_t (*size_y)(const cfp_array3f self);
  size_t (*size_z)(const cfp_array3f self);
  void (*resize)(cfp_array3f self, size_t nx, size_t ny, size_t nz, int clear);

  void (*get_array)(const cfp_array3f self, float* p);
  void (*set_array)(cfp_array3f self, const float* p);
  float (*get_flat)(const cfp_array3f self, size_t i);
  void (*set_flat)(cfp_array3f self, size_t i, float val);
  float (*get)(const cfp_array3f self, size_t i, size_t j, size_t k);
  void (*set)(cfp_array3f self, size_t i, size_t j, size_t k, float val);

  cfp_ref3f (*ref)(cfp_array3f self, size_t i, size_t j, size_t k);
  cfp_ref3f (*ref_flat)(cfp_array3f self, size_t i);

  cfp_ptr3f (*ptr)(cfp_array3f self, size_t i, size_t j, size_t k);
  cfp_ptr3f (*ptr_flat)(cfp_array3f self, size_t i);

  cfp_iter3f (*begin)(cfp_array3f self);
  cfp_iter3f (*end)(cfp_array3f self);

  cfp_ref3f_api reference;
  cfp_ptr3f_api pointer;
  cfp_iter3f_api iterator;
} cfp_array3f_api;

#endif
