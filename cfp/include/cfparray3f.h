#ifndef CFP_ARRAY_3F
#define CFP_ARRAY_3F

#include <stddef.h>
#include "zfp/types.h"

typedef struct {
  void* object;
} cfp_array3f;

typedef struct {
  uint i;
  uint j;
  uint k;
  cfp_array3f array;
} cfp_ref3f;

typedef struct {
  cfp_ref3f reference;
} cfp_ptr3f;

typedef struct {
  uint i;
  uint j;
  uint k;
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
  int (*distance)(cfp_ptr3f self, cfp_ptr3f src);
  cfp_ptr3f (*next)(cfp_ptr3f self, ptrdiff_t d);
  cfp_ptr3f (*prev)(cfp_ptr3f self, ptrdiff_t d);
  cfp_ptr3f (*inc)(cfp_ptr3f self);
  cfp_ptr3f (*dec)(cfp_ptr3f self);
} cfp_ptr3f_api;

typedef struct {
  cfp_ref3f (*ref)(cfp_iter3f self);
  cfp_iter3f (*next)(cfp_iter3f self, ptrdiff_t d);
  cfp_iter3f (*inc)(cfp_iter3f self);
  int (*eq)(cfp_iter3f self, cfp_iter3f src);
  uint (*i)(cfp_iter3f self);
  uint (*j)(cfp_iter3f self);
  uint (*k)(cfp_iter3f self);
} cfp_iter3f_api;

typedef struct {
  cfp_array3f (*ctor_default)();
  cfp_array3f (*ctor)(uint nx, uint ny, uint nz, double rate, const float* p, size_t csize);
  cfp_array3f (*ctor_copy)(const cfp_array3f src);
  void (*dtor)(cfp_array3f self);

  void (*deep_copy)(cfp_array3f self, const cfp_array3f src);

  double (*rate)(const cfp_array3f self);
  double (*set_rate)(cfp_array3f self, double rate);
  size_t (*cache_size)(const cfp_array3f self);
  void (*set_cache_size)(cfp_array3f self, size_t csize);
  void (*clear_cache)(const cfp_array3f self);
  void (*flush_cache)(const cfp_array3f self);
  size_t (*compressed_size)(const cfp_array3f self);
  uchar* (*compressed_data)(const cfp_array3f self);
  size_t (*size)(const cfp_array3f self);
  uint (*size_x)(const cfp_array3f self);
  uint (*size_y)(const cfp_array3f self);
  uint (*size_z)(const cfp_array3f self);
  void (*resize)(cfp_array3f self, uint nx, uint ny, uint nz, int clear);

  void (*get_array)(const cfp_array3f self, float* p);
  void (*set_array)(cfp_array3f self, const float* p);
  float (*get_flat)(const cfp_array3f self, uint i);
  void (*set_flat)(cfp_array3f self, uint i, float val);
  float (*get)(const cfp_array3f self, uint i, uint j, uint k);
  void (*set)(cfp_array3f self, uint i, uint j, uint k, float val);

  cfp_ref3f (*ref)(cfp_array3f self, uint i, uint j, uint k);
  cfp_ref3f (*ref_flat)(cfp_array3f self, uint i);

  cfp_ptr3f (*ptr)(cfp_array3f self, uint i, uint j, uint k);
  cfp_ptr3f (*ptr_flat)(cfp_array3f self, uint i);

  cfp_iter3f (*begin)(cfp_array3f self);
  cfp_iter3f (*end)(cfp_array3f self);

  cfp_ref3f_api reference;
  cfp_ptr3f_api pointer;
  cfp_iter3f_api iterator;
} cfp_array3f_api;

#endif
