#ifndef CFP_ARRAY_1F
#define CFP_ARRAY_1F

#include <stddef.h>
#include "zfp/types.h"

typedef struct {
  void* object;
} cfp_array1f;

typedef struct {
  uint i;
  cfp_array1f array;
} cfp_ref1f;

typedef struct {
  cfp_ref1f reference;
} cfp_ptr1f;

typedef struct {
  uint i;
  cfp_array1f array;
} cfp_iter1f;

typedef struct {
  void (*set)(cfp_ref1f self, float val);
  float (*get)(cfp_ref1f self);
  cfp_ptr1f (*ptr)(cfp_ref1f self);
  void (*copy)(cfp_ref1f self, cfp_ref1f src);
} cfp_ref1f_api;

typedef struct {
  void (*set)(cfp_ptr1f self, float val);
  void (*set_at)(cfp_ptr1f self, float val, ptrdiff_t d);
  float (*get)(cfp_ptr1f self);
  float (*get_at)(cfp_ptr1f self, ptrdiff_t d);
  cfp_ref1f (*ref)(cfp_ptr1f self);
  cfp_ref1f (*ref_at)(cfp_ptr1f self, ptrdiff_t d);
  int (*lt)(cfp_ptr1f self, cfp_ptr1f src);
  int (*gt)(cfp_ptr1f self, cfp_ptr1f src);
  int (*leq)(cfp_ptr1f self, cfp_ptr1f src);
  int (*geq)(cfp_ptr1f self, cfp_ptr1f src);
  int (*eq)(cfp_ptr1f self, cfp_ptr1f src);
  int (*neq)(cfp_ptr1f self, cfp_ptr1f src);
  int (*distance)(cfp_ptr1f self, cfp_ptr1f src);
  cfp_ptr1f (*next)(cfp_ptr1f self, ptrdiff_t d);
  cfp_ptr1f (*prev)(cfp_ptr1f self, ptrdiff_t d);
  cfp_ptr1f (*inc)(cfp_ptr1f self);
  cfp_ptr1f (*dec)(cfp_ptr1f self);
} cfp_ptr1f_api;

typedef struct {
  void (*set)(cfp_iter1f self, float val);
  void (*set_at)(cfp_iter1f self, float val, ptrdiff_t d);
  float (*get)(cfp_iter1f self);
  float (*get_at)(cfp_iter1f self, ptrdiff_t d);
  cfp_ref1f (*ref)(cfp_iter1f self);
  cfp_ref1f (*ref_at)(cfp_iter1f self, ptrdiff_t d);
  cfp_ptr1f (*ptr)(cfp_iter1f self);
  cfp_ptr1f (*ptr_at)(cfp_iter1f self, ptrdiff_t d);
  int (*lt)(cfp_iter1f self, cfp_iter1f src);
  int (*gt)(cfp_iter1f self, cfp_iter1f src);
  int (*leq)(cfp_iter1f self, cfp_iter1f src);
  int (*geq)(cfp_iter1f self, cfp_iter1f src);
  int (*eq)(cfp_iter1f self, cfp_iter1f src);
  int (*neq)(cfp_iter1f self, cfp_iter1f src);
  int (*distance)(cfp_iter1f self, cfp_iter1f src);
  cfp_iter1f (*next)(cfp_iter1f self, ptrdiff_t d);
  cfp_iter1f (*prev)(cfp_iter1f self, ptrdiff_t d);
  cfp_iter1f (*inc)(cfp_iter1f self);
  cfp_iter1f (*dec)(cfp_iter1f self);
  uint (*i)(cfp_iter1f self);
} cfp_iter1f_api;

typedef struct {
  cfp_array1f (*ctor_default)();
  cfp_array1f (*ctor)(uint n, double rate, const float* p, size_t csize);
  cfp_array1f (*ctor_copy)(const cfp_array1f src);
  void (*dtor)(cfp_array1f self);

  void (*deep_copy)(cfp_array1f self, const cfp_array1f src);

  double (*rate)(const cfp_array1f self);
  double (*set_rate)(cfp_array1f self, double rate);
  size_t (*cache_size)(const cfp_array1f self);
  void (*set_cache_size)(cfp_array1f self, size_t csize);
  void (*clear_cache)(const cfp_array1f self);
  void (*flush_cache)(const cfp_array1f self);
  size_t (*compressed_size)(const cfp_array1f self);
  uchar* (*compressed_data)(const cfp_array1f self);
  size_t (*size)(const cfp_array1f self);
  void (*resize)(cfp_array1f self, uint n, int clear);

  void (*get_array)(const cfp_array1f self, float* p);
  void (*set_array)(cfp_array1f self, const float* p);
  float (*get_flat)(const cfp_array1f self, uint i);
  void (*set_flat)(cfp_array1f self, uint i, float val);
  float (*get)(const cfp_array1f self, uint i);
  void (*set)(cfp_array1f self, uint i, float val);

  cfp_ref1f (*ref)(cfp_array1f self, uint i);
  cfp_ref1f (*ref_flat)(cfp_array1f self, uint i);

  cfp_ptr1f (*ptr)(cfp_array1f self, uint i);
  cfp_ptr1f (*ptr_flat)(cfp_array1f self, uint i);

  cfp_iter1f (*begin)(cfp_array1f self);
  cfp_iter1f (*end)(cfp_array1f self);

  cfp_ref1f_api reference;
  cfp_ptr1f_api pointer;
  cfp_iter1f_api iterator;
} cfp_array1f_api;

#endif
