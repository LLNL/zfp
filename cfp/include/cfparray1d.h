#ifndef CFP_ARRAY_1D
#define CFP_ARRAY_1D

#include <stddef.h>
#include "zfp/types.h"
#include "cfptypes.h"

struct cfp_array1d {
  void* object;
};

typedef struct {
  size_t i;
  cfp_array1d array;
} cfp_ref1d;

typedef struct {
  cfp_ref1d reference;
} cfp_ptr1d;

typedef struct {
  size_t i;
  cfp_array1d array;
} cfp_iter1d;

typedef struct {
  void (*set)(cfp_ref1d self, double val);
  double (*get)(cfp_ref1d self);
  cfp_ptr1d (*ptr)(cfp_ref1d self);
  void (*copy)(cfp_ref1d self, cfp_ref1d src);
} cfp_ref1d_api;

typedef struct {
  void (*set)(cfp_ptr1d self, double val);
  void (*set_at)(cfp_ptr1d self, double val, ptrdiff_t d);
  double (*get)(cfp_ptr1d self);
  double (*get_at)(cfp_ptr1d self, ptrdiff_t d);
  cfp_ref1d (*ref)(cfp_ptr1d self);
  cfp_ref1d (*ref_at)(cfp_ptr1d self, ptrdiff_t d);
  zfp_bool (*lt)(cfp_ptr1d self, cfp_ptr1d src);
  zfp_bool (*gt)(cfp_ptr1d self, cfp_ptr1d src);
  zfp_bool (*leq)(cfp_ptr1d self, cfp_ptr1d src);
  zfp_bool (*geq)(cfp_ptr1d self, cfp_ptr1d src);
  zfp_bool (*eq)(cfp_ptr1d self, cfp_ptr1d src);
  zfp_bool (*neq)(cfp_ptr1d self, cfp_ptr1d src);
  ptrdiff_t (*distance)(cfp_ptr1d self, cfp_ptr1d src);
  cfp_ptr1d (*next)(cfp_ptr1d self, ptrdiff_t d);
  cfp_ptr1d (*prev)(cfp_ptr1d self, ptrdiff_t d);
  cfp_ptr1d (*inc)(cfp_ptr1d self);
  cfp_ptr1d (*dec)(cfp_ptr1d self);
} cfp_ptr1d_api;

typedef struct {
  void (*set)(cfp_iter1d self, double val);
  void (*set_at)(cfp_iter1d self, double val, ptrdiff_t d);
  double (*get)(cfp_iter1d self);
  double (*get_at)(cfp_iter1d self, ptrdiff_t d);
  cfp_ref1d (*ref)(cfp_iter1d self);
  cfp_ref1d (*ref_at)(cfp_iter1d self, ptrdiff_t d);
  cfp_ptr1d (*ptr)(cfp_iter1d self);
  cfp_ptr1d (*ptr_at)(cfp_iter1d self, ptrdiff_t d);
  zfp_bool (*lt)(cfp_iter1d self, cfp_iter1d src);
  zfp_bool (*gt)(cfp_iter1d self, cfp_iter1d src);
  zfp_bool (*leq)(cfp_iter1d self, cfp_iter1d src);
  zfp_bool (*geq)(cfp_iter1d self, cfp_iter1d src);
  zfp_bool (*eq)(cfp_iter1d self, cfp_iter1d src);
  zfp_bool (*neq)(cfp_iter1d self, cfp_iter1d src);
  ptrdiff_t (*distance)(cfp_iter1d self, cfp_iter1d src);
  cfp_iter1d (*next)(cfp_iter1d self, ptrdiff_t d);
  cfp_iter1d (*prev)(cfp_iter1d self, ptrdiff_t d);
  cfp_iter1d (*inc)(cfp_iter1d self);
  cfp_iter1d (*dec)(cfp_iter1d self);
  size_t (*i)(cfp_iter1d self);
} cfp_iter1d_api;

typedef struct {
  cfp_array1d (*ctor_default)();
  cfp_array1d (*ctor)(size_t n, double rate, const double* p, size_t csize);
  cfp_array1d (*ctor_copy)(const cfp_array1d src);
  cfp_array1d (*ctor_header)(const cfp_header h, const void* buffer, size_t buffer_size_bytes);
  void (*dtor)(cfp_array1d self);

  void (*deep_copy)(cfp_array1d self, const cfp_array1d src);

  double (*rate)(const cfp_array1d self);
  double (*set_rate)(cfp_array1d self, double rate);
  size_t (*cache_size)(const cfp_array1d self);
  void (*set_cache_size)(cfp_array1d self, size_t csize);
  void (*clear_cache)(const cfp_array1d self);
  void (*flush_cache)(const cfp_array1d self);
  size_t (*compressed_size)(const cfp_array1d self);
  void* (*compressed_data)(const cfp_array1d self);
  size_t (*size)(const cfp_array1d self);
  void (*resize)(cfp_array1d self, size_t n, zfp_bool clear);

  void (*get_array)(const cfp_array1d self, double* p);
  void (*set_array)(cfp_array1d self, const double* p);
  double (*get_flat)(const cfp_array1d self, size_t i);
  void (*set_flat)(cfp_array1d self, size_t i, double val);
  double (*get)(const cfp_array1d self, size_t i);
  void (*set)(cfp_array1d self, size_t i, double val);

  cfp_ref1d (*ref)(cfp_array1d self, size_t i);
  cfp_ref1d (*ref_flat)(cfp_array1d self, size_t i);

  cfp_ptr1d (*ptr)(cfp_array1d self, size_t i);
  cfp_ptr1d (*ptr_flat)(cfp_array1d self, size_t i);

  cfp_iter1d (*begin)(cfp_array1d self);
  cfp_iter1d (*end)(cfp_array1d self);

  cfp_ref1d_api reference;
  cfp_ptr1d_api pointer;
  cfp_iter1d_api iterator;
} cfp_array1d_api;

#endif
