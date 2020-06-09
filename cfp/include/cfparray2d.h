#ifndef CFP_ARRAY_2D
#define CFP_ARRAY_2D

#include <stddef.h>
#include "zfp/types.h"
#include "cfptypes.h"

struct cfp_array2d {
  void* object;
};

typedef struct {
  size_t i;
  size_t j;
  cfp_array2d array;
} cfp_ref2d;

typedef struct {
  cfp_ref2d reference;
} cfp_ptr2d;

typedef struct {
  size_t i;
  size_t j;
  cfp_array2d array;
} cfp_iter2d;

typedef struct {
  double (*get)(cfp_ref2d self);
  void (*set)(cfp_ref2d self, double val);
  void (*copy)(cfp_ref2d self, cfp_ref2d src);
  cfp_ptr2d (*ptr)(cfp_ref2d self);
} cfp_ref2d_api;

typedef struct {
  void (*set)(cfp_ptr2d self, double val);
  void (*set_at)(cfp_ptr2d self, double val, ptrdiff_t d);
  double (*get)(cfp_ptr2d self);
  double (*get_at)(cfp_ptr2d self, ptrdiff_t d);
  cfp_ref2d (*ref)(cfp_ptr2d self);
  cfp_ref2d (*ref_at)(cfp_ptr2d self, ptrdiff_t d);
  int (*lt)(cfp_ptr2d self, cfp_ptr2d src);
  int (*gt)(cfp_ptr2d self, cfp_ptr2d src);
  int (*leq)(cfp_ptr2d self, cfp_ptr2d src);
  int (*geq)(cfp_ptr2d self, cfp_ptr2d src);
  int (*eq)(cfp_ptr2d self, cfp_ptr2d src);
  int (*neq)(cfp_ptr2d self, cfp_ptr2d src);
  ptrdiff_t (*distance)(cfp_ptr2d self, cfp_ptr2d src);
  cfp_ptr2d (*next)(cfp_ptr2d self, ptrdiff_t);
  cfp_ptr2d (*prev)(cfp_ptr2d self, ptrdiff_t);
  cfp_ptr2d (*inc)(cfp_ptr2d self);
  cfp_ptr2d (*dec)(cfp_ptr2d self);
} cfp_ptr2d_api;

typedef struct {
  void (*set)(cfp_iter2d self, double value);
  double (*get)(cfp_iter2d self);
  cfp_ref2d (*ref)(cfp_iter2d self);
  cfp_ptr2d (*ptr)(cfp_iter2d self);
  cfp_iter2d (*inc)(cfp_iter2d self);
  int (*eq)(cfp_iter2d self, cfp_iter2d src);
  int (*neq)(cfp_iter2d self, cfp_iter2d src);
  size_t (*i)(cfp_iter2d self);
  size_t (*j)(cfp_iter2d self);
} cfp_iter2d_api;

typedef struct {
  cfp_array2d (*ctor_default)();
  cfp_array2d (*ctor)(size_t nx, size_t ny, double rate, const double* p, size_t csize);
  cfp_array2d (*ctor_copy)(const cfp_array2d src);
  cfp_array2d (*ctor_header)(const cfp_header h);
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
  size_t (*size_x)(const cfp_array2d self);
  size_t (*size_y)(const cfp_array2d self);
  void (*resize)(cfp_array2d self, size_t nx, size_t ny, int clear);

  void (*get_array)(const cfp_array2d self, double* p);
  void (*set_array)(cfp_array2d self, const double* p);
  double (*get_flat)(const cfp_array2d self, size_t i);
  void (*set_flat)(cfp_array2d self, size_t i, double val);
  double (*get)(const cfp_array2d self, size_t i, size_t j);
  void (*set)(cfp_array2d self, size_t i, size_t j, double val);

  cfp_ref2d (*ref)(cfp_array2d self, size_t i, size_t j);
  cfp_ref2d (*ref_flat)(cfp_array2d self, size_t i);

  cfp_ptr2d (*ptr)(cfp_array2d self, size_t i, size_t j);
  cfp_ptr2d (*ptr_flat)(cfp_array2d self, size_t i);

  cfp_iter2d (*begin)(cfp_array2d self);
  cfp_iter2d (*end)(cfp_array2d self);

  cfp_ref2d_api reference;
  cfp_ptr2d_api pointer;
  cfp_iter2d_api iterator;
} cfp_array2d_api;

#endif
