#ifndef CFP_ARRAY_3D
#define CFP_ARRAY_3D

#include <stddef.h>
#include "zfp/types.h"
#include "cfptypes.h"

struct cfp_array3d {
  void* object;
};

typedef struct {
  size_t i;
  size_t j;
  size_t k;
  cfp_array3d array;
} cfp_ref3d;

typedef struct {
  cfp_ref3d reference;
} cfp_ptr3d;

typedef struct {
  size_t i;
  size_t j;
  size_t k;
  cfp_array3d array;
} cfp_iter3d;

typedef struct {
  double (*get)(cfp_ref3d self);
  void (*set)(cfp_ref3d self, double val);
  void (*copy)(cfp_ref3d self, cfp_ref3d src);
  cfp_ptr3d (*ptr)(cfp_ref3d self);
} cfp_ref3d_api;

typedef struct {
  void (*set)(cfp_ptr3d self, double val);
  void (*set_at)(cfp_ptr3d self, double val, ptrdiff_t d);
  double (*get)(cfp_ptr3d self);
  double (*get_at)(cfp_ptr3d self, ptrdiff_t d);
  cfp_ref3d (*ref)(cfp_ptr3d self);
  cfp_ref3d (*ref_at)(cfp_ptr3d self, ptrdiff_t d);
  int (*lt)(cfp_ptr3d self, cfp_ptr3d src);
  int (*gt)(cfp_ptr3d self, cfp_ptr3d src);
  int (*leq)(cfp_ptr3d self, cfp_ptr3d src);
  int (*geq)(cfp_ptr3d self, cfp_ptr3d src);
  int (*eq)(cfp_ptr3d self, cfp_ptr3d src);
  int (*neq)(cfp_ptr3d self, cfp_ptr3d src);
  ptrdiff_t (*distance)(cfp_ptr3d self, cfp_ptr3d src);
  cfp_ptr3d (*next)(cfp_ptr3d self, ptrdiff_t d);
  cfp_ptr3d (*prev)(cfp_ptr3d self, ptrdiff_t d);
  cfp_ptr3d (*inc)(cfp_ptr3d self);
  cfp_ptr3d (*dec)(cfp_ptr3d self);
} cfp_ptr3d_api;

typedef struct {
  void (*set)(cfp_iter3d self, double val);
  double (*get)(cfp_iter3d self);
  cfp_ref3d (*ref)(cfp_iter3d self);
  cfp_ptr3d (*ptr)(cfp_iter3d self);
  cfp_iter3d (*inc)(cfp_iter3d self);
  int (*eq)(cfp_iter3d self, cfp_iter3d src);
  int (*neq)(cfp_iter3d self, cfp_iter3d src);
  size_t (*i)(cfp_iter3d self);
  size_t (*j)(cfp_iter3d self);
  size_t (*k)(cfp_iter3d self);
} cfp_iter3d_api;

typedef struct {
  cfp_array3d (*ctor_default)();
  cfp_array3d (*ctor)(size_t nx, size_t ny, size_t nz, double rate, const double* p, size_t csize);
  cfp_array3d (*ctor_copy)(const cfp_array3d src);
  cfp_array3d (*ctor_header)(const cfp_header h, const void* buffer, size_t buffer_size_bytes);
  void (*dtor)(cfp_array3d self);

  void (*deep_copy)(cfp_array3d self, const cfp_array3d src);

  double (*rate)(const cfp_array3d self);
  double (*set_rate)(cfp_array3d self, double rate);
  size_t (*cache_size)(const cfp_array3d self);
  void (*set_cache_size)(cfp_array3d self, size_t csize);
  void (*clear_cache)(const cfp_array3d self);
  void (*flush_cache)(const cfp_array3d self);
  size_t (*compressed_size)(const cfp_array3d self);
  void* (*compressed_data)(const cfp_array3d self);
  size_t (*size)(const cfp_array3d self);
  size_t (*size_x)(const cfp_array3d self);
  size_t (*size_y)(const cfp_array3d self);
  size_t (*size_z)(const cfp_array3d self);
  void (*resize)(cfp_array3d self, size_t nx, size_t ny, size_t nz, int clear);

  void (*get_array)(const cfp_array3d self, double* p);
  void (*set_array)(cfp_array3d self, const double* p);
  double (*get_flat)(const cfp_array3d self, size_t i);
  void (*set_flat)(cfp_array3d self, size_t i, double val);
  double (*get)(const cfp_array3d self, size_t i, size_t j, size_t k);
  void (*set)(cfp_array3d self, size_t i, size_t j, size_t k, double val);

  cfp_ref3d (*ref)(cfp_array3d self, size_t i, size_t j, size_t k);
  cfp_ref3d (*ref_flat)(cfp_array3d self, size_t i);

  cfp_ptr3d (*ptr)(cfp_array3d self, size_t i, size_t j, size_t k);
  cfp_ptr3d (*ptr_flat)(cfp_array3d self, size_t i);

  cfp_iter3d (*begin)(cfp_array3d self);
  cfp_iter3d (*end)(cfp_array3d self);

  cfp_ref3d_api reference;
  cfp_ptr3d_api pointer;
  cfp_iter3d_api iterator;
} cfp_array3d_api;

#endif
