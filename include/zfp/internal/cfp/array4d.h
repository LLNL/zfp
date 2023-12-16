#ifndef CFP_ARRAY_4D_H
#define CFP_ARRAY_4D_H

#include <stddef.h>
#include "zfp.h"

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

typedef struct {
  /* member functions */
  double (*get)(const cfp_ref4d self);
  void (*set)(cfp_ref4d self, double val);
  cfp_ptr4d (*ptr)(cfp_ref4d self);
  void (*copy)(cfp_ref4d self, const cfp_ref4d src);
} cfp_ref4d_api;

typedef struct {
  /* member functions */
  double (*get)(const cfp_ptr4d self);
  double (*get_at)(const cfp_ptr4d self, ptrdiff_t d);
  void (*set)(cfp_ptr4d self, double val);
  void (*set_at)(cfp_ptr4d self, ptrdiff_t d, double val);
  cfp_ref4d (*ref)(cfp_ptr4d self);
  cfp_ref4d (*ref_at)(cfp_ptr4d self, ptrdiff_t d);
  /* non-member functions */
  zfp_bool (*lt)(const cfp_ptr4d lhs, const cfp_ptr4d rhs);
  zfp_bool (*gt)(const cfp_ptr4d lhs, const cfp_ptr4d rhs);
  zfp_bool (*leq)(const cfp_ptr4d lhs, const cfp_ptr4d rhs);
  zfp_bool (*geq)(const cfp_ptr4d lhs, const cfp_ptr4d rhs);
  zfp_bool (*eq)(const cfp_ptr4d lhs, const cfp_ptr4d rhs);
  zfp_bool (*neq)(const cfp_ptr4d lhs, const cfp_ptr4d rhs);
  ptrdiff_t (*distance)(const cfp_ptr4d first, const cfp_ptr4d last);
  cfp_ptr4d (*next)(const cfp_ptr4d p, ptrdiff_t d);
  cfp_ptr4d (*prev)(const cfp_ptr4d p, ptrdiff_t d);
  cfp_ptr4d (*inc)(const cfp_ptr4d p);
  cfp_ptr4d (*dec)(const cfp_ptr4d p);
} cfp_ptr4d_api;

typedef struct {
  /* member functions */
  double (*get)(const cfp_iter4d self);
  double (*get_at)(const cfp_iter4d self, ptrdiff_t d);
  void (*set)(cfp_iter4d self, double val);
  void (*set_at)(cfp_iter4d self, ptrdiff_t d, double val);
  cfp_ref4d (*ref)(cfp_iter4d self);
  cfp_ref4d (*ref_at)(cfp_iter4d self, ptrdiff_t d);
  cfp_ptr4d (*ptr)(cfp_iter4d self);
  cfp_ptr4d (*ptr_at)(cfp_iter4d self, ptrdiff_t d);
  size_t (*i)(const cfp_iter4d self);
  size_t (*j)(const cfp_iter4d self);
  size_t (*k)(const cfp_iter4d self);
  size_t (*l)(const cfp_iter4d self);
  /* non-member functions */
  zfp_bool (*lt)(const cfp_iter4d lhs, const cfp_iter4d rhs);
  zfp_bool (*gt)(const cfp_iter4d lhs, const cfp_iter4d rhs);
  zfp_bool (*leq)(const cfp_iter4d lhs, const cfp_iter4d rhs);
  zfp_bool (*geq)(const cfp_iter4d lhs, const cfp_iter4d rhs);
  zfp_bool (*eq)(const cfp_iter4d lhs, const cfp_iter4d rhs);
  zfp_bool (*neq)(const cfp_iter4d lhs, const cfp_iter4d rhs);
  ptrdiff_t (*distance)(const cfp_iter4d first, const cfp_iter4d last);
  cfp_iter4d (*next)(const cfp_iter4d it, ptrdiff_t d);
  cfp_iter4d (*prev)(const cfp_iter4d it, ptrdiff_t d);
  cfp_iter4d (*inc)(const cfp_iter4d it);
  cfp_iter4d (*dec)(const cfp_iter4d it);
} cfp_iter4d_api;

typedef struct {
  /* constructor/destructor */
  cfp_header (*ctor)(const cfp_array4d a);
  cfp_header (*ctor_buffer)(const void* data, size_t size);
  void (*dtor)(cfp_header self);
  /* array metadata */
  zfp_type (*scalar_type)(const cfp_header self);
  uint (*dimensionality)(const cfp_header self);
  size_t (*size_x)(const cfp_header self);
  size_t (*size_y)(const cfp_header self);
  size_t (*size_z)(const cfp_header self);
  size_t (*size_w)(const cfp_header self);
  double (*rate)(const cfp_header self);
  /* header payload: data pointer and byte size */
  const void* (*data)(const cfp_header self);
  size_t (*size_bytes)(const cfp_header self, uint mask);
} cfp_header4d_api;

typedef struct {
  cfp_array4d (*ctor_default)(void);
  cfp_array4d (*ctor)(size_t nx, size_t ny, size_t nz, size_t nw, double rate, const double* p, size_t cache_size);
  cfp_array4d (*ctor_copy)(const cfp_array4d src);
  cfp_array4d (*ctor_header)(const cfp_header h, const void* buffer, size_t buffer_size_bytes);
  void (*dtor)(cfp_array4d self);

  void (*deep_copy)(cfp_array4d self, const cfp_array4d src);

  double (*rate)(const cfp_array4d self);
  double (*set_rate)(cfp_array4d self, double rate);
  size_t (*cache_size)(const cfp_array4d self);
  void (*set_cache_size)(cfp_array4d self, size_t bytes);
  void (*clear_cache)(const cfp_array4d self);
  void (*flush_cache)(const cfp_array4d self);
  size_t (*size_bytes)(const cfp_array4d self, uint mask);
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
  cfp_header4d_api header;
} cfp_array4d_api;

#endif
