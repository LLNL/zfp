#ifndef CFP_ARRAY_4F_H
#define CFP_ARRAY_4F_H

#include <stddef.h>
#include "zfp.h"

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

typedef struct {
  /* member functions */
  float (*get)(const cfp_ref4f self);
  void (*set)(cfp_ref4f self, float val);
  cfp_ptr4f (*ptr)(cfp_ref4f self);
  void (*copy)(cfp_ref4f self, const cfp_ref4f src);
} cfp_ref4f_api;

typedef struct {
  /* member functions */
  float (*get)(const cfp_ptr4f self);
  float (*get_at)(const cfp_ptr4f self, ptrdiff_t d);
  void (*set)(cfp_ptr4f self, float val);
  void (*set_at)(cfp_ptr4f self, ptrdiff_t d, float val);
  cfp_ref4f (*ref)(cfp_ptr4f self);
  cfp_ref4f (*ref_at)(cfp_ptr4f self, ptrdiff_t d);
  /* non-member functions */
  zfp_bool (*lt)(const cfp_ptr4f lhs, const cfp_ptr4f rhs);
  zfp_bool (*gt)(const cfp_ptr4f lhs, const cfp_ptr4f rhs);
  zfp_bool (*leq)(const cfp_ptr4f lhs, const cfp_ptr4f rhs);
  zfp_bool (*geq)(const cfp_ptr4f lhs, const cfp_ptr4f rhs);
  zfp_bool (*eq)(const cfp_ptr4f lhs, const cfp_ptr4f rhs);
  zfp_bool (*neq)(const cfp_ptr4f lhs, const cfp_ptr4f rhs);
  ptrdiff_t (*distance)(const cfp_ptr4f first, const cfp_ptr4f last);
  cfp_ptr4f (*next)(const cfp_ptr4f p, ptrdiff_t d);
  cfp_ptr4f (*prev)(const cfp_ptr4f p, ptrdiff_t d);
  cfp_ptr4f (*inc)(const cfp_ptr4f p);
  cfp_ptr4f (*dec)(const cfp_ptr4f p);
} cfp_ptr4f_api;

typedef struct {
  /* member functions */
  float (*get)(const cfp_iter4f self);
  float (*get_at)(const cfp_iter4f self, ptrdiff_t d);
  void (*set)(cfp_iter4f self, float val);
  void (*set_at)(cfp_iter4f self, ptrdiff_t d, float val);
  cfp_ref4f (*ref)(cfp_iter4f self);
  cfp_ref4f (*ref_at)(cfp_iter4f self, ptrdiff_t d);
  cfp_ptr4f (*ptr)(cfp_iter4f self);
  cfp_ptr4f (*ptr_at)(cfp_iter4f self, ptrdiff_t d);
  size_t (*i)(const cfp_iter4f self);
  size_t (*j)(const cfp_iter4f self);
  size_t (*k)(const cfp_iter4f self);
  size_t (*l)(const cfp_iter4f self);
  /* non-member functions */
  zfp_bool (*lt)(const cfp_iter4f lhs, const cfp_iter4f rhs);
  zfp_bool (*gt)(const cfp_iter4f lhs, const cfp_iter4f rhs);
  zfp_bool (*leq)(const cfp_iter4f lhs, const cfp_iter4f rhs);
  zfp_bool (*geq)(const cfp_iter4f lhs, const cfp_iter4f rhs);
  zfp_bool (*eq)(const cfp_iter4f lhs, const cfp_iter4f rhs);
  zfp_bool (*neq)(const cfp_iter4f lhs, const cfp_iter4f rhs);
  ptrdiff_t (*distance)(const cfp_iter4f first, const cfp_iter4f last);
  cfp_iter4f (*next)(const cfp_iter4f it, ptrdiff_t d);
  cfp_iter4f (*prev)(const cfp_iter4f it, ptrdiff_t d);
  cfp_iter4f (*inc)(const cfp_iter4f it);
  cfp_iter4f (*dec)(const cfp_iter4f it);
} cfp_iter4f_api;

typedef struct {
  /* constructor/destructor */
  cfp_header (*ctor)(const cfp_array4f a);
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
} cfp_header4f_api;

typedef struct {
  cfp_array4f (*ctor_default)(void);
  cfp_array4f (*ctor)(size_t nx, size_t ny, size_t nz, size_t nw, double rate, const float* p, size_t cache_size);
  cfp_array4f (*ctor_copy)(const cfp_array4f src);
  cfp_array4f (*ctor_header)(const cfp_header h, const void* buffer, size_t buffer_size_bytes);
  void (*dtor)(cfp_array4f self);

  void (*deep_copy)(cfp_array4f self, const cfp_array4f src);

  double (*rate)(const cfp_array4f self);
  double (*set_rate)(cfp_array4f self, double rate);
  size_t (*cache_size)(const cfp_array4f self);
  void (*set_cache_size)(cfp_array4f self, size_t bytes);
  void (*clear_cache)(const cfp_array4f self);
  void (*flush_cache)(const cfp_array4f self);
  size_t (*size_bytes)(const cfp_array4f self, uint mask);
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
  cfp_header4f_api header;
} cfp_array4f_api;

#endif
