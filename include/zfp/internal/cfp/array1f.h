#ifndef CFP_ARRAY_1F_H
#define CFP_ARRAY_1F_H

#include <stddef.h>
#include "zfp.h"

typedef struct {
  void* object;
} cfp_array1f;

typedef struct {
  cfp_array1f array;
  size_t x;
} cfp_ref1f;

typedef struct {
  cfp_ref1f reference;
} cfp_ptr1f;

typedef struct {
  cfp_array1f array;
  size_t x;
} cfp_iter1f;

typedef struct {
  /* member functions */
  float (*get)(const cfp_ref1f self);
  void (*set)(cfp_ref1f self, float val);
  cfp_ptr1f (*ptr)(cfp_ref1f self);
  void (*copy)(cfp_ref1f self, const cfp_ref1f src);
} cfp_ref1f_api;

typedef struct {
  /* member functions */
  float (*get)(const cfp_ptr1f self);
  float (*get_at)(const cfp_ptr1f self, ptrdiff_t d);
  void (*set)(cfp_ptr1f self, float val);
  void (*set_at)(cfp_ptr1f self, ptrdiff_t d, float val);
  cfp_ref1f (*ref)(cfp_ptr1f self);
  cfp_ref1f (*ref_at)(cfp_ptr1f self, ptrdiff_t d);
  /* non-member functions */
  zfp_bool (*lt)(const cfp_ptr1f lhs, const cfp_ptr1f rhs);
  zfp_bool (*gt)(const cfp_ptr1f lhs, const cfp_ptr1f rhs);
  zfp_bool (*leq)(const cfp_ptr1f lhs, const cfp_ptr1f rhs);
  zfp_bool (*geq)(const cfp_ptr1f lhs, const cfp_ptr1f rhs);
  zfp_bool (*eq)(const cfp_ptr1f lhs, const cfp_ptr1f rhs);
  zfp_bool (*neq)(const cfp_ptr1f lhs, const cfp_ptr1f rhs);
  ptrdiff_t (*distance)(const cfp_ptr1f first, const cfp_ptr1f last);
  cfp_ptr1f (*next)(const cfp_ptr1f p, ptrdiff_t d);
  cfp_ptr1f (*prev)(const cfp_ptr1f p, ptrdiff_t d);
  cfp_ptr1f (*inc)(const cfp_ptr1f p);
  cfp_ptr1f (*dec)(const cfp_ptr1f p);
} cfp_ptr1f_api;

typedef struct {
  /* member functions */
  float (*get)(const cfp_iter1f self);
  float (*get_at)(const cfp_iter1f self, ptrdiff_t d);
  void (*set)(cfp_iter1f self, float val);
  void (*set_at)(cfp_iter1f self, ptrdiff_t d, float val);
  cfp_ref1f (*ref)(cfp_iter1f self);
  cfp_ref1f (*ref_at)(cfp_iter1f self, ptrdiff_t d);
  cfp_ptr1f (*ptr)(cfp_iter1f self);
  cfp_ptr1f (*ptr_at)(cfp_iter1f self, ptrdiff_t d);
  size_t (*i)(const cfp_iter1f self);
  /* non-member functions */
  zfp_bool (*lt)(const cfp_iter1f lhs, const cfp_iter1f rhs);
  zfp_bool (*gt)(const cfp_iter1f lhs, const cfp_iter1f rhs);
  zfp_bool (*leq)(const cfp_iter1f lhs, const cfp_iter1f rhs);
  zfp_bool (*geq)(const cfp_iter1f lhs, const cfp_iter1f rhs);
  zfp_bool (*eq)(const cfp_iter1f lhs, const cfp_iter1f rhs);
  zfp_bool (*neq)(const cfp_iter1f lhs, const cfp_iter1f rhs);
  ptrdiff_t (*distance)(const cfp_iter1f first, const cfp_iter1f last);
  cfp_iter1f (*next)(const cfp_iter1f it, ptrdiff_t d);
  cfp_iter1f (*prev)(const cfp_iter1f it, ptrdiff_t d);
  cfp_iter1f (*inc)(const cfp_iter1f it);
  cfp_iter1f (*dec)(const cfp_iter1f it);
} cfp_iter1f_api;

typedef struct {
  /* constructor/destructor */
  cfp_header (*ctor)(const cfp_array1f a);
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
} cfp_header1f_api;

typedef struct {
  cfp_array1f (*ctor_default)(void);
  cfp_array1f (*ctor)(size_t n, double rate, const float* p, size_t cache_size);
  cfp_array1f (*ctor_copy)(const cfp_array1f src);
  cfp_array1f (*ctor_header)(const cfp_header h, const void* buffer, size_t buffer_size_bytes);
  void (*dtor)(cfp_array1f self);

  void (*deep_copy)(cfp_array1f self, const cfp_array1f src);

  double (*rate)(const cfp_array1f self);
  double (*set_rate)(cfp_array1f self, double rate);
  size_t (*cache_size)(const cfp_array1f self);
  void (*set_cache_size)(cfp_array1f self, size_t bytes);
  void (*clear_cache)(const cfp_array1f self);
  void (*flush_cache)(const cfp_array1f self);
  size_t (*size_bytes)(const cfp_array1f self, uint mask);
  size_t (*compressed_size)(const cfp_array1f self);
  void* (*compressed_data)(const cfp_array1f self);
  size_t (*size)(const cfp_array1f self);
  void (*resize)(cfp_array1f self, size_t n, zfp_bool clear);

  void (*get_array)(const cfp_array1f self, float* p);
  void (*set_array)(cfp_array1f self, const float* p);
  float (*get_flat)(const cfp_array1f self, size_t i);
  void (*set_flat)(cfp_array1f self, size_t i, float val);
  float (*get)(const cfp_array1f self, size_t i);
  void (*set)(cfp_array1f self, size_t i, float val);

  cfp_ref1f (*ref)(cfp_array1f self, size_t i);
  cfp_ref1f (*ref_flat)(cfp_array1f self, size_t i);

  cfp_ptr1f (*ptr)(cfp_array1f self, size_t i);
  cfp_ptr1f (*ptr_flat)(cfp_array1f self, size_t i);

  cfp_iter1f (*begin)(cfp_array1f self);
  cfp_iter1f (*end)(cfp_array1f self);

  cfp_ref1f_api reference;
  cfp_ptr1f_api pointer;
  cfp_iter1f_api iterator;
  cfp_header1f_api header;
} cfp_array1f_api;

#endif
