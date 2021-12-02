#ifndef CFP_ARRAY_3F
#define CFP_ARRAY_3F

#include "cfptypes.h"
#include <stddef.h>
#include "zfp.h"

/* Cfp Types */
CFP_DECL_CONTAINER(array, 3, f)
CFP_DECL_CONTAINER(view, 3, f)
CFP_DECL_CONTAINER(flat_view, 3, f)
CFP_DECL_CONTAINER(private_view, 3, f)

CFP_DECL_ACCESSOR(ref_array, 3, f)
CFP_DECL_ACCESSOR(ptr_array, 3, f)
CFP_DECL_ACCESSOR(iter_array, 3, f)

CFP_DECL_ACCESSOR(ref_view, 3, f)
CFP_DECL_ACCESSOR(ptr_view, 3, f)
CFP_DECL_ACCESSOR(iter_view, 3, f)

CFP_DECL_ACCESSOR(ref_flat_view, 3, f)
CFP_DECL_ACCESSOR(ptr_flat_view, 3, f)
CFP_DECL_ACCESSOR(iter_flat_view, 3, f)

CFP_DECL_ACCESSOR(ref_private_view, 3, f)
CFP_DECL_ACCESSOR(ptr_private_view, 3, f)
CFP_DECL_ACCESSOR(iter_private_view, 3, f)

typedef cfp_ref_array3f cfp_ref3f;
typedef cfp_ptr_array3f cfp_ptr3f;
typedef cfp_iter_array3f cfp_iter3f;

/* API */
typedef struct {
  /* member functions */
  float (*get)(const cfp_ref_array3f self);
  void (*set)(cfp_ref_array3f self, float val);
  cfp_ptr_array3f (*ptr)(cfp_ref_array3f self);
  void (*copy)(cfp_ref_array3f self, const cfp_ref_array3f src);
} cfp_ref_array3f_api;

typedef struct {
  /* member functions */
  float (*get)(const cfp_ref_view3f self);
  void (*set)(cfp_ref_view3f self, float val);
  cfp_ptr_view3f (*ptr)(cfp_ref_view3f self);
  void (*copy)(cfp_ref_view3f self, const cfp_ref_view3f src);
} cfp_ref_view3f_api;

typedef struct {
  /* member functions */
  float (*get)(const cfp_ref_flat_view3f self);
  void (*set)(cfp_ref_flat_view3f self, float val);
  cfp_ptr_flat_view3f (*ptr)(cfp_ref_flat_view3f self);
  void (*copy)(cfp_ref_flat_view3f self, const cfp_ref_flat_view3f src);
} cfp_ref_flat_view3f_api;

typedef struct {
  /* member functions */
  float (*get)(const cfp_ref_private_view3f self);
  void (*set)(cfp_ref_private_view3f self, float val);
  cfp_ptr_private_view3f (*ptr)(cfp_ref_private_view3f self);
  void (*copy)(cfp_ref_private_view3f self, const cfp_ref_private_view3f src);
} cfp_ref_private_view3f_api;

typedef struct {
  /* member functions */
  float (*get)(const cfp_ptr_array3f self);
  float (*get_at)(const cfp_ptr_array3f self, ptrdiff_t d);
  void (*set)(cfp_ptr_array3f self, float val);
  void (*set_at)(cfp_ptr_array3f self, ptrdiff_t d, float val);
  cfp_ref_array3f (*ref)(cfp_ptr_array3f self);
  cfp_ref_array3f (*ref_at)(cfp_ptr_array3f self, ptrdiff_t d);
  /* non-member functions */
  zfp_bool (*lt)(const cfp_ptr_array3f lhs, const cfp_ptr_array3f rhs);
  zfp_bool (*gt)(const cfp_ptr_array3f lhs, const cfp_ptr_array3f rhs);
  zfp_bool (*leq)(const cfp_ptr_array3f lhs, const cfp_ptr_array3f rhs);
  zfp_bool (*geq)(const cfp_ptr_array3f lhs, const cfp_ptr_array3f rhs);
  zfp_bool (*eq)(const cfp_ptr_array3f lhs, const cfp_ptr_array3f rhs);
  zfp_bool (*neq)(const cfp_ptr_array3f lhs, const cfp_ptr_array3f rhs);
  ptrdiff_t (*distance)(const cfp_ptr_array3f first, const cfp_ptr_array3f last);
  cfp_ptr_array3f (*next)(const cfp_ptr_array3f p, ptrdiff_t d);
  cfp_ptr_array3f (*prev)(const cfp_ptr_array3f p, ptrdiff_t d);
  cfp_ptr_array3f (*inc)(const cfp_ptr_array3f p);
  cfp_ptr_array3f (*dec)(const cfp_ptr_array3f p);
} cfp_ptr_array3f_api;

typedef struct {
  /* member functions */
  float (*get)(const cfp_ptr_view3f self);
  float (*get_at)(const cfp_ptr_view3f self, ptrdiff_t d);
  void (*set)(cfp_ptr_view3f self, float val);
  void (*set_at)(cfp_ptr_view3f self, ptrdiff_t d, float val);
  cfp_ref_view3f (*ref)(cfp_ptr_view3f self);
  cfp_ref_view3f (*ref_at)(cfp_ptr_view3f self, ptrdiff_t d);
  /* non-member functions */
  zfp_bool (*lt)(const cfp_ptr_view3f lhs, const cfp_ptr_view3f rhs);
  zfp_bool (*gt)(const cfp_ptr_view3f lhs, const cfp_ptr_view3f rhs);
  zfp_bool (*leq)(const cfp_ptr_view3f lhs, const cfp_ptr_view3f rhs);
  zfp_bool (*geq)(const cfp_ptr_view3f lhs, const cfp_ptr_view3f rhs);
  zfp_bool (*eq)(const cfp_ptr_view3f lhs, const cfp_ptr_view3f rhs);
  zfp_bool (*neq)(const cfp_ptr_view3f lhs, const cfp_ptr_view3f rhs);
  ptrdiff_t (*distance)(const cfp_ptr_view3f first, const cfp_ptr_view3f last);
  cfp_ptr_view3f (*next)(const cfp_ptr_view3f p, ptrdiff_t d);
  cfp_ptr_view3f (*prev)(const cfp_ptr_view3f p, ptrdiff_t d);
  cfp_ptr_view3f (*inc)(const cfp_ptr_view3f p);
  cfp_ptr_view3f (*dec)(const cfp_ptr_view3f p);
} cfp_ptr_view3f_api;

typedef struct {
  /* member functions */
  float (*get)(const cfp_ptr_flat_view3f self);
  float (*get_at)(const cfp_ptr_flat_view3f self, ptrdiff_t d);
  void (*set)(cfp_ptr_flat_view3f self, float val);
  void (*set_at)(cfp_ptr_flat_view3f self, ptrdiff_t d, float val);
  cfp_ref_flat_view3f (*ref)(cfp_ptr_flat_view3f self);
  cfp_ref_flat_view3f (*ref_at)(cfp_ptr_flat_view3f self, ptrdiff_t d);
  /* non-member functions */
  zfp_bool (*lt)(const cfp_ptr_flat_view3f lhs, const cfp_ptr_flat_view3f rhs);
  zfp_bool (*gt)(const cfp_ptr_flat_view3f lhs, const cfp_ptr_flat_view3f rhs);
  zfp_bool (*leq)(const cfp_ptr_flat_view3f lhs, const cfp_ptr_flat_view3f rhs);
  zfp_bool (*geq)(const cfp_ptr_flat_view3f lhs, const cfp_ptr_flat_view3f rhs);
  zfp_bool (*eq)(const cfp_ptr_flat_view3f lhs, const cfp_ptr_flat_view3f rhs);
  zfp_bool (*neq)(const cfp_ptr_flat_view3f lhs, const cfp_ptr_flat_view3f rhs);
  ptrdiff_t (*distance)(const cfp_ptr_flat_view3f first, const cfp_ptr_flat_view3f last);
  cfp_ptr_flat_view3f (*next)(const cfp_ptr_flat_view3f p, ptrdiff_t d);
  cfp_ptr_flat_view3f (*prev)(const cfp_ptr_flat_view3f p, ptrdiff_t d);
  cfp_ptr_flat_view3f (*inc)(const cfp_ptr_flat_view3f p);
  cfp_ptr_flat_view3f (*dec)(const cfp_ptr_flat_view3f p);
} cfp_ptr_flat_view3f_api;

typedef struct {
  /* member functions */
  float (*get)(const cfp_ptr_private_view3f self);
  float (*get_at)(const cfp_ptr_private_view3f self, ptrdiff_t d);
  void (*set)(cfp_ptr_private_view3f self, float val);
  void (*set_at)(cfp_ptr_private_view3f self, ptrdiff_t d, float val);
  cfp_ref_private_view3f (*ref)(cfp_ptr_private_view3f self);
  cfp_ref_private_view3f (*ref_at)(cfp_ptr_private_view3f self, ptrdiff_t d);
  /* non-member functions */
  zfp_bool (*lt)(const cfp_ptr_private_view3f lhs, const cfp_ptr_private_view3f rhs);
  zfp_bool (*gt)(const cfp_ptr_private_view3f lhs, const cfp_ptr_private_view3f rhs);
  zfp_bool (*leq)(const cfp_ptr_private_view3f lhs, const cfp_ptr_private_view3f rhs);
  zfp_bool (*geq)(const cfp_ptr_private_view3f lhs, const cfp_ptr_private_view3f rhs);
  zfp_bool (*eq)(const cfp_ptr_private_view3f lhs, const cfp_ptr_private_view3f rhs);
  zfp_bool (*neq)(const cfp_ptr_private_view3f lhs, const cfp_ptr_private_view3f rhs);
  ptrdiff_t (*distance)(const cfp_ptr_private_view3f first, const cfp_ptr_private_view3f last);
  cfp_ptr_private_view3f (*next)(const cfp_ptr_private_view3f p, ptrdiff_t d);
  cfp_ptr_private_view3f (*prev)(const cfp_ptr_private_view3f p, ptrdiff_t d);
  cfp_ptr_private_view3f (*inc)(const cfp_ptr_private_view3f p);
  cfp_ptr_private_view3f (*dec)(const cfp_ptr_private_view3f p);
} cfp_ptr_private_view3f_api;

typedef struct {
  /* member functions */
  float (*get)(const cfp_iter_array3f self);
  float (*get_at)(const cfp_iter_array3f self, ptrdiff_t d);
  void (*set)(cfp_iter_array3f self, float val);
  void (*set_at)(cfp_iter_array3f self, ptrdiff_t d, float val);
  cfp_ref_array3f (*ref)(cfp_iter_array3f self);
  cfp_ref_array3f (*ref_at)(cfp_iter_array3f self, ptrdiff_t d);
  cfp_ptr_array3f (*ptr)(cfp_iter_array3f self);
  cfp_ptr_array3f (*ptr_at)(cfp_iter_array3f self, ptrdiff_t d);
  size_t (*i)(const cfp_iter_array3f self);
  size_t (*j)(const cfp_iter_array3f self);
  size_t (*k)(const cfp_iter_array3f self);
  /* non-member functions */
  zfp_bool (*lt)(const cfp_iter_array3f lhs, const cfp_iter_array3f rhs);
  zfp_bool (*gt)(const cfp_iter_array3f lhs, const cfp_iter_array3f rhs);
  zfp_bool (*leq)(const cfp_iter_array3f lhs, const cfp_iter_array3f rhs);
  zfp_bool (*geq)(const cfp_iter_array3f lhs, const cfp_iter_array3f rhs);
  zfp_bool (*eq)(const cfp_iter_array3f lhs, const cfp_iter_array3f rhs);
  zfp_bool (*neq)(const cfp_iter_array3f lhs, const cfp_iter_array3f rhs);
  ptrdiff_t (*distance)(const cfp_iter_array3f first, const cfp_iter_array3f last);
  cfp_iter_array3f (*next)(const cfp_iter_array3f it, ptrdiff_t d);
  cfp_iter_array3f (*prev)(const cfp_iter_array3f it, ptrdiff_t d);
  cfp_iter_array3f (*inc)(const cfp_iter_array3f it);
  cfp_iter_array3f (*dec)(const cfp_iter_array3f it);
} cfp_iter_array3f_api;

typedef struct {
  /* member functions */
  float (*get)(const cfp_iter_view3f self);
  float (*get_at)(const cfp_iter_view3f self, ptrdiff_t d);
  void (*set)(cfp_iter_view3f self, float val);
  void (*set_at)(cfp_iter_view3f self, ptrdiff_t d, float val);
  cfp_ref_view3f (*ref)(cfp_iter_view3f self);
  cfp_ref_view3f (*ref_at)(cfp_iter_view3f self, ptrdiff_t d);
  cfp_ptr_view3f (*ptr)(cfp_iter_view3f self);
  cfp_ptr_view3f (*ptr_at)(cfp_iter_view3f self, ptrdiff_t d);
  size_t (*i)(const cfp_iter_view3f self);
  size_t (*j)(const cfp_iter_view3f self);
  size_t (*k)(const cfp_iter_view3f self);
  /* non-member functions */
  zfp_bool (*lt)(const cfp_iter_view3f lhs, const cfp_iter_view3f rhs);
  zfp_bool (*gt)(const cfp_iter_view3f lhs, const cfp_iter_view3f rhs);
  zfp_bool (*leq)(const cfp_iter_view3f lhs, const cfp_iter_view3f rhs);
  zfp_bool (*geq)(const cfp_iter_view3f lhs, const cfp_iter_view3f rhs);
  zfp_bool (*eq)(const cfp_iter_view3f lhs, const cfp_iter_view3f rhs);
  zfp_bool (*neq)(const cfp_iter_view3f lhs, const cfp_iter_view3f rhs);
  ptrdiff_t (*distance)(const cfp_iter_view3f first, const cfp_iter_view3f last);
  cfp_iter_view3f (*next)(const cfp_iter_view3f it, ptrdiff_t d);
  cfp_iter_view3f (*prev)(const cfp_iter_view3f it, ptrdiff_t d);
  cfp_iter_view3f (*inc)(const cfp_iter_view3f it);
  cfp_iter_view3f (*dec)(const cfp_iter_view3f it);
} cfp_iter_view3f_api;

typedef struct {
  /* member functions */
  float (*get)(const cfp_iter_flat_view3f self);
  float (*get_at)(const cfp_iter_flat_view3f self, ptrdiff_t d);
  void (*set)(cfp_iter_flat_view3f self, float val);
  void (*set_at)(cfp_iter_flat_view3f self, ptrdiff_t d, float val);
  cfp_ref_flat_view3f (*ref)(cfp_iter_flat_view3f self);
  cfp_ref_flat_view3f (*ref_at)(cfp_iter_flat_view3f self, ptrdiff_t d);
  cfp_ptr_flat_view3f (*ptr)(cfp_iter_flat_view3f self);
  cfp_ptr_flat_view3f (*ptr_at)(cfp_iter_flat_view3f self, ptrdiff_t d);
  size_t (*i)(const cfp_iter_flat_view3f self);
  size_t (*j)(const cfp_iter_flat_view3f self);
  size_t (*k)(const cfp_iter_flat_view3f self);
  /* non-member functions */
  zfp_bool (*lt)(const cfp_iter_flat_view3f lhs, const cfp_iter_flat_view3f rhs);
  zfp_bool (*gt)(const cfp_iter_flat_view3f lhs, const cfp_iter_flat_view3f rhs);
  zfp_bool (*leq)(const cfp_iter_flat_view3f lhs, const cfp_iter_flat_view3f rhs);
  zfp_bool (*geq)(const cfp_iter_flat_view3f lhs, const cfp_iter_flat_view3f rhs);
  zfp_bool (*eq)(const cfp_iter_flat_view3f lhs, const cfp_iter_flat_view3f rhs);
  zfp_bool (*neq)(const cfp_iter_flat_view3f lhs, const cfp_iter_flat_view3f rhs);
  ptrdiff_t (*distance)(const cfp_iter_flat_view3f first, const cfp_iter_flat_view3f last);
  cfp_iter_flat_view3f (*next)(const cfp_iter_flat_view3f it, ptrdiff_t d);
  cfp_iter_flat_view3f (*prev)(const cfp_iter_flat_view3f it, ptrdiff_t d);
  cfp_iter_flat_view3f (*inc)(const cfp_iter_flat_view3f it);
  cfp_iter_flat_view3f (*dec)(const cfp_iter_flat_view3f it);
} cfp_iter_flat_view3f_api;

typedef struct {
  /* member functions */
  float (*get)(const cfp_iter_private_view3f self);
  float (*get_at)(const cfp_iter_private_view3f self, ptrdiff_t d);
  void (*set)(cfp_iter_private_view3f self, float val);
  void (*set_at)(cfp_iter_private_view3f self, ptrdiff_t d, float val);
  cfp_ref_private_view3f (*ref)(cfp_iter_private_view3f self);
  cfp_ref_private_view3f (*ref_at)(cfp_iter_private_view3f self, ptrdiff_t d);
  cfp_ptr_private_view3f (*ptr)(cfp_iter_private_view3f self);
  cfp_ptr_private_view3f (*ptr_at)(cfp_iter_private_view3f self, ptrdiff_t d);
  size_t (*i)(const cfp_iter_private_view3f self);
  size_t (*j)(const cfp_iter_private_view3f self);
  size_t (*k)(const cfp_iter_private_view3f self);
  /* non-member functions */
  zfp_bool (*lt)(const cfp_iter_private_view3f lhs, const cfp_iter_private_view3f rhs);
  zfp_bool (*gt)(const cfp_iter_private_view3f lhs, const cfp_iter_private_view3f rhs);
  zfp_bool (*leq)(const cfp_iter_private_view3f lhs, const cfp_iter_private_view3f rhs);
  zfp_bool (*geq)(const cfp_iter_private_view3f lhs, const cfp_iter_private_view3f rhs);
  zfp_bool (*eq)(const cfp_iter_private_view3f lhs, const cfp_iter_private_view3f rhs);
  zfp_bool (*neq)(const cfp_iter_private_view3f lhs, const cfp_iter_private_view3f rhs);
  ptrdiff_t (*distance)(const cfp_iter_private_view3f first, const cfp_iter_private_view3f last);
  cfp_iter_private_view3f (*next)(const cfp_iter_private_view3f it, ptrdiff_t d);
  cfp_iter_private_view3f (*prev)(const cfp_iter_private_view3f it, ptrdiff_t d);
  cfp_iter_private_view3f (*inc)(const cfp_iter_private_view3f it);
  cfp_iter_private_view3f (*dec)(const cfp_iter_private_view3f it);
} cfp_iter_private_view3f_api;

typedef struct {
  /* constructor/destructor */
  cfp_view3f (*ctor)(const cfp_array3f a);
  cfp_view3f (*ctor_subset)(cfp_array3f a, size_t x, size_t y, size_t z, size_t nx, size_t ny, size_t nz);
  void (*dtor)(cfp_view3f self);
  /* member functions */
  size_t (*global_x)(cfp_view3f self, size_t i);
  size_t (*global_y)(cfp_view3f self, size_t j);
  size_t (*global_z)(cfp_view3f self, size_t k);
  size_t (*size_x)(cfp_view3f self);
  size_t (*size_y)(cfp_view3f self);
  size_t (*size_z)(cfp_view3f self);
  float (*get)(const cfp_view3f self, size_t i, size_t j, size_t k);
  double (*rate)(const cfp_view3f self);
  size_t (*size)(cfp_view3f self);

  cfp_ref_view3f (*ref)(cfp_view3f self, size_t i, size_t j, size_t k);
  cfp_iter_view3f (*begin)(cfp_view3f self);
  cfp_iter_view3f (*end)(cfp_view3f self);
} cfp_view3f_api;

typedef struct {
  /* constructor/destructor */
  cfp_flat_view3f (*ctor)(const cfp_array3f a);
  cfp_flat_view3f (*ctor_subset)(cfp_array3f a, size_t x, size_t y, size_t z, size_t nx, size_t ny, size_t nz);
  void (*dtor)(cfp_flat_view3f self);
  /* member functions */
  size_t (*global_x)(cfp_flat_view3f self, size_t i);
  size_t (*global_y)(cfp_flat_view3f self, size_t j);
  size_t (*global_z)(cfp_flat_view3f self, size_t k);
  size_t (*size_x)(cfp_flat_view3f self);
  size_t (*size_y)(cfp_flat_view3f self);
  size_t (*size_z)(cfp_flat_view3f self);
  float (*get)(const cfp_flat_view3f self, size_t i, size_t j, size_t k);
  float (*get_flat)(const cfp_flat_view3f self, size_t i);
  void (*set_flat)(const cfp_flat_view3f self, size_t i, float val);
  double (*rate)(const cfp_flat_view3f self);
  size_t (*size)(cfp_flat_view3f self);

  cfp_ref_flat_view3f (*ref)(cfp_flat_view3f self, size_t i, size_t j, size_t k);
  cfp_iter_flat_view3f (*begin)(cfp_flat_view3f self);
  cfp_iter_flat_view3f (*end)(cfp_flat_view3f self);

  size_t (*index)(cfp_flat_view3f self, size_t i, size_t j, size_t k);
  void (*ijk)(cfp_flat_view3f self, size_t* i, size_t* j, size_t* k, size_t index);
} cfp_flat_view3f_api;

typedef struct {
  /* constructor/destructor */
  cfp_private_view3f (*ctor)(const cfp_array3f a);
  cfp_private_view3f (*ctor_subset)(cfp_array3f a, size_t x, size_t y, size_t z, size_t nx, size_t ny, size_t nz);
  void (*dtor)(cfp_private_view3f self);
  /* member functions */
  size_t (*global_x)(cfp_private_view3f self, size_t i);
  size_t (*global_y)(cfp_private_view3f self, size_t j);
  size_t (*global_z)(cfp_private_view3f self, size_t k);
  size_t (*size_x)(cfp_private_view3f self);
  size_t (*size_y)(cfp_private_view3f self);
  size_t (*size_z)(cfp_private_view3f self);
  float (*get)(const cfp_private_view3f self, size_t i, size_t j, size_t k);
  double (*rate)(const cfp_private_view3f self);
  size_t (*size)(cfp_private_view3f self);

  cfp_ref_private_view3f (*ref)(cfp_private_view3f self, size_t i, size_t j, size_t k);
  cfp_iter_private_view3f (*begin)(cfp_private_view3f self);
  cfp_iter_private_view3f (*end)(cfp_private_view3f self);

  void (*partition)(cfp_private_view3f self, size_t index, size_t count);
  void (*flush_cache)(cfp_private_view3f self);
} cfp_private_view3f_api;

typedef struct {
  /* constructor/destructor */
  cfp_header (*ctor)(const cfp_array3f a);
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
} cfp_header3f_api;

typedef struct {
  cfp_array3f (*ctor_default)();
  cfp_array3f (*ctor)(size_t nx, size_t ny, size_t nz, double rate, const float* p, size_t cache_size);
  cfp_array3f (*ctor_copy)(const cfp_array3f src);
  cfp_array3f (*ctor_header)(const cfp_header h, const void* buffer, size_t buffer_size_bytes);
  void (*dtor)(cfp_array3f self);

  void (*deep_copy)(cfp_array3f self, const cfp_array3f src);

  double (*rate)(const cfp_array3f self);
  double (*set_rate)(cfp_array3f self, double rate);
  size_t (*cache_size)(const cfp_array3f self);
  void (*set_cache_size)(cfp_array3f self, size_t bytes);
  void (*clear_cache)(const cfp_array3f self);
  void (*flush_cache)(const cfp_array3f self);
  size_t (*compressed_size)(const cfp_array3f self);
  void* (*compressed_data)(const cfp_array3f self);
  size_t (*size)(const cfp_array3f self);
  size_t (*size_x)(const cfp_array3f self);
  size_t (*size_y)(const cfp_array3f self);
  size_t (*size_z)(const cfp_array3f self);
  void (*resize)(cfp_array3f self, size_t nx, size_t ny, size_t nz, zfp_bool clear);

  void (*get_array)(const cfp_array3f self, float* p);
  void (*set_array)(cfp_array3f self, const float* p);
  float (*get_flat)(const cfp_array3f self, size_t i);
  void (*set_flat)(cfp_array3f self, size_t i, float val);
  float (*get)(const cfp_array3f self, size_t i, size_t j, size_t k);
  void (*set)(cfp_array3f self, size_t i, size_t j, size_t k, float val);

  cfp_ref_array3f (*ref)(cfp_array3f self, size_t i, size_t j, size_t k);
  cfp_ref_array3f (*ref_flat)(cfp_array3f self, size_t i);

  cfp_ptr_array3f (*ptr)(cfp_array3f self, size_t i, size_t j, size_t k);
  cfp_ptr_array3f (*ptr_flat)(cfp_array3f self, size_t i);

  cfp_iter_array3f (*begin)(cfp_array3f self);
  cfp_iter_array3f (*end)(cfp_array3f self);

  cfp_ref_array3f_api reference;
  cfp_ptr_array3f_api pointer;
  cfp_iter_array3f_api iterator;

  cfp_view3f_api view;
  cfp_ref_view3f_api view_reference;
  cfp_ptr_view3f_api view_pointer;
  cfp_iter_view3f_api view_iterator;

  cfp_flat_view3f_api flat_view;
  cfp_ref_flat_view3f_api flat_view_reference;
  cfp_ptr_flat_view3f_api flat_view_pointer;
  cfp_iter_flat_view3f_api flat_view_iterator;

  cfp_private_view3f_api private_view;
  cfp_ref_private_view3f_api private_view_reference;
  cfp_ptr_private_view3f_api private_view_pointer;
  cfp_iter_private_view3f_api private_view_iterator;

  cfp_header3f_api header;
} cfp_array3f_api;

#endif
