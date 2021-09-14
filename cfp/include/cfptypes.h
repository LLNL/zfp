#ifndef CFP_TYPES_H
#define CFP_TYPES_H

#define CFP_DECL_IDX1 size_t x;
#define CFP_DECL_IDX2 size_t x; size_t y;
#define CFP_DECL_IDX3 size_t x; size_t y; size_t z;
#define CFP_DECL_IDX4 size_t x; size_t y; size_t z; size_t w;

#define CFP_DECL_CONTAINER(NAME, DIM, SCALAR) \
typedef struct {\
  void* object;\
} cfp_ ## NAME ## DIM ## SCALAR;

#define CFP_DECL_ACCESSOR(NAME, DIM, SCALAR) \
typedef struct {\
  void* container;\
  CFP_DECL_IDX ## DIM\
} cfp_ ## NAME ## DIM ## SCALAR;

#endif
