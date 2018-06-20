#ifndef CUZFP_ZFP_STRUCTS
#define CUZFP_ZFP_STRUCTS
#include <algorithm>
#include <climits>
//
// These structures are being define so that in the future
// we can integrate with the main zfp code more easily
//

#define ZFP_MAX_PREC    64 /* maximum precision supported */
#define ZFP_MIN_EXP  -1074 /* minimum floating-point base-2 exponent */
#define ZFP_MAX_BITS  4171 /* maximum number of bits per block */
#define ZFP_MIN_BITS     0 /* minimum number of bits per block */
#define ZFP_HEADER_MAX_BITS 148 /* max number of header bits */

typedef unsigned long long Word;
#define wsize ((uint)(CHAR_BIT * sizeof(Word)))

namespace cuZFP
{

typedef struct
{
  uint minbits;
  uint maxbits;
  uint maxprec;
  int  minexp;
  Word *stream; // TODO: placeholder for actual zfp_stream
} zfp_stream;

typedef enum {
  zfp_type_none   = 0, // unspecified type
  zfp_type_int32  = 1, // 32-bit signed integer
  zfp_type_int64  = 2, // 64-bit signed integer
  zfp_type_float  = 3, // single precision floating point
  zfp_type_double = 4  // double precision floating point
} zfp_type;

typedef struct {
  zfp_type type;   // scalar type (e.g. int32, double)
  uint nx, ny, nz; // sizes (zero for unused dimensions)
  int sx, sy, sz;  // strides (zero for contiguous array a[nz][ny][nx])
  void* data;      // pointer to array data
} zfp_field;

static double
stream_set_rate(zfp_stream* zfp, double rate, zfp_type type, uint dims)
{
  uint n = 1u << (2 * dims);
  uint bits = (uint)std::floor(n * rate + 0.5);
  switch (type) {
    case zfp_type_float:
      bits = std::max(bits, 1 + 8u);
      break;
    case zfp_type_double:
      bits = std::max(bits, 1 + 11u);
      break;
    default:
      break;
  }
 
  // 3d currently expects word aligned rates.
  // so we are forcing this
  if (dims == 3) 
  {
    /* for write random access, round up to next multiple of stream word size */
    bits += (uint) wsize - 1;
    bits &= ~(wsize - 1);
  }

  zfp->minbits = bits;
  zfp->maxbits = bits;
  zfp->maxprec = ZFP_MAX_PREC;
  zfp->minexp = ZFP_MIN_EXP;
  return (double)bits / n;
} 

static zfp_field*
zfp_field_alloc()
{
  zfp_field* field = (zfp_field*)malloc(sizeof(zfp_field));
  if (field) {
    field->type = zfp_type_none;
    field->nx = field->ny = field->nz = 0;
    field->sx = field->sy = field->sz = 0;
    field->data = 0;
  }
  return field;
}

static zfp_field*
zfp_field_1d(void* data, zfp_type type, uint nx)
{
  zfp_field* field = zfp_field_alloc();
  if (field) {
    field->type = type;
    field->nx = nx;
    field->data = data;
  }
  return field;
}

static zfp_field*
zfp_field_2d(void* data, zfp_type type, uint nx, uint ny)
{
  zfp_field* field = zfp_field_alloc();
  if (field) {
    field->type = type;
    field->nx = nx;
    field->ny = ny;
    field->data = data;
  }
  return field;
}

static zfp_field*
zfp_field_3d(void* data, zfp_type type, uint nx, uint ny, uint nz)
{
  zfp_field* field = zfp_field_alloc();
  if (field) {
    field->type = type;
    field->nx = nx;
    field->ny = ny;
    field->nz = nz;
    field->data = data;
  }
  return field;
}

static void
zfp_field_free(zfp_field* field)
{
  free(field);
}


static zfp_stream*
zfp_stream_open(Word* stream)
{
  zfp_stream* zfp = (zfp_stream*)malloc(sizeof(zfp_stream));
  if (zfp) {
    zfp->stream = stream;
    zfp->minbits = ZFP_MIN_BITS;
    zfp->maxbits = ZFP_MAX_BITS;
    zfp->maxprec = ZFP_MAX_PREC;
    zfp->minexp = ZFP_MIN_EXP;
  }
  return zfp;
}


static uint
zfp_field_dimensionality(const zfp_field* field)
{
  return field->nx ? field->ny ? field->nz ? 3 : 2 : 1 : 0;
}

static uint
type_precision(zfp_type type)
{
  switch (type) {
    case zfp_type_int32:
      return CHAR_BIT * (uint)sizeof(int);
    case zfp_type_int64:
      return CHAR_BIT * (uint)sizeof(long long int);
    case zfp_type_float:
      return CHAR_BIT * (uint)sizeof(float);
    case zfp_type_double:
      return CHAR_BIT * (uint)sizeof(double);
    default:
      return 0;
  }
}

template<typename T> static
zfp_type get_zfp_type()
{
  return zfp_type_none;
}

template<>
zfp_type get_zfp_type<int>()
{
  return zfp_type_int32;
}

template<>
zfp_type get_zfp_type<long long int>()
{
  return zfp_type_int64;
}

template<>
zfp_type get_zfp_type<float>()
{
  return zfp_type_float;
}

template<>
zfp_type get_zfp_type<double>()
{
  return zfp_type_double;
}

static size_t
zfp_type_size(zfp_type type)
{
  switch (type) {
    case zfp_type_int32:
      return sizeof(int);
    case zfp_type_int64:
      return sizeof(long long int);
    case zfp_type_float:
      return sizeof(float);
    case zfp_type_double:
      return sizeof(double);
    default:
      return 0;
  }
}

static size_t
zfp_stream_maximum_size(const zfp_stream* zfp, const zfp_field* field)
{
  uint dims = zfp_field_dimensionality(field);
  uint mx = (std::max(field->nx, 1u) + 3) / 4;
  uint my = (std::max(field->ny, 1u) + 3) / 4;
  uint mz = (std::max(field->nz, 1u) + 3) / 4;
  size_t blocks = (size_t)mx * (size_t)my * (size_t)mz;
  uint values = 1u << (2 * dims);
  uint maxbits = 1;

  if (!dims)
    return 0;
  switch (field->type) {
    case zfp_type_none:
      return 0;
    case zfp_type_float:
      maxbits += 8;
      break;
    case zfp_type_double:
      maxbits += 11;
      break;
    default:
      break;
  }
  maxbits += values - 1 + values * std::min(zfp->maxprec, type_precision(field->type));
  maxbits = std::min(maxbits, zfp->maxbits);
  maxbits = std::max(maxbits, zfp->minbits);
  return ((ZFP_HEADER_MAX_BITS + blocks * maxbits + wsize - 1) & ~(wsize - 1)) / CHAR_BIT;
}

static void
zfp_stream_close(zfp_stream* zfp)
{
  free(zfp);
}


}; // namespace
#endif
