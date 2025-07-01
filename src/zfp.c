#include <limits.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include "zfp.h"
#include "zfp/internal/zfp/macros.h"
#include "zfp/version.h"
#include "template/template.h"

/* public data ------------------------------------------------------------- */

const uint zfp_codec_version = ZFP_CODEC;
const uint zfp_library_version = ZFP_VERSION;
const char* const zfp_version_string = "zfp version " ZFP_VERSION_STRING " (December 15, 2023)";

/* private functions ------------------------------------------------------- */

static size_t
field_index_span(const zfp_field* field, ptrdiff_t* min, ptrdiff_t* max)
{
  /* compute strides */
  ptrdiff_t sx = field->sx ? field->sx : 1;
  ptrdiff_t sy = field->sy ? field->sy : (ptrdiff_t)field->nx;
  ptrdiff_t sz = field->sz ? field->sz : (ptrdiff_t)(field->nx * field->ny);
  ptrdiff_t sw = field->sw ? field->sw : (ptrdiff_t)(field->nx * field->ny * field->nz);
  /* compute largest offsets from base pointer */
  ptrdiff_t dx = field->nx ? sx * (ptrdiff_t)(field->nx - 1) : 0;
  ptrdiff_t dy = field->ny ? sy * (ptrdiff_t)(field->ny - 1) : 0;
  ptrdiff_t dz = field->nz ? sz * (ptrdiff_t)(field->nz - 1) : 0;
  ptrdiff_t dw = field->nw ? sw * (ptrdiff_t)(field->nw - 1) : 0;
  /* compute lowest and highest offset */
  ptrdiff_t imin = MIN(dx, 0) + MIN(dy, 0) + MIN(dz, 0) + MIN(dw, 0);
  ptrdiff_t imax = MAX(dx, 0) + MAX(dy, 0) + MAX(dz, 0) + MAX(dw, 0);
  if (min)
    *min = imin;
  if (max)
    *max = imax;
  return (size_t)(imax - imin + 1);
}

static zfp_bool
is_reversible(const zfp_stream* zfp)
{
  return zfp->minexp < ZFP_MIN_EXP;
}

/* shared code across template instances ------------------------------------*/

#include "share/parallel.c"
#include "share/omp.c"

/* template instantiation of utility functions ------------------------------*/

#include "template/utils.c"

/* template instantiation of integer and float compressor -------------------*/

#define Scalar int32
#include "template/compress.c"
#include "template/decompress.c"
#include "template/ompcompress.c"
#include "template/cudacompress.c"
#include "template/cudadecompress.c"
#undef Scalar

#define Scalar int64
#include "template/compress.c"
#include "template/decompress.c"
#include "template/ompcompress.c"
#include "template/cudacompress.c"
#include "template/cudadecompress.c"
#undef Scalar

#define Scalar float
#include "template/compress.c"
#include "template/decompress.c"
#include "template/ompcompress.c"
#include "template/cudacompress.c"
#include "template/cudadecompress.c"
#undef Scalar

#define Scalar double
#include "template/compress.c"
#include "template/decompress.c"
#include "template/ompcompress.c"
#include "template/cudacompress.c"
#include "template/cudadecompress.c"
#undef Scalar

/* public functions: miscellaneous ----------------------------------------- */

size_t
zfp_type_size(zfp_type type)
{
  switch (type) {
    case zfp_type_int32:
      return sizeof(int32);
    case zfp_type_int64:
      return sizeof(int64);
    case zfp_type_float:
      return sizeof(float);
    case zfp_type_double:
      return sizeof(double);
    default:
      return 0;
  }
}

size_t
zfp_block_maximum_size(zfp_type type, uint dims, zfp_bool reversible)
{
  static const size_t size_table[2][4][4] = {
    /* non-reversible mode */
    {
      { 131,  527, 2111,  8447 }, /* int32 */
      { 259, 1039, 4159, 16639 }, /* int64 */
      { 140,  536, 2120,  8456 }, /* float */
      { 271, 1051, 4171, 16651 }, /* double */
    },
    /* reversible mode */
    {
      { 136,  532, 2116,  8452 }, /* int32 */
      { 265, 1045, 4165, 16645 }, /* int64 */
      { 146,  542, 2126,  8462 }, /* float */
      { 278, 1058, 4178, 16658 }, /* double */
    },
  };

  /* check arguments */
  if (!(zfp_type_int32 <= type && type <= zfp_type_double) ||
      !(1 <= dims && dims <= 4))
    return 0;

  return size_table[reversible ? 1 : 0][type - 1][dims - 1];
}

/* public functions: fields ------------------------------------------------ */

zfp_field*
zfp_field_alloc(void)
{
  zfp_field* field = malloc(sizeof(zfp_field));
  if (field) {
    field->type = zfp_type_none;
    field->nx = field->ny = field->nz = field->nw = 0;
    field->sx = field->sy = field->sz = field->sw = 0;
    field->data = 0;
  }
  return field;
}

zfp_field*
zfp_field_1d(void* data, zfp_type type, size_t nx)
{
  zfp_field* field = zfp_field_alloc();
  if (field) {
    field->type = type;
    field->nx = nx;
    field->data = data;
  }
  return field;
}

zfp_field*
zfp_field_2d(void* data, zfp_type type, size_t nx, size_t ny)
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

zfp_field*
zfp_field_3d(void* data, zfp_type type, size_t nx, size_t ny, size_t nz)
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

zfp_field*
zfp_field_4d(void* data, zfp_type type, size_t nx, size_t ny, size_t nz, size_t nw)
{
  zfp_field* field = zfp_field_alloc();
  if (field) {
    field->type = type;
    field->nx = nx;
    field->ny = ny;
    field->nz = nz;
    field->nw = nw;
    field->data = data;
  }
  return field;
}

void
zfp_field_free(zfp_field* field)
{
  free(field);
}

void*
zfp_field_pointer(const zfp_field* field)
{
  return field->data;
}

void*
zfp_field_begin(const zfp_field* field)
{
  if (field->data) {
    ptrdiff_t min;
    field_index_span(field, &min, NULL);
    return (uchar*)field->data + min * (ptrdiff_t)zfp_type_size(field->type);
  }
  else
    return NULL;
}

zfp_type
zfp_field_type(const zfp_field* field)
{
  return field->type;
}

uint
zfp_field_precision(const zfp_field* field)
{
  return (uint)(CHAR_BIT * zfp_type_size(field->type));
}

uint
zfp_field_dimensionality(const zfp_field* field)
{
  return field->nx ? field->ny ? field->nz ? field->nw ? 4 : 3 : 2 : 1 : 0;
}

size_t
zfp_field_size(const zfp_field* field, size_t* size)
{
  if (size)
    switch (zfp_field_dimensionality(field)) {
      case 4:
        size[3] = field->nw;
        fallthrough_
      case 3:
        size[2] = field->nz;
        fallthrough_
      case 2:
        size[1] = field->ny;
        fallthrough_
      case 1:
        size[0] = field->nx;
        break;
    }
  return MAX(field->nx, 1u) * MAX(field->ny, 1u) * MAX(field->nz, 1u) * MAX(field->nw, 1u);
}

size_t
zfp_field_size_bytes(const zfp_field* field)
{
  return field_index_span(field, NULL, NULL) * zfp_type_size(field->type);
}

size_t
zfp_field_blocks(const zfp_field* field)
{
  size_t bx = (field->nx + 3) / 4;
  size_t by = (field->ny + 3) / 4;
  size_t bz = (field->nz + 3) / 4;
  size_t bw = (field->nw + 3) / 4;
  switch (zfp_field_dimensionality(field)) {
    case 1: return bx;
    case 2: return bx * by;
    case 3: return bx * by * bz;
    case 4: return bx * by * bz * bw;
    default: return 0;
  }
}

zfp_bool
zfp_field_stride(const zfp_field* field, ptrdiff_t* stride)
{
  if (stride)
    switch (zfp_field_dimensionality(field)) {
      case 4:
        stride[3] = field->sw ? field->sw : (ptrdiff_t)(field->nx * field->ny * field->nz);
        fallthrough_
      case 3:
        stride[2] = field->sz ? field->sz : (ptrdiff_t)(field->nx * field->ny);
        fallthrough_
      case 2:
        stride[1] = field->sy ? field->sy : (ptrdiff_t)field->nx;
        fallthrough_
      case 1:
        stride[0] = field->sx ? field->sx : 1;
        break;
    }
  return field->sx || field->sy || field->sz || field->sw;
}

zfp_bool
zfp_field_is_contiguous(const zfp_field* field)
{
  return field_index_span(field, NULL, NULL) == zfp_field_size(field, NULL);
}

uint64
zfp_field_metadata(const zfp_field* field)
{
  uint64 meta = 0;
  /* 48 bits for dimensions */
  switch (zfp_field_dimensionality(field)) {
    case 1:
      if ((uint64)(field->nx - 1) >> 48)
        return ZFP_META_NULL;
      meta <<= 48; meta += field->nx - 1;
      break;
    case 2:
      if (((field->nx - 1) >> 24) ||
          ((field->ny - 1) >> 24))
        return ZFP_META_NULL;
      meta <<= 24; meta += field->ny - 1;
      meta <<= 24; meta += field->nx - 1;
      break;
    case 3:
      if (((field->nx - 1) >> 16) ||
          ((field->ny - 1) >> 16) ||
          ((field->nz - 1) >> 16))
        return ZFP_META_NULL;
      meta <<= 16; meta += field->nz - 1;
      meta <<= 16; meta += field->ny - 1;
      meta <<= 16; meta += field->nx - 1;
      break;
    case 4:
      if (((field->nx - 1) >> 12) ||
          ((field->ny - 1) >> 12) ||
          ((field->nz - 1) >> 12) ||
          ((field->nw - 1) >> 12))
        return ZFP_META_NULL;
      meta <<= 12; meta += field->nw - 1;
      meta <<= 12; meta += field->nz - 1;
      meta <<= 12; meta += field->ny - 1;
      meta <<= 12; meta += field->nx - 1;
      break;
  }
  /* 2 bits for dimensionality (1D, 2D, 3D, 4D) */
  meta <<= 2; meta += zfp_field_dimensionality(field) - 1;
  /* 2 bits for scalar type */
  meta <<= 2; meta += field->type - 1;
  return meta;
}

void
zfp_field_set_pointer(zfp_field* field, void* data)
{
  field->data = data;
}

zfp_type
zfp_field_set_type(zfp_field* field, zfp_type type)
{
  switch (type) {
    case zfp_type_int32:
    case zfp_type_int64:
    case zfp_type_float:
    case zfp_type_double:
      field->type = type;
      return type;
    default:
      return zfp_type_none;
  }
}

void
zfp_field_set_size_1d(zfp_field* field, size_t n)
{
  field->nx = n;
  field->ny = 0;
  field->nz = 0;
  field->nw = 0;
}

void
zfp_field_set_size_2d(zfp_field* field, size_t nx, size_t ny)
{
  field->nx = nx;
  field->ny = ny;
  field->nz = 0;
  field->nw = 0;
}

void
zfp_field_set_size_3d(zfp_field* field, size_t nx, size_t ny, size_t nz)
{
  field->nx = nx;
  field->ny = ny;
  field->nz = nz;
  field->nw = 0;
}

void
zfp_field_set_size_4d(zfp_field* field, size_t nx, size_t ny, size_t nz, size_t nw)
{
  field->nx = nx;
  field->ny = ny;
  field->nz = nz;
  field->nw = nw;
}

void
zfp_field_set_stride_1d(zfp_field* field, ptrdiff_t sx)
{
  field->sx = sx;
  field->sy = 0;
  field->sz = 0;
  field->sw = 0;
}

void
zfp_field_set_stride_2d(zfp_field* field, ptrdiff_t sx, ptrdiff_t sy)
{
  field->sx = sx;
  field->sy = sy;
  field->sz = 0;
  field->sw = 0;
}

void
zfp_field_set_stride_3d(zfp_field* field, ptrdiff_t sx, ptrdiff_t sy, ptrdiff_t sz)
{
  field->sx = sx;
  field->sy = sy;
  field->sz = sz;
  field->sw = 0;
}

void
zfp_field_set_stride_4d(zfp_field* field, ptrdiff_t sx, ptrdiff_t sy, ptrdiff_t sz, ptrdiff_t sw)
{
  field->sx = sx;
  field->sy = sy;
  field->sz = sz;
  field->sw = sw;
}

zfp_bool
zfp_field_set_metadata(zfp_field* field, uint64 meta)
{
  uint64 dims;
  /* ensure value is in range */
  if (meta >> ZFP_META_BITS)
    return zfp_false;
  field->type = (zfp_type)((meta & 0x3u) + 1); meta >>= 2;
  dims = (meta & 0x3u) + 1; meta >>= 2;
  switch (dims) {
    case 1:
      /* currently dimensions are limited to 2^32 - 1 */
      field->nx = (size_t)(meta & UINT64C(0x0000ffffffff)) + 1; meta >>= 48;
      field->ny = 0;
      field->nz = 0;
      field->nw = 0;
      break;
    case 2:
      field->nx = (size_t)(meta & UINT64C(0xffffff)) + 1; meta >>= 24;
      field->ny = (size_t)(meta & UINT64C(0xffffff)) + 1; meta >>= 24;
      field->nz = 0;
      field->nw = 0;
      break;
    case 3:
      field->nx = (size_t)(meta & UINT64C(0xffff)) + 1; meta >>= 16;
      field->ny = (size_t)(meta & UINT64C(0xffff)) + 1; meta >>= 16;
      field->nz = (size_t)(meta & UINT64C(0xffff)) + 1; meta >>= 16;
      field->nw = 0;
      break;
    case 4:
      field->nx = (size_t)(meta & UINT64C(0xfff)) + 1; meta >>= 12;
      field->ny = (size_t)(meta & UINT64C(0xfff)) + 1; meta >>= 12;
      field->nz = (size_t)(meta & UINT64C(0xfff)) + 1; meta >>= 12;
      field->nw = (size_t)(meta & UINT64C(0xfff)) + 1; meta >>= 12;
      break;
  }
  field->sx = field->sy = field->sz = field->sw = 0;
  return zfp_true;
}

/* public functions: compression mode and parameter settings --------------- */

zfp_config
zfp_config_none(void)
{
  zfp_config config;
  config.mode = zfp_mode_null;
  return config;
}

zfp_config
zfp_config_rate(
  double rate,
  zfp_bool align
)
{
  zfp_config config;
  config.mode = zfp_mode_fixed_rate;
  config.arg.rate = align ? -rate : +rate;
  return config;
}

zfp_config
zfp_config_precision(
  uint precision
)
{
  zfp_config config;
  config.mode = zfp_mode_fixed_precision;
  config.arg.precision = precision;
  return config;
}

zfp_config
zfp_config_accuracy(
  double tolerance
)
{
  zfp_config config;
  config.mode = zfp_mode_fixed_accuracy;
  config.arg.tolerance = tolerance;
  return config;
}

zfp_config
zfp_config_reversible(void)
{
  zfp_config config;
  config.mode = zfp_mode_reversible;
  return config;
}

zfp_config
zfp_config_expert(    
  uint minbits,
  uint maxbits,
  uint maxprec,
  int minexp
)
{
  zfp_config config;
  config.mode = zfp_mode_expert;
  config.arg.expert.minbits = minbits;
  config.arg.expert.maxbits = maxbits;
  config.arg.expert.maxprec = maxprec;
  config.arg.expert.minexp = minexp;
  return config;
}

/* public functions: zfp compressed stream --------------------------------- */

zfp_stream*
zfp_stream_open(bitstream* stream)
{
  zfp_stream* zfp = malloc(sizeof(zfp_stream));
  if (zfp) {
    zfp->stream = stream;
    zfp->minbits = ZFP_MIN_BITS;
    zfp->maxbits = ZFP_MAX_BITS;
    zfp->maxprec = ZFP_MAX_PREC;
    zfp->minexp = ZFP_MIN_EXP;
    zfp->exec.policy = zfp_exec_serial;
    zfp->exec.params = NULL;
  }
  return zfp;
}

void
zfp_stream_close(zfp_stream* zfp)
{
  if (zfp->exec.params != NULL)
    free(zfp->exec.params);
  free(zfp);
}

bitstream*
zfp_stream_bit_stream(const zfp_stream* zfp)
{
  return zfp->stream;
}

zfp_mode
zfp_stream_compression_mode(const zfp_stream* zfp)
{
  if (zfp->minbits > zfp->maxbits || !(0 < zfp->maxprec && zfp->maxprec <= 64))
    return zfp_mode_null;

  /* default values are considered expert mode */
  if (zfp->minbits == ZFP_MIN_BITS &&
      zfp->maxbits == ZFP_MAX_BITS &&
      zfp->maxprec == ZFP_MAX_PREC &&
      zfp->minexp == ZFP_MIN_EXP)
    return zfp_mode_expert;

  /* fixed rate? */
  if (zfp->minbits == zfp->maxbits &&
      1 <= zfp->maxbits && zfp->maxbits <= ZFP_MAX_BITS &&
      zfp->maxprec >= ZFP_MAX_PREC &&
      zfp->minexp == ZFP_MIN_EXP)
    return zfp_mode_fixed_rate;

  /* fixed precision? */
  if (zfp->minbits <= ZFP_MIN_BITS &&
      zfp->maxbits >= ZFP_MAX_BITS &&
      zfp->maxprec >= 1 &&
      zfp->minexp == ZFP_MIN_EXP)
    return zfp_mode_fixed_precision;

  /* fixed accuracy? */
  if (zfp->minbits <= ZFP_MIN_BITS &&
      zfp->maxbits >= ZFP_MAX_BITS &&
      zfp->maxprec >= ZFP_MAX_PREC &&
      zfp->minexp >= ZFP_MIN_EXP)
    return zfp_mode_fixed_accuracy;

  /* reversible? */
  if (zfp->minbits <= ZFP_MIN_BITS &&
      zfp->maxbits >= ZFP_MAX_BITS &&
      zfp->maxprec >= ZFP_MAX_PREC &&
      zfp->minexp < ZFP_MIN_EXP)
    return zfp_mode_reversible;

  return zfp_mode_expert;
}

double
zfp_stream_rate(const zfp_stream* zfp, uint dims)
{
  return (zfp_stream_compression_mode(zfp) == zfp_mode_fixed_rate)
           ? (double)zfp->maxbits / (1u << (2 * dims))
           : 0.0;
}

uint
zfp_stream_precision(const zfp_stream* zfp)
{
  return (zfp_stream_compression_mode(zfp) == zfp_mode_fixed_precision)
           ? zfp->maxprec
           : 0;
}

double
zfp_stream_accuracy(const zfp_stream* zfp)
{
  return (zfp_stream_compression_mode(zfp) == zfp_mode_fixed_accuracy)
           ? ldexp(1.0, zfp->minexp)
           : 0.0;
}

uint64
zfp_stream_mode(const zfp_stream* zfp)
{
  uint64 mode = 0;
  uint minbits;
  uint maxbits;
  uint maxprec;
  uint minexp;

  /* common configurations mapped to short representation */
  switch (zfp_stream_compression_mode(zfp)) {
    case zfp_mode_fixed_rate:
      if (zfp->maxbits <= 2048)
        /* maxbits is [1, 2048] */
        /* returns [0, 2047] */
        return (zfp->maxbits - 1);
      else
        break;

    case zfp_mode_fixed_precision:
      if (zfp->maxprec <= 128)
        /* maxprec is [1, 128] */
        /* returns [2048, 2175] */
        return (zfp->maxprec - 1) + (2048);
      else
        break;

    case zfp_mode_fixed_accuracy:
      if (zfp->minexp <= 843)
        /* minexp is [ZFP_MIN_EXP=-1074, 843] */
        /* returns [2177, ZFP_MODE_SHORT_MAX=4094] */
        /* +1 because skipped 2176 */
        return (uint64)(zfp->minexp - ZFP_MIN_EXP) + (2048 + 128 + 1);
      else
        break;

    case zfp_mode_reversible:
      /* returns 2176 */
      return 2048 + 128;

    default:
      break;
  }

  /* encode each parameter separately */
  minbits = MAX(1, MIN(zfp->minbits, 0x8000u)) - 1;
  maxbits = MAX(1, MIN(zfp->maxbits, 0x8000u)) - 1;
  maxprec = MAX(1, MIN(zfp->maxprec, 0x0080u)) - 1;
  minexp = (uint)MAX(0, MIN(zfp->minexp + 16495, 0x7fff));
  mode <<= 15; mode += minexp;
  mode <<=  7; mode += maxprec;
  mode <<= 15; mode += maxbits;
  mode <<= 15; mode += minbits;
  mode <<= 12; mode += 0xfffu;

  return mode;
}

void
zfp_stream_params(const zfp_stream* zfp, uint* minbits, uint* maxbits, uint* maxprec, int* minexp)
{
  if (minbits)
    *minbits = zfp->minbits;
  if (maxbits)
    *maxbits = zfp->maxbits;
  if (maxprec)
    *maxprec = zfp->maxprec;
  if (minexp)
    *minexp = zfp->minexp;
}

size_t
zfp_stream_compressed_size(const zfp_stream* zfp)
{
  return stream_size(zfp->stream);
}

size_t
zfp_stream_maximum_size(const zfp_stream* zfp, const zfp_field* field)
{
  zfp_bool reversible = is_reversible(zfp);
  uint dims = zfp_field_dimensionality(field);
  size_t blocks = zfp_field_blocks(field);
  uint values = 1u << (2 * dims);
  uint maxbits = 0;
  bitstream_size maxsize;

  if (!dims)
    return 0;
  switch (field->type) {
    case zfp_type_int32:
      maxbits += reversible ? 5 : 0;
      break;
    case zfp_type_int64:
      maxbits += reversible ? 6 : 0;
      break;
    case zfp_type_float:
      maxbits += reversible ? 1 + 1 + 8 + 5 : 1 + 8;
      break;
    case zfp_type_double:
      maxbits += reversible ? 1 + 1 + 11 + 6 : 1 + 11;
      break;
    default:
      return 0;
  }
  maxbits += values - 1 + values * MIN(zfp->maxprec, zfp_field_precision(field));
  maxbits = MIN(maxbits, zfp->maxbits);
  maxbits = MAX(maxbits, zfp->minbits);
  
  /* compute number of bytes in multiples of words */
  maxsize = ZFP_HEADER_MAX_BITS + (bitstream_size)blocks * maxbits;
  maxsize = _t1(round_up, bitstream_size)(maxsize, stream_word_bits);
  maxsize /= CHAR_BIT;

  /* ensure maxsize fits in size_t to avoid silent truncation */
  if ((size_t)maxsize != maxsize)
    return 0;

  return (size_t)maxsize;
}

void
zfp_stream_set_bit_stream(zfp_stream* zfp, bitstream* stream)
{
  zfp->stream = stream;
}

void
zfp_stream_set_reversible(zfp_stream* zfp)
{
  zfp->minbits = ZFP_MIN_BITS;
  zfp->maxbits = ZFP_MAX_BITS;
  zfp->maxprec = ZFP_MAX_PREC;
  zfp->minexp = ZFP_MIN_EXP - 1;
}

double
zfp_stream_set_rate(zfp_stream* zfp, double rate, zfp_type type, uint dims, zfp_bool align)
{
  const uint n = 1u << (2 * dims);
  uint bits = (uint)floor(n * rate + 0.5);

  switch (type) {
    case zfp_type_float:
      bits = MAX(bits, 1 + 8u);
      break;
    case zfp_type_double:
      bits = MAX(bits, 1 + 11u);
      break;
    default:
      break;
  }

  /* for write random access, round up to next multiple of stream word size */
  if (align)
    bits = _t1(round_up, uint)(bits, (uint)stream_word_bits);

  zfp->minbits = bits;
  zfp->maxbits = bits;
  zfp->maxprec = ZFP_MAX_PREC;
  zfp->minexp = ZFP_MIN_EXP;

  return (double)bits / n;
}

uint
zfp_stream_set_precision(zfp_stream* zfp, uint precision)
{
  zfp->minbits = ZFP_MIN_BITS;
  zfp->maxbits = ZFP_MAX_BITS;
  zfp->maxprec = precision ? MIN(precision, ZFP_MAX_PREC) : ZFP_MAX_PREC;
  zfp->minexp = ZFP_MIN_EXP;
  return zfp->maxprec;
}

double
zfp_stream_set_accuracy(zfp_stream* zfp, double tolerance)
{
  int emin = ZFP_MIN_EXP;
  if (tolerance > 0) {
    /* tolerance = x * 2^emin, with 0.5 <= x < 1 */
    frexp(tolerance, &emin);
    emin--;
    /* assert: 2^emin <= tolerance < 2^(emin+1) */
  }
  zfp->minbits = ZFP_MIN_BITS;
  zfp->maxbits = ZFP_MAX_BITS;
  zfp->maxprec = ZFP_MAX_PREC;
  zfp->minexp = emin;
  return tolerance > 0 ? ldexp(1.0, emin) : 0;
}

zfp_mode
zfp_stream_set_mode(zfp_stream* zfp, uint64 mode)
{
  uint minbits, maxbits, maxprec;
  int minexp;

  if (mode <= ZFP_MODE_SHORT_MAX) {
    /* 12-bit (short) encoding of one of four modes */
    if (mode < 2048) {
      /* fixed rate */
      minbits = maxbits = (uint)mode + 1;
      maxprec = ZFP_MAX_PREC;
      minexp = ZFP_MIN_EXP;
    }
    else if (mode < (2048 + 128)) {
      /* fixed precision */
      minbits = ZFP_MIN_BITS;
      maxbits = ZFP_MAX_BITS;
      maxprec = (uint)mode + 1 - (2048);
      minexp = ZFP_MIN_EXP;
    }
    else if (mode == (2048 + 128)) {
      /* reversible */
      minbits = ZFP_MIN_BITS;
      maxbits = ZFP_MAX_BITS;
      maxprec = ZFP_MAX_PREC;
      minexp = ZFP_MIN_EXP - 1;
    }
    else {
      /* fixed accuracy */
      minbits = ZFP_MIN_BITS;
      maxbits = ZFP_MAX_BITS;
      maxprec = ZFP_MAX_PREC;
      minexp = (int)mode + ZFP_MIN_EXP - (2048 + 128 + 1);
    }
  }
  else {
    /* 64-bit encoding */
    mode >>= 12; minbits = (uint)(mode & 0x7fffu) + 1;
    mode >>= 15; maxbits = (uint)(mode & 0x7fffu) + 1;
    mode >>= 15; maxprec = (uint)(mode & 0x007fu) + 1;
    mode >>=  7; minexp  = (int)(mode & 0x7fffu) - 16495;
  }

  if (!zfp_stream_set_params(zfp, minbits, maxbits, maxprec, minexp))
    return zfp_mode_null;

  return zfp_stream_compression_mode(zfp);
}

zfp_bool
zfp_stream_set_params(zfp_stream* zfp, uint minbits, uint maxbits, uint maxprec, int minexp)
{
  if (minbits > maxbits || !(0 < maxprec && maxprec <= 64))
    return zfp_false;
  zfp->minbits = minbits;
  zfp->maxbits = maxbits;
  zfp->maxprec = maxprec;
  zfp->minexp = minexp;
  return zfp_true;
}

size_t
zfp_stream_flush(zfp_stream* zfp)
{
  return stream_flush(zfp->stream);
}

size_t
zfp_stream_align(zfp_stream* zfp)
{
  return stream_align(zfp->stream);
}

void
zfp_stream_rewind(zfp_stream* zfp)
{
  stream_rewind(zfp->stream);
}

/* public functions: execution policy -------------------------------------- */

zfp_exec_policy
zfp_stream_execution(const zfp_stream* zfp)
{
  return zfp->exec.policy;
}

uint
zfp_stream_omp_threads(const zfp_stream* zfp)
{
  if (zfp->exec.policy == zfp_exec_omp) 
    return ((zfp_exec_params_omp*)zfp->exec.params)->threads;
  return 0u;
}

uint
zfp_stream_omp_chunk_size(const zfp_stream* zfp)
{
  if (zfp->exec.policy == zfp_exec_omp) 
    return ((zfp_exec_params_omp*)zfp->exec.params)->chunk_size;
  return 0u;
}

zfp_bool
zfp_stream_set_execution(zfp_stream* zfp, zfp_exec_policy policy)
{
  switch (policy) {
    case zfp_exec_serial:
      if (zfp->exec.policy != policy && zfp->exec.params != NULL) {
        free(zfp->exec.params);
        zfp->exec.params = NULL;
      }
      break;
#ifdef ZFP_WITH_CUDA
    case zfp_exec_cuda:
      if (zfp->exec.policy != policy && zfp->exec.params != NULL) {
        free(zfp->exec.params);
        zfp->exec.params = NULL;
      }
      break;
#endif
    case zfp_exec_omp:
#ifdef _OPENMP
      if (zfp->exec.policy != policy) {
        if (zfp->exec.params != NULL) {
          free(zfp->exec.params);
        }
        zfp_exec_params_omp* params = malloc(sizeof(zfp_exec_params_omp));
        params->threads = 0;
        params->chunk_size = 0;
        zfp->exec.params = params;
      }
      break;
#else
      return zfp_false;
#endif
    default:
      return zfp_false;
  }
  zfp->exec.policy = policy;
  return zfp_true;
}

zfp_bool
zfp_stream_set_omp_threads(zfp_stream* zfp, uint threads)
{
  if (!zfp_stream_set_execution(zfp, zfp_exec_omp))
    return zfp_false;
  ((zfp_exec_params_omp*)zfp->exec.params)->threads = threads;
  return zfp_true;
}

zfp_bool
zfp_stream_set_omp_chunk_size(zfp_stream* zfp, uint chunk_size)
{
  if (!zfp_stream_set_execution(zfp, zfp_exec_omp))
    return zfp_false;
  ((zfp_exec_params_omp*)zfp->exec.params)->chunk_size = chunk_size;
  return zfp_true;
}

/* public functions: utility functions --------------------------------------*/

void
zfp_promote_int8_to_int32(int32* oblock, const int8* iblock, uint dims)
{
  uint count = 1u << (2 * dims);
  while (count--)
    *oblock++ = (int32)*iblock++ << 23;
}

void
zfp_promote_uint8_to_int32(int32* oblock, const uint8* iblock, uint dims)
{
  uint count = 1u << (2 * dims);
  while (count--)
    *oblock++ = ((int32)*iblock++ - 0x80) << 23;
}

void
zfp_promote_int16_to_int32(int32* oblock, const int16* iblock, uint dims)
{
  uint count = 1u << (2 * dims);
  while (count--)
    *oblock++ = (int32)*iblock++ << 15;
}

void
zfp_promote_uint16_to_int32(int32* oblock, const uint16* iblock, uint dims)
{
  uint count = 1u << (2 * dims);
  while (count--)
    *oblock++ = ((int32)*iblock++ - 0x8000) << 15;
}

void
zfp_demote_int32_to_int8(int8* oblock, const int32* iblock, uint dims)
{
  uint count = 1u << (2 * dims);
  while (count--) {
    int32 i = *iblock++ >> 23;
    *oblock++ = (int8)MAX(-0x80, MIN(i, 0x7f));
  }
}

void
zfp_demote_int32_to_uint8(uint8* oblock, const int32* iblock, uint dims)
{
  uint count = 1u << (2 * dims);
  while (count--) {
    int32 i = (*iblock++ >> 23) + 0x80;
    *oblock++ = (uint8)MAX(0x00, MIN(i, 0xff));
  }
}

void
zfp_demote_int32_to_int16(int16* oblock, const int32* iblock, uint dims)
{
  uint count = 1u << (2 * dims);
  while (count--) {
    int32 i = *iblock++ >> 15;
    *oblock++ = (int16)MAX(-0x8000, MIN(i, 0x7fff));
  }
}

void
zfp_demote_int32_to_uint16(uint16* oblock, const int32* iblock, uint dims)
{
  uint count = 1u << (2 * dims);
  while (count--) {
    int32 i = (*iblock++ >> 15) + 0x8000;
    *oblock++ = (uint16)MAX(0x0000, MIN(i, 0xffff));
  }
}

/* public functions: compression and decompression --------------------------*/

size_t
zfp_compress(zfp_stream* zfp, const zfp_field* field)
{
  /* function table [execution][strided][dimensionality][scalar type] */
  void (*ftable[3][2][4][4])(zfp_stream*, const zfp_field*) = {
    /* serial */
    {{{ compress_int32_1,         compress_int64_1,         compress_float_1,         compress_double_1 },
      { compress_strided_int32_2, compress_strided_int64_2, compress_strided_float_2, compress_strided_double_2 },
      { compress_strided_int32_3, compress_strided_int64_3, compress_strided_float_3, compress_strided_double_3 },
      { compress_strided_int32_4, compress_strided_int64_4, compress_strided_float_4, compress_strided_double_4 }},
     {{ compress_strided_int32_1, compress_strided_int64_1, compress_strided_float_1, compress_strided_double_1 },
      { compress_strided_int32_2, compress_strided_int64_2, compress_strided_float_2, compress_strided_double_2 },
      { compress_strided_int32_3, compress_strided_int64_3, compress_strided_float_3, compress_strided_double_3 },
      { compress_strided_int32_4, compress_strided_int64_4, compress_strided_float_4, compress_strided_double_4 }}},

    /* OpenMP */
#ifdef _OPENMP
    {{{ compress_omp_int32_1,         compress_omp_int64_1,         compress_omp_float_1,         compress_omp_double_1 },
      { compress_strided_omp_int32_2, compress_strided_omp_int64_2, compress_strided_omp_float_2, compress_strided_omp_double_2 },
      { compress_strided_omp_int32_3, compress_strided_omp_int64_3, compress_strided_omp_float_3, compress_strided_omp_double_3 },
      { compress_strided_omp_int32_4, compress_strided_omp_int64_4, compress_strided_omp_float_4, compress_strided_omp_double_4 }},
     {{ compress_strided_omp_int32_1, compress_strided_omp_int64_1, compress_strided_omp_float_1, compress_strided_omp_double_1 },
      { compress_strided_omp_int32_2, compress_strided_omp_int64_2, compress_strided_omp_float_2, compress_strided_omp_double_2 },
      { compress_strided_omp_int32_3, compress_strided_omp_int64_3, compress_strided_omp_float_3, compress_strided_omp_double_3 },
      { compress_strided_omp_int32_4, compress_strided_omp_int64_4, compress_strided_omp_float_4, compress_strided_omp_double_4 }}},
#else
    {{{ NULL }}},
#endif

    /* CUDA */
#ifdef ZFP_WITH_CUDA
    {{{ compress_cuda_int32_1,         compress_cuda_int64_1,         compress_cuda_float_1,         compress_cuda_double_1 },
      { compress_strided_cuda_int32_2, compress_strided_cuda_int64_2, compress_strided_cuda_float_2, compress_strided_cuda_double_2 },
      { compress_strided_cuda_int32_3, compress_strided_cuda_int64_3, compress_strided_cuda_float_3, compress_strided_cuda_double_3 },
      { NULL,                            NULL,                            NULL,                            NULL }},
     {{ compress_strided_cuda_int32_1, compress_strided_cuda_int64_1, compress_strided_cuda_float_1, compress_strided_cuda_double_1 },
      { compress_strided_cuda_int32_2, compress_strided_cuda_int64_2, compress_strided_cuda_float_2, compress_strided_cuda_double_2 },
      { compress_strided_cuda_int32_3, compress_strided_cuda_int64_3, compress_strided_cuda_float_3, compress_strided_cuda_double_3 },
      { NULL,                            NULL,                            NULL,                            NULL }}},
#else
    {{{ NULL }}},
#endif
  };
  uint exec = zfp->exec.policy;
  uint strided = (uint)zfp_field_stride(field, NULL);
  uint dims = zfp_field_dimensionality(field);
  uint type = field->type;
  void (*compress)(zfp_stream*, const zfp_field*);

  switch (type) {
    case zfp_type_int32:
    case zfp_type_int64:
    case zfp_type_float:
    case zfp_type_double:
      break;
    default:
      return 0;
  }

  /* return 0 if compression mode is not supported */
  compress = ftable[exec][strided][dims - 1][type - zfp_type_int32];
  if (!compress)
    return 0;

  /* compress field and align bit stream on word boundary */
  compress(zfp, field);
  stream_flush(zfp->stream);

  return stream_size(zfp->stream);
}

size_t
zfp_decompress(zfp_stream* zfp, zfp_field* field)
{
  /* function table [execution][strided][dimensionality][scalar type] */
  void (*ftable[3][2][4][4])(zfp_stream*, zfp_field*) = {
    /* serial */
    {{{ decompress_int32_1,         decompress_int64_1,         decompress_float_1,         decompress_double_1 },
      { decompress_strided_int32_2, decompress_strided_int64_2, decompress_strided_float_2, decompress_strided_double_2 },
      { decompress_strided_int32_3, decompress_strided_int64_3, decompress_strided_float_3, decompress_strided_double_3 },
      { decompress_strided_int32_4, decompress_strided_int64_4, decompress_strided_float_4, decompress_strided_double_4 }},
     {{ decompress_strided_int32_1, decompress_strided_int64_1, decompress_strided_float_1, decompress_strided_double_1 },
      { decompress_strided_int32_2, decompress_strided_int64_2, decompress_strided_float_2, decompress_strided_double_2 },
      { decompress_strided_int32_3, decompress_strided_int64_3, decompress_strided_float_3, decompress_strided_double_3 },
      { decompress_strided_int32_4, decompress_strided_int64_4, decompress_strided_float_4, decompress_strided_double_4 }}},

    /* OpenMP; not yet supported */
    {{{ NULL }}},

    /* CUDA */
#ifdef ZFP_WITH_CUDA
    {{{ decompress_cuda_int32_1,         decompress_cuda_int64_1,         decompress_cuda_float_1,         decompress_cuda_double_1 },
      { decompress_strided_cuda_int32_2, decompress_strided_cuda_int64_2, decompress_strided_cuda_float_2, decompress_strided_cuda_double_2 },
      { decompress_strided_cuda_int32_3, decompress_strided_cuda_int64_3, decompress_strided_cuda_float_3, decompress_strided_cuda_double_3 },
      { NULL,                            NULL,                            NULL,                            NULL }},
     {{ decompress_strided_cuda_int32_1, decompress_strided_cuda_int64_1, decompress_strided_cuda_float_1, decompress_strided_cuda_double_1 },
      { decompress_strided_cuda_int32_2, decompress_strided_cuda_int64_2, decompress_strided_cuda_float_2, decompress_strided_cuda_double_2 },
      { decompress_strided_cuda_int32_3, decompress_strided_cuda_int64_3, decompress_strided_cuda_float_3, decompress_strided_cuda_double_3 },
      { NULL,                            NULL,                            NULL,                            NULL }}},
#else
    {{{ NULL }}},
#endif
  };
  uint exec = zfp->exec.policy;
  uint strided = (uint)zfp_field_stride(field, NULL);
  uint dims = zfp_field_dimensionality(field);
  uint type = field->type;
  void (*decompress)(zfp_stream*, zfp_field*);

  switch (type) {
    case zfp_type_int32:
    case zfp_type_int64:
    case zfp_type_float:
    case zfp_type_double:
      break;
    default:
      return 0;
  }

  /* return 0 if decompression mode is not supported */
  decompress = ftable[exec][strided][dims - 1][type - zfp_type_int32];
  if (!decompress)
    return 0;

  /* decompress field and align bit stream on word boundary */
  decompress(zfp, field);
  stream_align(zfp->stream);

  return stream_size(zfp->stream);
}

size_t
zfp_write_header(zfp_stream* zfp, const zfp_field* field, uint mask)
{
  size_t bits = 0;
  uint64 meta = 0;

  /* first make sure field dimensions fit in header */
  if (mask & ZFP_HEADER_META) {
    meta = zfp_field_metadata(field);
    if (meta == ZFP_META_NULL)
      return 0;
  }

  /* 32-bit magic */
  if (mask & ZFP_HEADER_MAGIC) {
    stream_write_bits(zfp->stream, 'z', 8);
    stream_write_bits(zfp->stream, 'f', 8);
    stream_write_bits(zfp->stream, 'p', 8);
    stream_write_bits(zfp->stream, zfp_codec_version, 8);
    bits += ZFP_MAGIC_BITS;
  }
  /* 52-bit field metadata */
  if (mask & ZFP_HEADER_META) {
    stream_write_bits(zfp->stream, meta, ZFP_META_BITS);
    bits += ZFP_META_BITS;
  }
  /* 12- or 64-bit compression parameters */
  if (mask & ZFP_HEADER_MODE) {
    uint64 mode = zfp_stream_mode(zfp);
    uint size = mode > ZFP_MODE_SHORT_MAX ? ZFP_MODE_LONG_BITS : ZFP_MODE_SHORT_BITS;
    stream_write_bits(zfp->stream, mode, size);
    bits += size;
  }

  return bits;
}

size_t
zfp_read_header(zfp_stream* zfp, zfp_field* field, uint mask)
{
  size_t bits = 0;
  if (mask & ZFP_HEADER_MAGIC) {
    if (stream_read_bits(zfp->stream, 8) != 'z' ||
        stream_read_bits(zfp->stream, 8) != 'f' ||
        stream_read_bits(zfp->stream, 8) != 'p' ||
        stream_read_bits(zfp->stream, 8) != zfp_codec_version)
      return 0;
    bits += ZFP_MAGIC_BITS;
  }
  if (mask & ZFP_HEADER_META) {
    uint64 meta = stream_read_bits(zfp->stream, ZFP_META_BITS);
    if (!zfp_field_set_metadata(field, meta))
      return 0;
    bits += ZFP_META_BITS;
  }
  if (mask & ZFP_HEADER_MODE) {
    uint64 mode = stream_read_bits(zfp->stream, ZFP_MODE_SHORT_BITS);
    bits += ZFP_MODE_SHORT_BITS;
    if (mode > ZFP_MODE_SHORT_MAX) {
      uint size = ZFP_MODE_LONG_BITS - ZFP_MODE_SHORT_BITS;
      mode += stream_read_bits(zfp->stream, size) << ZFP_MODE_SHORT_BITS;
      bits += size;
    }
    if (zfp_stream_set_mode(zfp, mode) == zfp_mode_null)
      return 0;
  }
  return bits;
}
