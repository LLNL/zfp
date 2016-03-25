#include "zfp.h"
#include "zfpcodec1.h"
#include "zfpcodec2.h"
#include "zfpcodec3.h"

// private functions ----------------------------------------------------------

static size_t
compress1f(MemoryBitStream& stream, const float* in, uint nx, uint minbits, uint maxbits, uint maxprec, int minexp)
{
  ZFP::Codec1f<MemoryBitStream> codec(stream, minbits, maxbits, maxprec, minexp);
  const float* p = in;
  for (uint x = 0; x < nx; x += 4, p += 4)
    codec.encode(p, 1, codec.dims(std::min(nx - x, 4u)));
  stream.flush();
  return stream.size();
}

static size_t
compress1d(MemoryBitStream& stream, const double* in, uint nx, uint minbits, uint maxbits, uint maxprec, int minexp)
{
  ZFP::Codec1d<MemoryBitStream> codec(stream, minbits, maxbits, maxprec, minexp);
  const double* p = in;
  for (uint x = 0; x < nx; x += 4, p += 4)
    codec.encode(p, 1, codec.dims(std::min(nx - x, 4u)));
  stream.flush();
  return stream.size();
}

static void
decompress1f(MemoryBitStream& stream, float* out, uint nx, uint minbits, uint maxbits, uint maxprec, int minexp)
{
  ZFP::Codec1f<MemoryBitStream> codec(stream, minbits, maxbits, maxprec, minexp);
  float* p = out;
  for (uint x = 0; x < nx; x += 4, p += 4)
    codec.decode(p, 1, codec.dims(std::min(nx - x, 4u)));
}

static void
decompress1d(MemoryBitStream& stream, double* out, uint nx, uint minbits, uint maxbits, uint maxprec, int minexp)
{
  ZFP::Codec1d<MemoryBitStream> codec(stream, minbits, maxbits, maxprec, minexp);
  double* p = out;
  for (uint x = 0; x < nx; x += 4, p += 4)
    codec.decode(p, 1, codec.dims(std::min(nx - x, 4u)));
}

static size_t
compress2f(MemoryBitStream& stream, const float* in, uint nx, uint ny, uint minbits, uint maxbits, uint maxprec, int minexp)
{
  ZFP::Codec2f<MemoryBitStream> codec(stream, minbits, maxbits, maxprec, minexp);
  const float* p = in;
  for (uint y = 0; y < ny; y += 4, p += 4 * (nx - (nx + 3) / 4))
    for (uint x = 0; x < nx; x += 4, p += 4)
      codec.encode(p, 1, nx, codec.dims(std::min(nx - x, 4u), std::min(ny - y, 4u)));
  stream.flush();
  return stream.size();
}

static size_t
compress2d(MemoryBitStream& stream, const double* in, uint nx, uint ny, uint minbits, uint maxbits, uint maxprec, int minexp)
{
  ZFP::Codec2d<MemoryBitStream> codec(stream, minbits, maxbits, maxprec, minexp);
  const double* p = in;
  for (uint y = 0; y < ny; y += 4, p += 4 * (nx - (nx + 3) / 4))
    for (uint x = 0; x < nx; x += 4, p += 4)
      codec.encode(p, 1, nx, codec.dims(std::min(nx - x, 4u), std::min(ny - y, 4u)));
  stream.flush();
  return stream.size();
}

static void
decompress2f(MemoryBitStream& stream, float* out, uint nx, uint ny, uint minbits, uint maxbits, uint maxprec, int minexp)
{
  ZFP::Codec2f<MemoryBitStream> codec(stream, minbits, maxbits, maxprec, minexp);
  float* p = out;
  for (uint y = 0; y < ny; y += 4, p += 4 * (nx - (nx + 3) / 4))
    for (uint x = 0; x < nx; x += 4, p += 4)
      codec.decode(p, 1, nx, codec.dims(std::min(nx - x, 4u), std::min(ny - y, 4u)));
}

static void
decompress2d(MemoryBitStream& stream, double* out, uint nx, uint ny, uint minbits, uint maxbits, uint maxprec, int minexp)
{
  ZFP::Codec2d<MemoryBitStream> codec(stream, minbits, maxbits, maxprec, minexp);
  double* p = out;
  for (uint y = 0; y < ny; y += 4, p += 4 * (nx - (nx + 3) / 4))
    for (uint x = 0; x < nx; x += 4, p += 4)
      codec.decode(p, 1, nx, codec.dims(std::min(nx - x, 4u), std::min(ny - y, 4u)));
}

static size_t
compress3f(MemoryBitStream& stream, const float* in, uint nx, uint ny, uint nz, uint minbits, uint maxbits, uint maxprec, int minexp)
{
  ZFP::Codec3f<MemoryBitStream> codec(stream, minbits, maxbits, maxprec, minexp);
  const float* p = in;
  for (uint z = 0; z < nz; z += 4, p += 4 * nx * (ny - (ny + 3) / 4))
    for (uint y = 0; y < ny; y += 4, p += 4 * (nx - (nx + 3) / 4))
      for (uint x = 0; x < nx; x += 4, p += 4)
        codec.encode(p, 1, nx, nx * ny, codec.dims(std::min(nx - x, 4u), std::min(ny - y, 4u), std::min(nz - z, 4u)));
  stream.flush();
  return stream.size();
}

static size_t
compress3d(MemoryBitStream& stream, const double* in, uint nx, uint ny, uint nz, uint minbits, uint maxbits, uint maxprec, int minexp)
{
  ZFP::Codec3d<MemoryBitStream> codec(stream, minbits, maxbits, maxprec, minexp);
  const double* p = in;
  for (uint z = 0; z < nz; z += 4, p += 4 * nx * (ny - (ny + 3) / 4))
    for (uint y = 0; y < ny; y += 4, p += 4 * (nx - (nx + 3) / 4))
      for (uint x = 0; x < nx; x += 4, p += 4)
        codec.encode(p, 1, nx, nx * ny, codec.dims(std::min(nx - x, 4u), std::min(ny - y, 4u), std::min(nz - z, 4u)));
  stream.flush();
  return stream.size();
}

static void
decompress3f(MemoryBitStream& stream, float* out, uint nx, uint ny, uint nz, uint minbits, uint maxbits, uint maxprec, int minexp)
{
  ZFP::Codec3f<MemoryBitStream> codec(stream, minbits, maxbits, maxprec, minexp);
  float* p = out;
  for (uint z = 0; z < nz; z += 4, p += 4 * nx * (ny - (ny + 3) / 4))
    for (uint y = 0; y < ny; y += 4, p += 4 * (nx - (nx + 3) / 4))
      for (uint x = 0; x < nx; x += 4, p += 4)
        codec.decode(p, 1, nx, nx * ny, codec.dims(std::min(nx - x, 4u), std::min(ny - y, 4u), std::min(nz - z, 4u)));
}

static void
decompress3d(MemoryBitStream& stream, double* out, uint nx, uint ny, uint nz, uint minbits, uint maxbits, uint maxprec, int minexp)
{
  ZFP::Codec3d<MemoryBitStream> codec(stream, minbits, maxbits, maxprec, minexp);
  double* p = out;
  for (uint z = 0; z < nz; z += 4, p += 4 * nx * (ny - (ny + 3) / 4))
    for (uint y = 0; y < ny; y += 4, p += 4 * (nx - (nx + 3) / 4))
      for (uint x = 0; x < nx; x += 4, p += 4)
        codec.decode(p, 1, nx, nx * ny, codec.dims(std::min(nx - x, 4u), std::min(ny - y, 4u), std::min(nz - z, 4u)));
}

// public functions -----------------------------------------------------------

void
zfp_init(zfp_params* p)
{
  p->type = ZFP_TYPE_FLOAT;
  p->nx = 0;
  p->ny = 0;
  p->nz = 0;
  p->minbits = 0;
  p->maxbits = 0;
  p->maxprec = 0;
  p->minexp = INT_MIN;
}

uint
zfp_set_type(zfp_params* p, uint type)
{
  switch (type) {
    case ZFP_TYPE_FLOAT:
    case ZFP_TYPE_DOUBLE:
      p->type = type;
      return type;
    default:
      return 0;
  }
}

void
zfp_set_size_1d(zfp_params* p, uint n)
{
  p->nx = n;
  p->ny = 0;
  p->nz = 0;
}

void
zfp_set_size_2d(zfp_params* p, uint nx, uint ny)
{
  p->nx = nx;
  p->ny = ny;
  p->nz = 0;
}

void
zfp_set_size_3d(zfp_params* p, uint nx, uint ny, uint nz)
{
  p->nx = nx;
  p->ny = ny;
  p->nz = nz;
}

double
zfp_set_rate(zfp_params* p, double rate)
{
  uint dims = p->nz ? 3 : p->ny ? 2 : 1;
  uint n = 1u << (2 * dims);
  uint bits = lrint(n * rate);
  bits = std::max(bits, p->type == ZFP_TYPE_FLOAT ? 8u : 11u);
  p->minbits = bits;
  p->maxbits = bits;
  p->maxprec = 0;
  p->minexp = INT_MIN;
  return double(bits) / n;
}

uint
zfp_set_precision(zfp_params* p, uint precision)
{
  uint maxprec = p->type * CHAR_BIT * sizeof(float);
  p->minbits = 0;
  p->maxbits = UINT_MAX;
  p->maxprec = precision ? std::min(maxprec, precision) : maxprec;
  p->minexp = INT_MIN;
  return precision;
}

double
zfp_set_accuracy(zfp_params* p, double tolerance)
{
  int emin = INT_MIN;
  if (tolerance > 0) {
    frexp(tolerance, &emin);
    emin--;
  }
  p->minbits = 0;
  p->maxbits = UINT_MAX;
  p->maxprec = 0;
  p->minexp = emin;
  return tolerance > 0 ? ldexp(1.0, emin) : 0;
}

size_t
zfp_estimate_compressed_size(const zfp_params* p)
{
  if (!p->nx)
    return 0;
  bool dp;
  switch (p->type) {
    case ZFP_TYPE_FLOAT:
      dp = false;
      break;
    case ZFP_TYPE_DOUBLE:
      dp = true;
      break;
    default:
      return 0;
  }
  uint mx = (std::max(p->nx, 1u) + 3) / 4;
  uint my = (std::max(p->ny, 1u) + 3) / 4;
  uint mz = (std::max(p->nz, 1u) + 3) / 4;
  size_t blocks = size_t(mx) * size_t(my) * size_t(mz);
  uint dims = 1 + (p->ny == 0 ? 0 : 1) + (p->nz == 0 ? 0 : 1);
  uint values = 1u << (2 * dims);
  uint maxprec = p->maxprec ? p->maxprec : p->type * CHAR_BIT * sizeof(float);
  uint maxbits = (dp ? 11 : 8) + 3 * dims + values * (maxprec + 1);
  maxbits = std::min(maxbits, p->maxbits);
  maxbits = std::max(maxbits, p->minbits);
  return (blocks * maxbits + CHAR_BIT - 1) / CHAR_BIT;
}

size_t
zfp_compress(const zfp_params* p, const void* in, void* out, size_t outsize)
{
  if (!p->nx)
    return 0;
  MemoryBitStream stream;
  stream.open(out, outsize);
  bool dp = (p->type == ZFP_TYPE_DOUBLE);
  if (p->nz)
    return dp ? compress3d(stream, (const double*)in, p->nx, p->ny, p->nz, p->minbits, p->maxbits, p->maxprec, p->minexp)
              : compress3f(stream, (const float*)in,  p->nx, p->ny, p->nz, p->minbits, p->maxbits, p->maxprec, p->minexp);
  else if (p->ny)
    return dp ? compress2d(stream, (const double*)in, p->nx, p->ny, p->minbits, p->maxbits, p->maxprec, p->minexp)
              : compress2f(stream, (const float*)in,  p->nx, p->ny, p->minbits, p->maxbits, p->maxprec, p->minexp);
  else
    return dp ? compress1d(stream, (const double*)in, p->nx, p->minbits, p->maxbits, p->maxprec, p->minexp)
              : compress1f(stream, (const float*)in,  p->nx, p->minbits, p->maxbits, p->maxprec, p->minexp);
}

int
zfp_decompress(const zfp_params* p, void* out, const void* in, size_t insize)
{
  if (!p->nx)
    return false;
  MemoryBitStream stream;
  stream.open((void*)in, insize);
  bool dp = (p->type == ZFP_TYPE_DOUBLE);
  if (p->nz)
    dp ? decompress3d(stream, (double*)out, p->nx, p->ny, p->nz, p->minbits, p->maxbits, p->maxprec, p->minexp)
       : decompress3f(stream, (float*)out,  p->nx, p->ny, p->nz, p->minbits, p->maxbits, p->maxprec, p->minexp);
  else if (p->ny)
    dp ? decompress2d(stream, (double*)out, p->nx, p->ny, p->minbits, p->maxbits, p->maxprec, p->minexp)
       : decompress2f(stream, (float*)out,  p->nx, p->ny, p->minbits, p->maxbits, p->maxprec, p->minexp);
  else
    dp ? decompress1d(stream, (double*)out, p->nx, p->minbits, p->maxbits, p->maxprec, p->minexp)
       : decompress1f(stream, (float*)out,  p->nx, p->minbits, p->maxbits, p->maxprec, p->minexp);
  return true;
}
