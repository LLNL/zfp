#include "zfpcompress.h"
#include "zfpcodec1f.h"
#include "zfpcodec1d.h"
#include "zfpcodec2f.h"
#include "zfpcodec2d.h"
#include "zfpcodec3f.h"
#include "zfpcodec3d.h"

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

namespace ZFP {

size_t
compress(const void* in, void* out, bool dp, uint nx, uint ny, uint nz, uint minbits, uint maxbits, uint maxprec, int minexp)
{
  if (!nx)
    return 0;
  size_t inbytes = nx * std::max(ny, 1u) * std::max(nz, 1u) * sizeof(double);
  MemoryBitStream stream;
  stream.open(out, 2 * inbytes);
  if (nz)
    return dp ? compress3d(stream, (const double*)in, nx, ny, nz, minbits, maxbits, maxprec, minexp)
              : compress3f(stream, (const float*)in,  nx, ny, nz, minbits, maxbits, maxprec, minexp);
  else if (ny)
    return dp ? compress2d(stream, (const double*)in, nx, ny, minbits, maxbits, maxprec, minexp)
              : compress2f(stream, (const float*)in,  nx, ny, minbits, maxbits, maxprec, minexp);
  else
    return dp ? compress1d(stream, (const double*)in, nx, minbits, maxbits, maxprec, minexp)
              : compress1f(stream, (const float*)in,  nx, minbits, maxbits, maxprec, minexp);
}

bool
decompress(const void* in, void* out, bool dp, uint nx, uint ny, uint nz, uint minbits, uint maxbits, uint maxprec, int minexp)
{
  if (!nx)
    return false;
  size_t inbytes = nx * std::max(ny, 1u) * std::max(nz, 1u) * sizeof(double);
  MemoryBitStream stream;
  stream.open((void*)in, 2 * inbytes);
  if (nz)
    dp ? decompress3d(stream, (double*)out, nx, ny, nz, minbits, maxbits, maxprec, minexp)
       : decompress3f(stream, (float*)out,  nx, ny, nz, minbits, maxbits, maxprec, minexp);
  else if (ny)
    dp ? decompress2d(stream, (double*)out, nx, ny, minbits, maxbits, maxprec, minexp)
       : decompress2f(stream, (float*)out,  nx, ny, minbits, maxbits, maxprec, minexp);
  else
    dp ? decompress1d(stream, (double*)out, nx, minbits, maxbits, maxprec, minexp)
       : decompress1f(stream, (float*)out,  nx, minbits, maxbits, maxprec, minexp);
  return true;
}

}
