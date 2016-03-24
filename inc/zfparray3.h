#ifndef ZFP_ARRAY3_H
#define ZFP_ARRAY3_H

#include <algorithm>
#include <climits>
#include <cmath>
#include <cstdlib>
#include "types.h"
#include "memory.h"
#include "cache.h"
#include "zfpcodec3.h"

namespace ZFP {

// compressed 3D array of scalars
template <typename Scalar>
class Array3 {
public:
  Array3() : nx(0), ny(0), nz(0), mx(0), my(0), mz(0), blksize(0), bytes(0), data(0), codec(stream, 0, 0), cache(0), dims(0) {}

  // constructor of nx * ny * nz array using rate bits per value, at least
  // csize bytes of cache, and optionally initialized from flat array p
  Array3(uint nx, uint ny, uint nz, double rate, const Scalar* p = 0, size_t csize = 0) :
    blksize(block_size(rate)),
    bytes(0),
    data(0),
    codec(stream, 0, uint(CHAR_BIT * blksize)),
    cache(lines(csize, nx, ny, nz)),
    dims(0)
  {
    resize(nx, ny, nz, p == 0);
    if (p)
      set(p);
  }

  // destructor
  ~Array3() { free(); }

  // total number of elements in array
  size_t size() const { return size_t(nx) * size_t(ny) * size_t(nz); }

  // array dimensions
  uint size_x() const { return nx; }
  uint size_y() const { return ny; }
  uint size_z() const { return nz; }

  // resize the array (all previously stored data will be lost)
  void resize(uint nx, uint ny, uint nz, bool clear = true)
  {
    if (nx == 0 || ny == 0 || nz == 0)
      free();
    else {
      this->nx = nx;
      this->ny = ny;
      this->nz = nz;
      mx = (nx + 3) / 4;
      my = (ny + 3) / 4;
      mz = (nz + 3) / 4;
      alloc(clear);

      // precompute block dimensions
      deallocate(dims);
      if ((nx | ny | nz) & 3u) {
        dims = (uchar*)allocate(mx * my * mz);
        uchar* d = dims;
        for (uint k = 0; k < mz; k++)
          for (uint j = 0; j < my; j++)
            for (uint i = 0; i < mx; i++)
              *d++ = (i == mx - 1 ? -nx & 3u : 0) + 4 * ((j == my - 1 ? -ny & 3u : 0) + 4 * (k == mz - 1 ? -nz & 3u : 0));
      }
      else
        dims = 0;
    }
  }

  // rate in bits per value
  double rate() const { return CHAR_BIT * blksize / 64.0; }

  // set compression rate in bits per value
  void set_rate(double rate)
  {
    blksize = block_size(rate);
    codec.configure(0, CHAR_BIT * blksize, 0, INT_MIN),
    alloc();
  }

  // cache size in number of bytes
  size_t cache_size() const { return cache.size() * sizeof(CacheLine); }

  // set minimum cache size in bytes (array dimensions must be known)
  void set_cache_size(size_t csize)
  {
    flush();
    cache.resize(lines(csize, nx, ny, nz));
  }

  // flush cache by compressing all modified cached blocks
  void flush() const
  {
    for (typename Cache<CacheLine>::const_iterator p = cache.first(); p; p++) {
      if (p->tag.dirty()) {
        uint b = p->tag.index() - 1;
        stream.seek(b * blksize);
        codec.encode(p->line->a, 1, 4, 16, dims ? dims[b] : 0);
        stream.flush();
      }
      cache.flush(p->line);
    }
  }

  // decompress array and store at p
  void get(Scalar* p) const
  {
    size_t offset = 0;
    const uchar* d = dims;
    for (uint k = 0; k < mz; k++, p += 4 * nx * (ny - my))
      for (uint j = 0; j < my; j++, p += 4 * (nx - mx))
        for (uint i = 0; i < mx; i++, p += 4, offset += blksize) {
          uint b = block(i, j, k);
          const CacheLine* line = cache.lookup(b + 1);
          if (line)
            line->get(p, 1, nx, nx * ny, d ? *d++ : 0);
          else {
            stream.seek(offset);
            codec.decode(p, 1, nx, nx * ny, d ? *d++ : 0);
          }
        }
  }

  // initialize array by copying and compressing data stored at p
  void set(const Scalar* p)
  {
    size_t offset = 0;
    const uchar* d = dims;
    for (uint k = 0; k < mz; k++, p += 4 * nx * (ny - my))
      for (uint j = 0; j < my; j++, p += 4 * (nx - mx))
        for (uint i = 0; i < mx; i++, p += 4, offset += blksize) {
          stream.seek(offset);
          codec.encode(p, 1, nx, nx * ny, d ? *d++ : 0);
          stream.flush();
        }
    cache.clear();
  }

  // number of bytes of compressed data
  size_t compressed_size() const { return bytes; }

  // pointer to compressed data for read or write access
  uchar* compressed_data() const
  {
    // first write back any modified cached data
    flush();
    return data;
  }

  // reference to a single array value
  class reference {
  public:
    operator Scalar() const { return array->get(i, j, k); }
    reference operator=(const reference& r) { array->set(i, j, k, r.operator Scalar()); return *this; }
    reference operator=(Scalar val) { array->set(i, j, k, val); return *this; }
    reference operator+=(Scalar val) { array->add(i, j, k, val); return *this; }
    reference operator-=(Scalar val) { array->sub(i, j, k, val); return *this; }
    reference operator*=(Scalar val) { array->mul(i, j, k, val); return *this; }
    reference operator/=(Scalar val) { array->div(i, j, k, val); return *this; }
  protected:
    friend class Array3;
    reference(Array3* array, uint i, uint j, uint k) : array(array), i(i), j(j), k(k) {}
    Array3* array;
    uint i, j, k;
  };

  // (i, j, k) accessors
  const Scalar& operator()(uint i, uint j, uint k) const { return get(i, j, k); }
  reference operator()(uint i, uint j, uint k) { return reference(this, i, j, k); }

  // flat index accessors
  const Scalar& operator[](uint index) const
  {
    uint i, j, k;
    ijk(i, j, k, index);
    return get(i, j, k);
  }
  reference operator[](uint index)
  {
    uint i, j, k;
    ijk(i, j, k, index);
    return reference(this, i, j, k);
  }

protected:
  // cache line representing one block of decompressed values
  class CacheLine {
  public:
    friend class Array3;
    const Scalar& operator()(uint i, uint j, uint k) const { return a[index(i, j, k)]; }
    Scalar& operator()(uint i, uint j, uint k) { return a[index(i, j, k)]; }
    // copy cache line
    void get(Scalar* p, uint sx, uint sy, uint sz) const
    {
      const Scalar* q = a;
      for (uint z = 0; z < 4; z++, p += sz - 4 * sy)
        for (uint y = 0; y < 4; y++, p += sy - 4 * sx)
          for (uint x = 0; x < 4; x++, p += sx, q++)
            *p = *q;
    }
    void get(Scalar* p, uint sx, uint sy, uint sz, uchar dims) const
    {
      if (!dims)
        get(p, sx, sy, sz);
      else {
        // determine block dimensions
        uint nx = 4 - (dims & 3u); dims >>= 2;
        uint ny = 4 - (dims & 3u); dims >>= 2;
        uint nz = 4 - (dims & 3u); dims >>= 2;
        const Scalar* q = a;
        for (uint z = 0; z < nz; z++, p += sz - ny * sy, q += 16 - 4 * ny)
          for (uint y = 0; y < ny; y++, p += sy - nx * sx, q += 4 - nx)
            for (uint x = 0; x < nx; x++, p += sx, q++)
              *p = *q;
      }
    }
  protected:
    static uint index(uint i, uint j, uint k) { return (i & 3) + 4 * ((j & 3) + 4 * (k & 3)); }
    Scalar a[64];
  };

  // inspector
  const Scalar& get(uint i, uint j, uint k) const
  {
    CacheLine* p = line(i, j, k, false);
    return (*p)(i, j, k);
  }

  // mutator
  void set(uint i, uint j, uint k, Scalar val)
  {
    CacheLine* p = line(i, j, k, true);
    (*p)(i, j, k) = val;
  }

  // in-place updates
  void add(uint i, uint j, uint k, Scalar val) { (*line(i, j, k, true))(i, j, k) += val; }
  void sub(uint i, uint j, uint k, Scalar val) { (*line(i, j, k, true))(i, j, k) -= val; }
  void mul(uint i, uint j, uint k, Scalar val) { (*line(i, j, k, true))(i, j, k) *= val; }
  void div(uint i, uint j, uint k, Scalar val) { (*line(i, j, k, true))(i, j, k) /= val; }

  // return cache line for (i, j, k); may require write-back and fetch
  CacheLine* line(uint i, uint j, uint k, bool write) const
  {
    CacheLine* p = 0;
    uint b = block(i, j, k);
    typename Cache<CacheLine>::Tag t = cache.access(p, b + 1, write);
    uint c = t.index() - 1;
    if (c != b) {
      if (t.dirty()) {
        // write back dirty cache line
        stream.seek(c * blksize);
        codec.encode(p->a, 1, 4, 16, dims ? dims[c] : 0);
        stream.flush();
      }
      // fetch cache line
      stream.seek(b * blksize);
      codec.decode(p->a, 1, 4, 16);
    }
    return p;
  }

  // allocate memory for compressed data
  void alloc(bool clear = true)
  {
    bytes = mx * my * mz * blksize;
    reallocate(data, bytes, 0x100u);
    if (clear)
      std::fill(data, data + bytes, 0);
    stream.open(data, bytes);
    cache.clear();
  }

  // free memory associated with compressed data
  void free()
  {
    nx = ny = nz = 0;
    mx = my = mz = 0;
    bytes = 0;
    deallocate(data);
    data = 0;
    deallocate(dims);
    dims = 0;
  }

  // block index for (i, j, k)
  uint block(uint i, uint j, uint k) const { return (i / 4) + mx * ((j / 4) + my * (k / 4)); }

  // convert flat index to (i, j, k)
  void ijk(uint& i, uint& j, uint& k, uint index) const
  {
    i = index % nx;
    index /= nx;
    j = index % ny;
    index /= ny;
    k = index;
  }

  // compressed block size in bytes for given rate
  static size_t block_size(double rate) { return (lrint(64 * rate) + CHAR_BIT - 1) / CHAR_BIT; }

  // number of cache lines corresponding to size (or suggested size if zero)
  static uint lines(size_t size, uint nx, uint ny, uint nz)
  {
    uint n = uint((size ? size : 8 * nx * ny * sizeof(Scalar)) / sizeof(CacheLine));
    return std::max(n, 1u);
  }

  uint nx, ny, nz; // array dimensions
  uint mx, my, mz; // array dimensions in number of 4x4x4 blocks
  size_t blksize; // byte size of single compressed block
  size_t bytes; // total bytes of compressed data
  mutable uchar* data; // pointer to compressed data
  mutable MemoryBitStream stream; // bit stream for compressed data
  mutable Codec3<MemoryBitStream, Scalar> codec; // compression codec
  mutable Cache<CacheLine> cache; // cache of decompressed blocks
  uchar* dims; // precomputed block dimensions (or null if uniform)
};

typedef Array3<float> Array3f;
typedef Array3<double> Array3d;

}

#endif
