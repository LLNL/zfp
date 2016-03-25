#ifndef ZFP_ARRAY2_H
#define ZFP_ARRAY2_H

#include <algorithm>
#include <climits>
#include <cmath>
#include <cstdlib>
#include "types.h"
#include "memory.h"
#include "cache.h"
#include "zfpcodec2.h"

namespace ZFP {

// compressed 2D array of scalars
template <typename Scalar>
class Array2 {
public:
  Array2() : nx(0), ny(0), mx(0), my(0), blksize(0), bytes(0), data(0), codec(stream, 0, 0), cache(0), dims(0) {}

  // constructor of nx * ny array using rate bits per value, at least
  // csize bytes of cache, and optionally initialized from flat array p
  Array2(uint nx, uint ny, double rate, const Scalar* p = 0, size_t csize = 0) :
    blksize(block_size(rate)),
    bytes(0),
    data(0),
    codec(stream, 0, uint(CHAR_BIT * blksize)),
    cache(lines(csize, nx, ny)),
    dims(0)
  {
    resize(nx, ny, p == 0);
    if (p)
      set(p);
  }

  // destructor
  ~Array2() { free(); }

  // total number of elements in array
  size_t size() const { return size_t(nx) * size_t(ny); }

  // array dimensions
  uint size_x() const { return nx; }
  uint size_y() const { return ny; }

  // resize the array (all previously stored data will be lost)
  void resize(uint nx, uint ny, bool clear = true)
  {
    if (nx == 0 || ny == 0)
      free();
    else {
      this->nx = nx;
      this->ny = ny;
      mx = (nx + 3) / 4;
      my = (ny + 3) / 4;
      alloc(clear);

      // precompute block dimensions
      deallocate(dims);
      if ((nx | ny) & 3u) {
        dims = (uchar*)allocate(mx * my);
        uchar* d = dims;
        for (uint j = 0; j < my; j++)
          for (uint i = 0; i < mx; i++)
            *d++ = (i == mx - 1 ? -nx & 3u : 0) + 4 * (j == my - 1 ? -ny & 3u : 0);
      }
      else
        dims = 0;
    }
  }

  // rate in bits per value
  double rate() const { return CHAR_BIT * blksize / 16.0; }

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
    cache.resize(lines(csize, nx, ny));
  }

  // flush cache by compressing all modified cached blocks
  void flush() const
  {
    for (typename Cache<CacheLine>::const_iterator p = cache.first(); p; p++) {
      if (p->tag.dirty()) {
        uint b = p->tag.index() - 1;
        stream.seek(b * blksize);
        codec.encode(p->line->a, 1, 4, dims ? dims[b] : 0);
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
    for (uint j = 0; j < my; j++, p += 4 * (nx - mx))
      for (uint i = 0; i < mx; i++, p += 4, offset += blksize) {
        uint b = block(i, j);
        const CacheLine* line = cache.lookup(b + 1);
        if (line)
          line->get(p, 1, nx, d ? *d++ : 0);
        else {
          stream.seek(offset);
          codec.decode(p, 1, nx, d ? *d++ : 0);
        }
      }
  }

  // initialize array by copying and compressing data stored at p
  void set(const Scalar* p)
  {
    size_t offset = 0;
    const uchar* d = dims;
    for (uint j = 0; j < my; j++, p += 4 * (nx - mx))
      for (uint i = 0; i < mx; i++, p += 4, offset += blksize) {
        stream.seek(offset);
        codec.encode(p, 1, nx, d ? *d++ : 0);
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
    operator Scalar() const { return array->get(i, j); }
    reference operator=(const reference& r) { array->set(i, j, r.operator Scalar()); return *this; }
    reference operator=(Scalar val) { array->set(i, j, val); return *this; }
    reference operator+=(Scalar val) { array->add(i, j, val); return *this; }
    reference operator-=(Scalar val) { array->sub(i, j, val); return *this; }
    reference operator*=(Scalar val) { array->mul(i, j, val); return *this; }
    reference operator/=(Scalar val) { array->div(i, j, val); return *this; }
  protected:
    friend class Array2;
    reference(Array2* array, uint i, uint j) : array(array), i(i), j(j) {}
    Array2* array;
    uint i, j;
  };

  // (i, j) accessors
  const Scalar& operator()(uint i, uint j) const { return get(i, j); }
  reference operator()(uint i, uint j) { return reference(this, i, j); }

  // flat index accessors
  const Scalar& operator[](uint index) const
  {
    uint i, j;
    ij(i, j, index);
    return get(i, j);
  }
  reference operator[](uint index)
  {
    uint i, j;
    ij(i, j, index);
    return reference(this, i, j);
  }

protected:
  // cache line representing one block of decompressed values
  class CacheLine {
  public:
    friend class Array2;
    const Scalar& operator()(uint i, uint j) const { return a[index(i, j)]; }
    Scalar& operator()(uint i, uint j) { return a[index(i, j)]; }
    // copy cache line
    void get(Scalar* p, uint sx, uint sy) const
    {
      const Scalar* q = a;
      for (uint y = 0; y < 4; y++, p += sy - 4 * sx)
        for (uint x = 0; x < 4; x++, p += sx, q++)
          *p = *q;
    }
    void get(Scalar* p, uint sx, uint sy, uchar dims) const
    {
      if (!dims)
        get(p, sx, sy);
      else {
        // determine block dimensions
        uint nx = 4 - (dims & 3u); dims >>= 2;
        uint ny = 4 - (dims & 3u); dims >>= 2;
        const Scalar* q = a;
        for (uint y = 0; y < ny; y++, p += sy - nx * sx, q += 4 - nx)
          for (uint x = 0; x < nx; x++, p += sx, q++)
            *p = *q;
      }
    }
  protected:
    static uint index(uint i, uint j) { return (i & 3) + 4 * (j & 3); }
    Scalar a[16];
  };

  // inspector
  const Scalar& get(uint i, uint j) const
  {
    CacheLine* p = line(i, j, false);
    return (*p)(i, j);
  }

  // mutator
  void set(uint i, uint j, Scalar val)
  {
    CacheLine* p = line(i, j, true);
    (*p)(i, j) = val;
  }

  // in-place updates
  void add(uint i, uint j, Scalar val) { (*line(i, j, true))(i, j) += val; }
  void sub(uint i, uint j, Scalar val) { (*line(i, j, true))(i, j) -= val; }
  void mul(uint i, uint j, Scalar val) { (*line(i, j, true))(i, j) *= val; }
  void div(uint i, uint j, Scalar val) { (*line(i, j, true))(i, j) /= val; }

  // return cache line for (i, j); may require write-back and fetch
  CacheLine* line(uint i, uint j, bool write) const
  {
    CacheLine* p = 0;
    uint b = block(i, j);
    typename Cache<CacheLine>::Tag t = cache.access(p, b + 1, write);
    uint c = t.index() - 1;
    if (c != b) {
      if (t.dirty()) {
        // write back dirty cache line
        stream.seek(c * blksize);
        codec.encode(p->a, 1, 4, dims ? dims[c] : 0);
        stream.flush();
      }
      // fetch cache line
      stream.seek(b * blksize);
      codec.decode(p->a, 1, 4);
    }
    return p;
  }

  // allocate memory for compressed data
  void alloc(bool clear = true)
  {
    bytes = mx * my * blksize;
    reallocate(data, bytes, 0x100u);
    if (clear)
      std::fill(data, data + bytes, 0);
    stream.open(data, bytes);
    cache.clear();
  }

  // free memory associated with compressed data
  void free()
  {
    nx = ny = 0;
    mx = my = 0;
    bytes = 0;
    deallocate(data);
    data = 0;
    deallocate(dims);
    dims = 0;
  }

  // block index for (i, j)
  uint block(uint i, uint j) const { return (i / 4) + mx * (j / 4); }

  // convert flat index to (i, j)
  void ij(uint& i, uint& j, uint index) const
  {
    i = index % nx;
    index /= nx;
    j = index;
  }

  // compressed block size in bytes for given rate
  static size_t block_size(double rate) { return (lrint(16 * rate) + CHAR_BIT - 1) / CHAR_BIT; }

  // number of cache lines corresponding to size (or suggested size if zero)
  static uint lines(size_t size, uint nx, uint ny)
  {
    uint n = uint((size ? size : 8 * nx * sizeof(Scalar)) / sizeof(CacheLine));
    return std::max(n, 1u);
  }

  uint nx, ny; // array dimensions
  uint mx, my; // array dimensions in number of 4x4 blocks
  size_t blksize; // byte size of single compressed block
  size_t bytes; // total bytes of compressed data
  mutable uchar* data; // pointer to compressed data
  mutable MemoryBitStream stream; // bit stream for compressed data
  mutable Codec2<MemoryBitStream, Scalar> codec; // compression codec
  mutable Cache<CacheLine> cache; // cache of decompressed blocks
  uchar* dims; // precomputed block dimensions (or null if uniform)
};

typedef Array2<float> Array2f;
typedef Array2<double> Array2d;

}

#endif
