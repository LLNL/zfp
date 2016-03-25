#ifndef ZFP_ARRAY1_H
#define ZFP_ARRAY1_H

#include <algorithm>
#include <climits>
#include <cmath>
#include <cstdlib>
#include "types.h"
#include "memory.h"
#include "cache.h"
#include "zfpcodec1.h"

namespace ZFP {

// compressed 1D array of scalars
template <typename Scalar>
class Array1 {
public:
  Array1() : n(0), m(0), blksize(0), bytes(0), data(0), codec(stream, 0, 0), cache(0), dims(0) {}

  // constructor of n-sample array using rate bits per value, at least
  // csize bytes of cache, and optionally initialized from flat array p
  Array1(uint n, double rate, const Scalar* p = 0, size_t csize = 0) :
    blksize(block_size(rate)),
    bytes(0),
    data(0),
    codec(stream, 0, uint(CHAR_BIT * blksize)),
    cache(lines(csize, n)),
    dims(0)
  {
    resize(n, p == 0);
    if (p)
      set(p);
  }

  // destructor
  ~Array1() { free(); }

  // total number of elements in array
  size_t size() const { return size_t(n); }

  // resize the array (all previously stored data will be lost)
  void resize(uint n, bool clear = true)
  {
    if (n == 0)
      free();
    else {
      this->n = n;
      m = (n + 3) / 4;
      alloc(clear);

      // precompute block dimensions
      deallocate(dims);
      if (n & 3u) {
        dims = (uchar*)allocate(m);
        uchar* d = dims;
        for (uint i = 0; i < m; i++)
          *d++ = (i == m - 1 ? -n & 3u : 0);
      }
      else
        dims = 0;
    }
  }

  // rate in bits per value
  double rate() const { return CHAR_BIT * blksize / 4.0; }

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
    cache.resize(lines(csize, n));
  }

  // flush cache by compressing all modified cached blocks
  void flush() const
  {
    for (typename Cache<CacheLine>::const_iterator p = cache.first(); p; p++) {
      if (p->tag.dirty()) {
        uint b = p->tag.index() - 1;
        stream.seek(b * blksize);
        codec.encode(p->line->a, 1, dims ? dims[b] : 0);
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
    for (uint i = 0; i < m; i++, p += 4, offset += blksize) {
      uint b = block(i);
      const CacheLine* line = cache.lookup(b + 1);
      if (line)
        line->get(p, 1, d ? *d++ : 0);
      else {
        stream.seek(offset);
        codec.decode(p, 1, d ? *d++ : 0);
      }
    }
  }

  // initialize array by copying and compressing data stored at p
  void set(const Scalar* p)
  {
    size_t offset = 0;
    const uchar* d = dims;
    for (uint i = 0; i < m; i++, p += 4, offset += blksize) {
      stream.seek(offset);
      codec.encode(p, 1, d ? *d++ : 0);
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
    operator Scalar() const { return array->get(i); }
    reference operator=(const reference& r) { array->set(i, r.operator Scalar()); return *this; }
    reference operator=(Scalar val) { array->set(i, val); return *this; }
    reference operator+=(Scalar val) { array->add(i, val); return *this; }
    reference operator-=(Scalar val) { array->sub(i, val); return *this; }
    reference operator*=(Scalar val) { array->mul(i, val); return *this; }
    reference operator/=(Scalar val) { array->div(i, val); return *this; }
  protected:
    friend class Array1;
    reference(Array1* array, uint i) : array(array), i(i) {}
    Array1* array;
    uint i;
  };

  // (i) accessors
  const Scalar& operator()(uint i) const { return get(i); }
  reference operator()(uint i) { return reference(this, i); }

  // flat index accessors
  const Scalar& operator[](uint index) const { return get(index); }
  reference operator[](uint index) { return reference(this, index); }

protected:
  // cache line representing one block of decompressed values
  class CacheLine {
  public:
    friend class Array1;
    const Scalar& operator()(uint i) const { return a[index(i)]; }
    Scalar& operator()(uint i) { return a[index(i)]; }
    // copy cache line
    void get(Scalar* p, uint sx) const
    {
      const Scalar* q = a;
      for (uint x = 0; x < 4; x++, p += sx, q++)
        *p = *q;
    }
    void get(Scalar* p, uint sx, uchar dims) const
    {
      if (!dims)
        get(p, sx);
      else {
        // determine block dimensions
        uint nx = 4 - (dims & 3u); dims >>= 2;
        const Scalar* q = a;
        for (uint x = 0; x < nx; x++, p += sx, q++)
          *p = *q;
      }
    }
  protected:
    static uint index(uint i) { return i & 3u; }
    Scalar a[4];
  };

  // inspector
  const Scalar& get(uint i) const
  {
    CacheLine* p = line(i, false);
    return (*p)(i);
  }

  // mutator
  void set(uint i, Scalar val)
  {
    CacheLine* p = line(i, true);
    (*p)(i) = val;
  }

  // in-place updates
  void add(uint i, Scalar val) { (*line(i, true))(i) += val; }
  void sub(uint i, Scalar val) { (*line(i, true))(i) -= val; }
  void mul(uint i, Scalar val) { (*line(i, true))(i) *= val; }
  void div(uint i, Scalar val) { (*line(i, true))(i) /= val; }

  // return cache line for i; may require write-back and fetch
  CacheLine* line(uint i, bool write) const
  {
    CacheLine* p = 0;
    uint b = block(i);
    typename Cache<CacheLine>::Tag t = cache.access(p, b + 1, write);
    uint c = t.index() - 1;
    if (c != b) {
      if (t.dirty()) {
        // write back dirty cache line
        stream.seek(c * blksize);
        codec.encode(p->a, 1, dims ? dims[c] : 0);
        stream.flush();
      }
      // fetch cache line
      stream.seek(b * blksize);
      codec.decode(p->a, 1);
    }
    return p;
  }

  // allocate memory for compressed data
  void alloc(bool clear = true)
  {
    bytes = m * blksize;
    reallocate(data, bytes, 0x100u);
    if (clear)
      std::fill(data, data + bytes, 0);
    stream.open(data, bytes);
    cache.clear();
  }

  // free memory associated with compressed data
  void free()
  {
    n = 0;
    m = 0;
    bytes = 0;
    deallocate(data);
    data = 0;
    deallocate(dims);
    dims = 0;
  }

  // block index for i
  static uint block(uint i) { return i / 4; }

  // compressed block size in bytes for given rate
  static size_t block_size(double rate) { return (lrint(4 * rate) + CHAR_BIT - 1) / CHAR_BIT; }

  // number of cache lines corresponding to size (or suggested size if zero)
  static uint lines(size_t size, uint n)
  {
    n = uint((size ? size : 8 * sizeof(Scalar)) / sizeof(CacheLine));
    return std::max(n, 1u);
  }

  uint n; // array dimensions
  uint m; // array dimensions in number of 4-sample blocks
  size_t blksize; // byte size of single compressed block
  size_t bytes; // total bytes of compressed data
  mutable uchar* data; // pointer to compressed data
  mutable MemoryBitStream stream; // bit stream for compressed data
  mutable Codec1<MemoryBitStream, Scalar> codec; // compression codec
  mutable Cache<CacheLine> cache; // cache of decompressed blocks
  uchar* dims; // precomputed block dimensions (or null if uniform)
};

typedef Array1<float> Array1f;
typedef Array1<double> Array1d;

}

#endif
