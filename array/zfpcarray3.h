#ifndef ZFP_CARRAY3_H
#define ZFP_CARRAY3_H

#include <cstddef>
#include <iterator>
#include <cstring>
#include "zfpcarray.h"
#include "zfpcodec.h"
#include "zfp/cache.h"

namespace zfp {

// compressed 3D array of scalars
template < typename Scalar, class Codec = zfp::codec<Scalar> >
class const_array3 : public const_array {
public:
#if 0
  // forward declarations
  class reference;
  class pointer;
  class iterator;
  class view;
  #include "zfp/reference3.h"
  #include "zfp/pointer3.h"
  #include "zfp/iterator3.h"
  #include "zfp/view3.h"
#endif

  // default constructor
  const_array3() : const_array(3, Codec::type) {}

  // constructor of nx * ny * nz array using rate bits per value, at least
  // csize bytes of cache, and optionally initialized from flat array p
  const_array3(uint nx, uint ny, uint nz, double rate, const Scalar* p = 0, size_t csize = 0) :
    const_array(3, Codec::type),
    cache(lines(csize, nx, ny, nz)),
    block_index(blocks)
  {
    set_rate(rate);
    resize(nx, ny, nz, p == 0);
    if (p)
      set(p);
  }

#if 0
  // constructor, from previously-serialized compressed array
  array3(const zfp::array::header& h, const uchar* buffer = 0, size_t buffer_size_bytes = 0) :
    array(3, Codec::type, h, buffer_size_bytes)
  {
    resize(nx, ny, nz, false);
    if (buffer)
      memcpy(data, buffer, bytes);
  }

  // copy constructor--performs a deep copy
  array3(const array3& a) :
    array()
  {
    deep_copy(a);
  }

  // construction from view--perform deep copy of (sub)array
  template <class View>
  array3(const View& v) :
    array(3, Codec::type),
    cache(lines(0, v.size_x(), v.size_y(), v.size_z()))
  {
    set_rate(v.rate());
    resize(v.size_x(), v.size_y(), v.size_z(), true);
    // initialize array in its preferred order
    for (iterator it = begin(); it != end(); ++it)
      *it = v(it.i(), it.j(), it.k());
  }
#endif

  // virtual destructor
  virtual ~const_array3() {}

#if 0
  // assignment operator--performs a deep copy
  array3& operator=(const array3& a)
  {
    if (this != &a)
      deep_copy(a);
    return *this;
  }
#endif

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
      bx = (nx + 3) / 4;
      by = (ny + 3) / 4;
      bz = (nz + 3) / 4;
      blocks = bx * by * bz;
      alloc(clear);

      // precompute block dimensions
      zfp::deallocate(shape);
      if ((nx | ny | nz) & 3u) {
        shape = (uchar*)zfp::allocate(blocks);
        uchar* p = shape;
        for (uint k = 0; k < bz; k++)
          for (uint j = 0; j < by; j++)
            for (uint i = 0; i < bx; i++)
              *p++ = (i == bx - 1 ? -nx & 3u : 0) + 4 * ((j == by - 1 ? -ny & 3u : 0) + 4 * (k == bz - 1 ? -nz & 3u : 0));
      }
      else
        shape = 0;

      // reset block index
      block_index.resize(blocks);
    }
  }

  // cache size in number of bytes
  size_t cache_size() const { return cache.size() * sizeof(CacheLine); }

  // set minimum cache size in bytes (array dimensions must be known)
  void set_cache_size(size_t csize)
  {
    cache.resize(lines(csize, nx, ny, nz));
  }

  // empty cache without compressing modified cached blocks
  void clear_cache() const { cache.clear(); }

  // decompress array and store at p
  void get(Scalar* p) const
  {
    uint b = 0;
    for (uint k = 0; k < bz; k++, p += 4 * nx * (ny - by))
      for (uint j = 0; j < by; j++, p += 4 * (nx - bx))
        for (uint i = 0; i < bx; i++, p += 4, b++) {
          const CacheLine* line = cache.lookup(b + 1);
          if (line)
            line->get(p, 1, nx, nx * ny, shape ? shape[b] : 0);
          else
            decode(b, p, 1, nx, nx * ny);
        }
  }

  // initialize array by copying and compressing data stored at p
  void set(const Scalar* p)
  {
    stream_rewind(zfp->stream);
    block_index.clear();
    uint b = 0;
    for (uint k = 0; k < bz; k++, p += 4 * nx * (ny - by))
      for (uint j = 0; j < by; j++, p += 4 * (nx - bx))
        for (uint i = 0; i < bx; i++, p += 4, b++) {
          size_t size = encode(b, p, 1, nx, nx * ny);
#if DEBUG
    fprintf(stderr, "size=%zu\n", size);
#endif
          block_index.push(size);
        }
    // flush final block
    block_index.flush();
//for (uint i = 0; i < blocks; i++)
//printf("block=%i offset=%lu\n", i, block_index(i));
//exit(0);
    stream_flush(zfp->stream);
    bytes = stream_size(zfp->stream);
#warning "should reallocate memory here"
    cache.clear();
  }

  // (i, j, k) accessors
  Scalar operator()(uint i, uint j, uint k) const { return get(i, j, k); }

  // flat index corresponding to (i, j, k)
  uint index(uint i, uint j, uint k) const { return i + nx * (j + ny * k); }

  // flat index accessors
  Scalar operator[](uint index) const
  {
    uint i, j, k;
    ijk(i, j, k, index);
    return get(i, j, k);
  }

#if 0
  // sequential iterators
  iterator begin() { return iterator(this, 0, 0, 0); }
  iterator end() { return iterator(this, 0, 0, nz); }
#endif

protected:
  // cache line representing one block of decompressed values
  class CacheLine {
  public:
    Scalar operator()(uint i, uint j, uint k) const { return a[index(i, j, k)]; }
    const Scalar* data() const { return a; }
    Scalar* data() { return a; }
    // copy cache line
    void get(Scalar* p, int sx, int sy, int sz) const
    {
      const Scalar* q = a;
      for (uint z = 0; z < 4; z++, p += sz - 4 * sy)
        for (uint y = 0; y < 4; y++, p += sy - 4 * sx)
          for (uint x = 0; x < 4; x++, p += sx, q++)
            *p = *q;
    }
    void get(Scalar* p, int sx, int sy, int sz, uint shape) const
    {
      if (!shape)
        get(p, sx, sy, sz);
      else {
        // determine block dimensions
        uint nx = 4 - (shape & 3u); shape >>= 2;
        uint ny = 4 - (shape & 3u); shape >>= 2;
        uint nz = 4 - (shape & 3u); shape >>= 2;
        const Scalar* q = a;
        for (uint z = 0; z < nz; z++, p += sz - (ptrdiff_t)ny * sy, q += 16 - 4 * ny)
          for (uint y = 0; y < ny; y++, p += sy - (ptrdiff_t)nx * sx, q += 4 - nx)
            for (uint x = 0; x < nx; x++, p += sx, q++)
              *p = *q;
      }
    }
  protected:
    static uint index(uint i, uint j, uint k) { return (i & 3u) + 4 * ((j & 3u) + 4 * (k & 3u)); }
    Scalar a[64];
  };

#if 0
  // uncompressed index
  class Index { // templetize?
  public:
    // constructor for given nbumber of blocks
    Index(uint blocks) :
      data(0),
      block(0) 
    {
      resize(blocks);
    }

    // reset index
    void clear() { block = 0; }

    void resize(uint blocks)
    {
      delete[] data;
      data = new uint64[blocks + 1];
      clear();
    }

    // push block bit size
    void push(size_t size)
    {
      if (block)
        data[block + 1] = data[block] + size;
      else {
        data[0] = 0;
        data[1] = size;
      }
      block++;
    }

    // flush any buffered data
    void flush() {}

    // bit offset of given block id
    size_t operator()(uint id) const { return data[id]; }

  protected:
    uint64* data;
    uint block;
  };
#else
  // compressed index
  class Index { // templetize?
  public:
    // constructor for given nbumber of blocks
    Index(uint blocks) :
      data(0),
      block(0) 
    {
      resize(blocks);
    }

    // reset index
    void clear()
    {
      block = 0;
      ptr = 0;
    }

    void resize(uint blocks)
    {
      delete[] data;
      data = new uint64[2 * ((blocks + 7) / 8)];
      clear();
    }

    // push block bit size
    void push(size_t size)
    {
      uint chunk = block / 8;
      buffer[block & 0x7u] = size;
      block++;
      if (!(block & 0x7u)) {
        // store all but low 8-bits of offset
        uint64 hi = (ptr >> 8) << 28;
        uint64 lo = (ptr & UINT64C(0xff)) << 56;
        for (uint k = 0; k < 7; k++) {
          // partition block size into 4 high and 8 low bits
          hi += (buffer[k] >> 8) << (4 * (6 - k));
          lo += (buffer[k] & 0xffu) << (8 * (6 - k));
          ptr += buffer[k];
        }
//fprintf(stderr, "hi=%016lx lo=%016lx\n", hi, lo);
        ptr += buffer[7];
        data[2 * chunk + 0] = hi;
        data[2 * chunk + 1] = lo;
      }
    }

    // flush any buffered data
    void flush()
    {
      while (block & 0x7u)
        push(0);
    }

    // bit offset of given block id
    size_t operator()(uint id) const
    {
      uint chunk = id / 8;
      uint which = id % 8;
//fprintf(stderr, "offset(%u) ", id);
      return offset(data[2 * chunk + 0], data[2 * chunk + 1], which);
    }

  protected:
    // kth offset in chunk, 0 <= k <= 7
    static uint64 offset(uint64 h, uint64 l, uint k)
    {
//fprintf(stderr, "(%016lx %016lx %u) ", h, l, k);
      uint64 base = h >> 32;
      h &= UINT64C(0xffffffff);
      uint64 hi = sum4(h >> (4 * (7 - k)));
      uint64 lo = sum8(l >> (8 * (7 - k)));
//fprintf(stderr, "(%016lx %016lx %016lx)\n", base, hi, lo);
      return (base << 12) + (hi << 8) + lo;
    }

    // sum of eight packed 4-bit numbers
    static uint64 sum4(uint64 x)
    {
      uint64 y = x & UINT64C(0xf0f0f0f0);
      x -= y;
      x += y >> 4;
      x += x >> 16;
      x += x >> 8;
      return x & UINT64C(0xff);
    }
  
    // sum of eight packed 8-bit numbers
    static uint64 sum8(uint64 x)
    {
      uint64 y = x & UINT64C(0xff00ff00ff00ff00);
      x -= y;
      x += y >> 8;
      x += x >> 32;
      x += x >> 16;
      return x & UINT64C(0xffff);
    }

    uint64* data;
    uint block;
    uint64 ptr;
    size_t buffer[8];
  };
#endif

#if 0
  // perform a deep copy
  void deep_copy(const array3& a)
  {
    // copy base class members
    array::deep_copy(a);
    // copy cache
    cache = a.cache;
  }
#endif

  // inspector
  Scalar get(uint i, uint j, uint k) const
  {
    const CacheLine* p = line(i, j, k);
    return (*p)(i, j, k);
  }

  // return cache line for (i, j, k); may require write-back and fetch
  const CacheLine* line(uint i, uint j, uint k) const
  {
    CacheLine* p = 0;
    uint b = block(i, j, k);
    typename zfp::Cache<CacheLine>::Tag t = cache.access(p, b + 1, false);
    uint c = t.index() - 1;
    if (c != b) {
/*
      // write back occupied cache line if it is dirty
      if (t.dirty())
        encode(c, p->data());
*/
      // fetch cache line
      decode(b, p->data());
    }
    return p;
  }

  // encode block with given index
  size_t encode(uint index, const Scalar* block) const
  {
    return Codec::encode_block_3(zfp, block, shape ? shape[index] : 0);
  }

  // encode block with given index from strided array
  size_t encode(uint index, const Scalar* p, int sx, int sy, int sz) const
  {
    return Codec::encode_block_strided_3(zfp, p, shape ? shape[index] : 0, sx, sy, sz);
  }

  // decode block with given index
  void decode(uint index, Scalar* block) const
  {
    stream_rseek(zfp->stream, block_index(index));
    Codec::decode_block_3(zfp, block, shape ? shape[index] : 0);
  }

  // decode block with given index to strided array
  void decode(uint index, Scalar* p, int sx, int sy, int sz) const
  {
    stream_rseek(zfp->stream, block_index(index));
    Codec::decode_block_strided_3(zfp, p, shape ? shape[index] : 0, sx, sy, sz);
  }

  // block index for (i, j, k)
  uint block(uint i, uint j, uint k) const { return (i / 4) + bx * ((j / 4) + by * (k / 4)); }

  // convert flat index to (i, j, k)
  void ijk(uint& i, uint& j, uint& k, uint index) const
  {
    i = index % nx;
    index /= nx;
    j = index % ny;
    index /= ny;
    k = index;
  }

  // number of cache lines corresponding to size (or suggested size if zero)
  static uint lines(size_t size, uint nx, uint ny, uint nz)
  {
    uint n = size ? (size + sizeof(CacheLine) - 1) / sizeof(CacheLine) : const_array::lines(size_t((nx + 3) / 4) * size_t((ny + 3) / 4) * size_t((nz + 3) / 4));
    return std::max(n, 1u);
  }

  mutable zfp::Cache<CacheLine> cache; // cache of decompressed blocks
  Index block_index; // block index
};

typedef const_array3<float> const_array3f;
typedef const_array3<double> const_array3d;

}

#endif
