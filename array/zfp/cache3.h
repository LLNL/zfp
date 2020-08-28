#ifndef ZFP_CACHE3_H
#define ZFP_CACHE3_H

#include "cache.h"
#include "store3.h"

namespace zfp {

template <typename Scalar, class Codec>
class BlockCache3 {
public:
  // constructor of cache of given size
  BlockCache3(BlockStore3<Scalar, Codec>& store, size_t bytes = 0) :
    cache((uint)((bytes + sizeof(CacheLine) - 1) / sizeof(CacheLine))),
    store(store),
    codec(0)
  {
    alloc();
  }

  // destructor
  ~BlockCache3() { free(); }

  // cache size in number of bytes
  size_t size() const { return cache.size() * sizeof(CacheLine); }

  // set minimum cache size in bytes (inferred from blocks if zero)
  void resize(size_t bytes)
  {
    flush();
    cache.resize(lines(bytes, store.blocks()));
  }

  // rate in bits per value
  double rate() const { return store.rate(); }

  // set rate in bits per value
  double set_rate(double rate)
  {
    cache.clear();
    free();
    rate = store.set_rate(rate);
    alloc();
    return rate;
  }

  // empty cache without compressing modified cached blocks
  void clear() const { cache.clear(); }

  // flush cache by compressing all modified cached blocks
  void flush() const
  {
    for (typename zfp::Cache<CacheLine>::const_iterator p = cache.first(); p; p++) {
      if (p->tag.dirty()) {
        size_t block_index = p->tag.index() - 1;
        store.encode(codec, block_index, p->line->data());
      }
      cache.flush(p->line);
    }
  }

  // perform a deep copy
  void deep_copy(const BlockCache3& c)
  {
    free();
    cache = c.cache;
    alloc();
  }

  // inspector
  Scalar get(size_t i, size_t j, size_t k) const
  {
    const CacheLine* p = line(i, j, k, false);
    return (*p)(i, j, k);
  }

  // mutator
  void set(size_t i, size_t j, size_t k, Scalar val)
  {
    CacheLine* p = line(i, j, k, true);
    (*p)(i, j, k) = val;
  }

  // reference to cached element
  Scalar& ref(size_t i, size_t j, size_t k)
  {
    CacheLine* p = line(i, j, k, true);
    return (*p)(i, j, k);
  }

  // copy block from cache, if cached, or fetch from persistent storage without caching
  void get_block(size_t block_index, Scalar* p, ptrdiff_t sx, ptrdiff_t sy, ptrdiff_t sz) const
  {
    const CacheLine* line = cache.lookup((uint)block_index + 1, false);
    if (line)
      line->get(p, sx, sy, sz, store.block_shape(block_index));
    else
      store.decode(codec, block_index, p, sx, sy, sz);
  }

  // copy vlock to cache, if cached, or store to persistent storage without caching
  void put_block(size_t block_index, const Scalar* p, ptrdiff_t sx, ptrdiff_t sy, ptrdiff_t sz) const
  {
    CacheLine* line = cache.lookup((uint)block_index + 1, true);
    if (line)
      line->put(p, sx, sy, sz, store.block_shape(block_index));
    else
      store.encode(codec, block_index, p, sx, sy, sz);
  }

protected:
  // allocate codec
  void alloc()
  {
    codec = new Codec(store.compressed_data(), store.compressed_size());
    codec->set_rate(store.rate());
  }

  // free allocated data
  void free()
  {
    if (codec) {
      delete codec;
      codec = 0;
    }
  }

  // cache line representing one block of decompressed values
  class CacheLine {
  public:
    // accessors
    Scalar operator()(size_t i, size_t j, size_t k) const { return a[index(i, j, k)]; }
    Scalar& operator()(size_t i, size_t j, size_t k) { return a[index(i, j, k)]; }

    // pointer to decompressed block data
    const Scalar* data() const { return a; }
    Scalar* data() { return a; }

    // copy whole block from cache line
    void get(Scalar* p, ptrdiff_t sx, ptrdiff_t sy, ptrdiff_t sz) const
    {
      const Scalar* q = a;
      for (uint z = 0; z < 4; z++, p += sz - 4 * sy)
        for (uint y = 0; y < 4; y++, p += sy - 4 * sx)
          for (uint x = 0; x < 4; x++, p += sx, q++)
            *p = *q;
    }

    // copy partial block from cache line
    void get(Scalar* p, ptrdiff_t sx, ptrdiff_t sy, ptrdiff_t sz, uint shape) const
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

    // copy whole block to cache line
    void put(const Scalar* p, ptrdiff_t sx, ptrdiff_t sy, ptrdiff_t sz)
    {
      Scalar* q = a;
      for (uint z = 0; z < 4; z++, p += sz - 4 * sy)
        for (uint y = 0; y < 4; y++, p += sy - 4 * sx)
          for (uint x = 0; x < 4; x++, p += sx, q++)
            *q = *p;
    }

    // copy partial block to cache line
    void put(const Scalar* p, ptrdiff_t sx, ptrdiff_t sy, ptrdiff_t sz, uint shape)
    {
      if (!shape)
        put(p, sx, sy, sz);
      else {
        // determine block dimensions
        uint nx = 4 - (shape & 3u); shape >>= 2;
        uint ny = 4 - (shape & 3u); shape >>= 2;
        uint nz = 4 - (shape & 3u); shape >>= 2;
        Scalar* q = a;
        for (uint z = 0; z < nz; z++, p += sz - (ptrdiff_t)ny * sy, q += 16 - 4 * ny)
          for (uint y = 0; y < ny; y++, p += sy - (ptrdiff_t)nx * sx, q += 4 - nx)
            for (uint x = 0; x < nx; x++, p += sx, q++)
              *q = *p;
      }
    }

  protected:
    static size_t index(size_t i, size_t j, size_t k) { return (i & 3u) + 4 * ((j & 3u) + 4 * (k & 3u)); }
    Scalar a[4 * 4 * 4];
  };

  // return cache line for (i, j, k); may require write-back and fetch
  CacheLine* line(size_t i, size_t j, size_t k, bool write) const
  {
    CacheLine* p = 0;
    size_t block_index = store.block_index(i, j, k);
    typename zfp::Cache<CacheLine>::Tag tag = cache.access(p, (uint)block_index + 1, write);
    size_t stored_block_index = tag.index() - 1;
    if (stored_block_index != block_index) {
      // write back occupied cache line if it is dirty
      if (tag.dirty())
        store.encode(codec, stored_block_index, p->data());
      // fetch cache line
      store.decode(codec, block_index, p->data());
    }
    return p;
  }

  // default number of cache lines for array with given number of blocks
  static uint lines(size_t blocks)
  {
    // compute m = O(sqrt(n))
    size_t m;
    for (m = 1; m * m < blocks; m *= 2);
    return static_cast<uint>(m);
  }

  // number of cache lines corresponding to size (or suggested size if zero)
  static uint lines(size_t bytes, size_t blocks)
  {
    uint n = bytes ? static_cast<uint>((bytes + sizeof(CacheLine) - 1) / sizeof(CacheLine)) : lines(blocks);
    return std::max(n, 1u);
  }

  mutable Cache<CacheLine> cache;    // cache of decompressed blocks
  BlockStore3<Scalar, Codec>& store; // store backed by cache
  Codec* codec;                      // compression codec
};

}

#endif
