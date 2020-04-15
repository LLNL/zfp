#ifndef ZFP_CACHE2_H
#define ZFP_CACHE2_H

#include "cache.h"
#include "store2.h"

namespace zfp {

template <typename Scalar, class Codec>
class BlockCache2 {
public:
  // constructor of cache of given size
  BlockCache2(BlockStore2<Scalar, Codec>& store, size_t bytes = 0) : cache(bytes), store(store) {}

  // cache size in number of bytes
  size_t size() const { return cache.size() * sizeof(CacheLine); }

  // set minimum cache size in bytes (inferred from blocks if zero)
  void resize(size_t bytes)
  {
    flush();
    cache.resize(lines(bytes, store.blocks()));
  }

  // empty cache without compressing modified cached blocks
  void clear() const { cache.clear(); }

  // flush cache by compressing all modified cached blocks
  void flush() const
  {
    for (typename zfp::Cache<CacheLine>::const_iterator p = cache.first(); p; p++) {
      if (p->tag.dirty()) {
        uint block_index = p->tag.index() - 1;
        store.encode(block_index, p->line->data());
      }
      cache.flush(p->line);
    }
  }

  // perform a deep copy
  void deep_copy(const BlockCache2& c) { cache = c.cache; }

  // inspector
  Scalar get(uint i, uint j) const
  {
    const CacheLine* p = line(i, j, false);
    return (*p)(i, j);
  }

  // mutator
  void set(uint i, uint j, Scalar val)
  {
    CacheLine* p = line(i, j, true);
    (*p)(i, j) = val;
  }

  // reference to cached element
  Scalar& ref(uint i, uint j)
  {
    CacheLine* p = line(i, j, true);
    return (*p)(i, j);
  }

  // fetch block without caching
  void get_block(uint block_index, Scalar* p, ptrdiff_t sx, ptrdiff_t sy) const
  {
    const CacheLine* line = cache.lookup(block_index);
    if (line)
      line->get(p, sx, sy, store.block_shape(block_index));
    else
      store.decode(block_index, p, sx, sy);
  }

protected:
  // cache line representing one block of decompressed values
  class CacheLine {
  public:
    // accessors
    Scalar operator()(uint i, uint j) const { return a[index(i, j)]; }
    Scalar& operator()(uint i, uint j) { return a[index(i, j)]; }

    // pointer to decompressed block data
    const Scalar* data() const { return a; }
    Scalar* data() { return a; }

    // copy cache line
    void get(Scalar* p, ptrdiff_t sx, ptrdiff_t sy) const
    {
      const Scalar* q = a;
      for (uint y = 0; y < 4; y++, p += sy - 4 * sx)
        for (uint x = 0; x < 4; x++, p += sx, q++)
          *p = *q;
    }

    // copy cache line to strided storage
    void get(Scalar* p, ptrdiff_t sx, ptrdiff_t sy, uint shape) const
    {
      if (!shape)
        get(p, sx, sy);
      else {
        // determine block dimensions
        uint nx = 4 - (shape & 3u); shape >>= 2;
        uint ny = 4 - (shape & 3u); shape >>= 2;
        const Scalar* q = a;
        for (uint y = 0; y < ny; y++, p += sy - (ptrdiff_t)nx * sx, q += 4 - nx)
          for (uint x = 0; x < nx; x++, p += sx, q++)
            *p = *q;
      }
    }

  protected:
    static uint index(uint i, uint j) { return (i & 3u) + 4 * (j & 3u); }
    Scalar a[4 * 4];
  };

  // return cache line for (i, j); may require write-back and fetch
  CacheLine* line(uint i, uint j, bool write) const
  {
    CacheLine* p = 0;
    uint block_index = store.block_index(i, j);
    typename zfp::Cache<CacheLine>::Tag tag = cache.access(p, block_index + 1, write);
    uint stored_block_index = tag.index() - 1;
    if (stored_block_index != block_index) {
      // write back occupied cache line if it is dirty
      if (tag.dirty())
        store.encode(stored_block_index, p->data());
      // fetch cache line
      store.decode(block_index, p->data());
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
    uint n = bytes ? uint((bytes + sizeof(CacheLine) - 1) / sizeof(CacheLine)) : lines(blocks);
    return std::max(n, 1u);
  }

  mutable Cache<CacheLine> cache;    // cache of decompressed blocks
  BlockStore2<Scalar, Codec>& store; // store backed by cache
};

}

#endif
