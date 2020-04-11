#ifndef ZFP_CACHE1_H
#define ZFP_CACHE1_H

#include "zfp.h"
#include "cache.h"
#include "block1.h"

namespace zfp {

template <typename Scalar, class Codec>
class BlockCache1 {
public:
  // constructor of cache of given size
  BlockCache1(BlockStorage1<Scalar, Codec>& storage, size_t bytes = 0) : cache(bytes), storage(storage) {}

  // cache size in number of bytes
  size_t size() const { return cache.size() * sizeof(CacheLine); }

  // set minimum cache size in bytes (inferred from blocks if zero)
  void resize(size_t bytes)
  {
    flush();
    cache.resize(lines(bytes, storage.blocks()));
  }

  // empty cache without compressing modified cached blocks
  void clear() const { cache.clear(); }

  // flush cache by compressing all modified cached blocks
  void flush() const
  {
    for (typename zfp::Cache<CacheLine>::const_iterator p = cache.first(); p; p++) {
      if (p->tag.dirty()) {
        uint block_index = p->tag.index() - 1;
        storage.encode(block_index, p->line->data());
      }
      cache.flush(p->line);
    }
  }

  // perform a deep copy
  void deep_copy(const BlockCache1& c) { cache = c.cache; }

  // inspector
  Scalar get(uint i) const
  {
    const CacheLine* p = line(i, false);
    return (*p)(i);
  }

  // mutator
  void set(uint i, Scalar val)
  {
    CacheLine* p = line(i, true);
    (*p)(i) = val;
  }

  // reference to cached element
  Scalar& ref(uint i)
  {
    CacheLine* p = line(i, true);
    return (*p)(i);
  }

  // fetch block without caching
  void get_block(uint block_index, Scalar* p, ptrdiff_t sx) const
  {
    const CacheLine* line = cache.lookup(block_index);
    if (line)
      line->get(p, sx, storage.block_shape(block_index));
    else
      storage.decode(block_index, p, sx);
  }

protected:
  // cache line representing one block of decompressed values
  class CacheLine {
  public:
    // accessors
    Scalar operator()(uint i) const { return a[index(i)]; }
    Scalar& operator()(uint i) { return a[index(i)]; }

    // pointer to decompressed block data
    const Scalar* data() const { return a; }
    Scalar* data() { return a; }

    // copy cache line
    void get(Scalar* p, ptrdiff_t sx) const
    {
      const Scalar* q = a;
      for (uint x = 0; x < 4; x++, p += sx, q++)
        *p = *q;
    }

    // copy cache line to strided storage
    void get(Scalar* p, ptrdiff_t sx, uint shape) const
    {
      if (!shape)
        get(p, sx);
      else {
        // determine block dimensions
        uint nx = 4 - (shape & 3u); shape >>= 2;
        const Scalar* q = a;
        for (uint x = 0; x < nx; x++, p += sx, q++)
          *p = *q;
      }
    }

  protected:
    static uint index(uint i) { return (i & 3u); }
    Scalar a[4];
  };

  // return cache line for i; may require write-back and fetch
  CacheLine* line(uint i, bool write) const
  {
    CacheLine* p = 0;
    uint block_index = storage.block_index(i);
    typename zfp::Cache<CacheLine>::Tag tag = cache.access(p, block_index + 1, write);
    uint stored_block_index = tag.index() - 1;
    if (stored_block_index != block_index) {
      // write back occupied cache line if it is dirty
      if (tag.dirty())
        storage.encode(stored_block_index, p->data());
      // fetch cache line
      storage.decode(block_index, p->data());
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

  mutable Cache<CacheLine> cache;        // cache of decompressed blocks
  BlockStorage1<Scalar, Codec>& storage; // storage backed by cache
};

}

#endif
