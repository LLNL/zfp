#ifndef ZFP_CACHE4_HPP
#define ZFP_CACHE4_HPP

#include "zfp/internal/array/cache.hpp"

namespace zfp {
namespace internal {

template <typename Scalar, class Store>
class BlockCache4 {
public:
  // constructor of cache of given size
  BlockCache4(Store& store, size_t bytes = 0) :
    cache(lines(bytes, store.blocks())),
    store(store)
  {}

  // byte size of cache data structure components indicated by mask
  size_t size_bytes(uint mask = ZFP_DATA_ALL) const
  {
    size_t size = 0;
    size += cache.size_bytes(mask);
    if (mask & ZFP_DATA_META)
      size += sizeof(*this);
    return size;
  }

  // cache size in number of bytes (cache line payload data only)
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
    for (typename zfp::internal::Cache<CacheLine>::const_iterator p = cache.first(); p; p++) {
      if (p->tag.dirty()) {
        size_t block_index = p->tag.index() - 1;
        store.encode(block_index, p->line->data());
      }
      cache.flush(p->line);
    }
  }

  // perform a deep copy
  void deep_copy(const BlockCache4& c) { cache = c.cache; }

  // inspector
  Scalar get(size_t i, size_t j, size_t k, size_t l) const
  {
    const CacheLine* p = line(i, j, k, l, false);
    return (*p)(i, j, k, l);
  }

  // mutator
  void set(size_t i, size_t j, size_t k, size_t l, Scalar val)
  {
    CacheLine* p = line(i, j, k, l, true);
    (*p)(i, j, k, l) = val;
  }

  // reference to cached element
  Scalar& ref(size_t i, size_t j, size_t k, size_t l)
  {
    CacheLine* p = line(i, j, k, l, true);
    return (*p)(i, j, k, l);
  }

  // read-no-allocate: copy block from cache on hit, else from store without caching
  void get_block(size_t block_index, Scalar* p, ptrdiff_t sx, ptrdiff_t sy, ptrdiff_t sz, ptrdiff_t sw) const
  {
    const CacheLine* line = cache.lookup((uint)block_index + 1, false);
    if (line)
      line->get(p, sx, sy, sz, sw, store.block_shape(block_index));
    else
      store.decode(block_index, p, sx, sy, sz, sw);
  }

  // write-no-allocate: copy block to cache on hit, else to store without caching
  void put_block(size_t block_index, const Scalar* p, ptrdiff_t sx, ptrdiff_t sy, ptrdiff_t sz, ptrdiff_t sw) const
  {
    CacheLine* line = cache.lookup((uint)block_index + 1, true);
    if (line)
      line->put(p, sx, sy, sz, sw, store.block_shape(block_index));
    else
      store.encode(block_index, p, sx, sy, sz, sw);
  }

protected:
  // cache line representing one block of decompressed values
  class CacheLine {
  public:
    // accessors
    Scalar operator()(size_t i, size_t j, size_t k, size_t l) const { return a[index(i, j, k, l)]; }
    Scalar& operator()(size_t i, size_t j, size_t k, size_t l) { return a[index(i, j, k, l)]; }

    // pointer to decompressed block data
    const Scalar* data() const { return a; }
    Scalar* data() { return a; }

    // copy whole block from cache line
    void get(Scalar* p, ptrdiff_t sx, ptrdiff_t sy, ptrdiff_t sz, ptrdiff_t sw) const
    {
      const Scalar* q = a;
      for (uint w = 0; w < 4; w++, p += sw - 4 * sz)
        for (uint z = 0; z < 4; z++, p += sz - 4 * sy)
          for (uint y = 0; y < 4; y++, p += sy - 4 * sx)
            for (uint x = 0; x < 4; x++, p += sx, q++)
              *p = *q;
    }

    // copy partial block from cache line
    void get(Scalar* p, ptrdiff_t sx, ptrdiff_t sy, ptrdiff_t sz, ptrdiff_t sw, uint shape) const
    {
      if (!shape)
        get(p, sx, sy, sz, sw);
      else {
        // determine block dimensions
        uint nx = 4 - (shape & 3u); shape >>= 2;
        uint ny = 4 - (shape & 3u); shape >>= 2;
        uint nz = 4 - (shape & 3u); shape >>= 2;
        uint nw = 4 - (shape & 3u); shape >>= 2;
        const Scalar* q = a;
        for (uint w = 0; w < nw; w++, p += sw - (ptrdiff_t)nz * sz, q += 64 - 16 * nz)
          for (uint z = 0; z < nz; z++, p += sz - (ptrdiff_t)ny * sy, q += 16 - 4 * ny)
            for (uint y = 0; y < ny; y++, p += sy - (ptrdiff_t)nx * sx, q += 4 - nx)
              for (uint x = 0; x < nx; x++, p += sx, q++)
                *p = *q;
      }
    }

    // copy whole block to cache line
    void put(const Scalar* p, ptrdiff_t sx, ptrdiff_t sy, ptrdiff_t sz, ptrdiff_t sw)
    {
      Scalar* q = a;
      for (uint w = 0; w < 4; w++, p += sw - 4 * sz)
        for (uint z = 0; z < 4; z++, p += sz - 4 * sy)
          for (uint y = 0; y < 4; y++, p += sy - 4 * sx)
            for (uint x = 0; x < 4; x++, p += sx, q++)
              *q = *p;
    }

    // copy partial block to cache line
    void put(const Scalar* p, ptrdiff_t sx, ptrdiff_t sy, ptrdiff_t sz, ptrdiff_t sw, uint shape)
    {
      if (!shape)
        put(p, sx, sy, sz, sw);
      else {
        // determine block dimensions
        uint nx = 4 - (shape & 3u); shape >>= 2;
        uint ny = 4 - (shape & 3u); shape >>= 2;
        uint nz = 4 - (shape & 3u); shape >>= 2;
        uint nw = 4 - (shape & 3u); shape >>= 2;
        Scalar* q = a;
        for (uint w = 0; w < nw; w++, p += sw - (ptrdiff_t)nz * sz, q += 64 - 16 * nz)
          for (uint z = 0; z < nz; z++, p += sz - (ptrdiff_t)ny * sy, q += 16 - 4 * ny)
            for (uint y = 0; y < ny; y++, p += sy - (ptrdiff_t)nx * sx, q += 4 - nx)
              for (uint x = 0; x < nx; x++, p += sx, q++)
                *q = *p;
      }
    }

  protected:
    static size_t index(size_t i, size_t j, size_t k, size_t l) { return (i & 3u) + 4 * ((j & 3u) + 4 * ((k & 3u) + 4 * (l & 3u))); }
    Scalar a[4 * 4 * 4 * 4];
  };

  // return cache line for (i, j, k, l); may require write-back and fetch
  CacheLine* line(size_t i, size_t j, size_t k, size_t l, bool write) const
  {
    CacheLine* p = 0;
    size_t block_index = store.block_index(i, j, k, l);
    typename zfp::internal::Cache<CacheLine>::Tag tag = cache.access(p, (uint)block_index + 1, write);
    size_t stored_block_index = tag.index() - 1;
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
    // ensure block index fits in tag
    if (blocks >> ((sizeof(uint) * CHAR_BIT) - 1))
      throw zfp::exception("zfp array too large for cache");
    uint n = bytes ? static_cast<uint>((bytes + sizeof(CacheLine) - 1) / sizeof(CacheLine)) : lines(blocks);
    return std::max(n, 1u);
  }

  mutable Cache<CacheLine> cache; // cache of decompressed blocks
  Store& store;                   // store backed by cache
};

} // internal
} // zfp

#endif
