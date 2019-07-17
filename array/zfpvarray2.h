#ifndef ZFP_VARRAY2_H
#define ZFP_VARRAY2_H

#include <cstddef>
#include <iterator>
#include <cstring>
#include "zfpvarray.h"
#include "zfpcodec.h"
#include "zfp/tile2.h"
#include "zfp/cache.h"

namespace zfp {

// compressed 2D array of scalars
template < typename Scalar, class Codec = zfp::codec<Scalar> >
class varray2 : public varray {
public:
  // forward declarations
  class reference;
  class pointer;
  class iterator;
//#warning "no views yet"
//  class view;
  #include "zfp/vreference2.h"
  #include "zfp/vpointer2.h"
  #include "zfp/viterator2.h"
//  #include "zfp/view2.h"

  // default constructor
  varray2() : varray(2, Codec::type) {}

  // constructor of nx * ny array using rate bits per value, at least
  // csize bytes of cache, and optionally initialized from flat array p
  varray2(uint nx, uint ny, uint prec, const Scalar* p = 0, size_t csize = 0) :
    varray(2, Codec::type),
    cache(lines(csize, nx, ny))
  {
    set_precision(prec);
    resize(nx, ny, p == 0);
    if (p)
      set(p);
  }

  // copy constructor--performs a deep copy
  varray2(const varray2& a) :
    varray()
  {
    deep_copy(a);
  }

  // virtual destructor
  virtual ~varray2() {}

  // assignment operator--performs a deep copy
  varray2& operator=(const varray2& a)
  {
    if (this != &a)
      deep_copy(a);
    return *this;
  }

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
      // precompute block dimensions
      this->nx = nx;
      this->ny = ny;
      bx = (nx + 3) / 4;
      by = (ny + 3) / 4;
      blocks = bx * by;
      tx = (bx + Tile2<Scalar>::bx - 1) / Tile2<Scalar>::bx;
      ty = (by + Tile2<Scalar>::by - 1) / Tile2<Scalar>::by;
      tiles = tx * ty;
      alloc(clear);
    }
  }

  // cache size in number of bytes
  size_t cache_size() const { return cache.size() * sizeof(CacheLine); }

  // set minimum cache size in bytes (array dimensions must be known)
  void set_cache_size(size_t csize)
  {
    flush_cache();
    cache.resize(lines(csize, nx, ny));
  }

  // empty cache without compressing modified cached blocks
  void clear_cache() const { cache.clear(); }

  // flush cache by compressing all modified cached blocks
  void flush_cache() const
  {
    for (typename zfp::Cache<CacheLine>::const_iterator p = cache.first(); p; p++) {
      if (p->tag.dirty()) {
        uint b = p->tag.index() - 1;
        encode(b, p->line->data());
      }
      cache.flush(p->line);
    }
  }

  // decompress array and store at p
  void get(Scalar* p) const
  {
#if 0
    uint b = 0;
    for (uint j = 0; j < by; j++, p += 4 * (nx - bx))
      for (uint i = 0; i < bx; i++, p += 4, b++) {
        const CacheLine* line = cache.lookup(b + 1);
        if (line)
          line->get(p, 1, nx, shape ? shape[b] : 0);
        else
          decode(b, p, 1, nx);
      }
#endif
//#warning "get not implemented"
    (void)p;
    abort();
  }

  // initialize array by copying and compressing data stored at p
  void set(const Scalar* p)
  {
#if 0
    uint b = 0;
    for (uint j = 0; j < by; j++, p += 4 * (nx - bx))
      for (uint i = 0; i < bx; i++, p += 4, b++)
        encode(b, p, 1, nx); // error: no strided encode
    cache.clear();
#endif
//#warning "set not implemented"
    (void)p;
    abort();
  }

  // (i, j) accessors
  Scalar operator()(uint i, uint j) const { return get(i, j); }
  reference operator()(uint i, uint j) { return reference(this, i, j); }

  // flat index accessors
  Scalar operator[](uint index) const
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

  // sequential iterators
  iterator begin() { return iterator(this, 0, 0); }
  iterator end() { return iterator(this, 0, ny); }

protected:
  // cache line representing one block of decompressed values
  class CacheLine {
  public:
    Scalar operator()(uint i, uint j) const { return a[index(i, j)]; }
    Scalar& operator()(uint i, uint j) { return a[index(i, j)]; }
    const Scalar* data() const { return a; }
    Scalar* data() { return a; }
    // copy cache line
    void get(Scalar* p, int sx, int sy) const
    {
      const Scalar* q = a;
      for (uint y = 0; y < 4; y++, p += sy - 4 * sx)
        for (uint x = 0; x < 4; x++, p += sx, q++)
          *p = *q;
    }
    void get(Scalar* p, int sx, int sy, uint shape) const
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
    Scalar a[16];
  };

  // allocate memory for compressed data
  void alloc(bool clear = true)
  {
//#warning "what does clear do?"
    (void)clear;
//#warning "assert"
    assert(!tile);
    tile = new Tile*[tiles];
//#warning "where does minrate come from?"
    for (uint t = 0; t < tiles; t++)
      tile[t] = new Tile2<Scalar>();
  }

  // free memory associated with compressed data
  void free()
  {
    for (uint t = 0; t < tiles; t++)
      delete tile[t];
    delete[] tile;
    tile = 0;
    varray::free();
  }

  // perform a deep copy
  void deep_copy(const varray2& a)
  {
    // copy base class members
    varray::deep_copy(a);
    // copy cache
    cache = a.cache;
  }

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
    typename zfp::Cache<CacheLine>::Tag t = cache.access(p, b + 1, write);
    uint c = t.index() - 1;
    if (c != b) {
      // write back occupied cache line if it is dirty
      if (t.dirty())
        encode(c, p->data());
      // fetch cache line
      decode(b, p->data());
    }
    return p;
  }

//#warning "avoid dynamic casts; move tile[] to derived classes?"
/*
  const Tile2<Scalar>* tile_ptr(uint t) const
  {
    return dynamic_cast<const Tile2<Scalar>*>(tile[t]);
  }
*/

  Tile2<Scalar>* tile_ptr(uint t) const
  {
    return dynamic_cast<Tile2<Scalar>*>(tile[t]);
  }

  // encode block with given index
  void encode(uint index, const Scalar* block) const
  {
    uint t = tile_id(index);
    uint b = block_id(index);
    tile_ptr(t)->compress(zfp, block, b, shape ? shape[index] : 0);
  }

  // decode block with given index
  void decode(uint index, Scalar* block) const
  {
    uint t = tile_id(index);
    uint b = block_id(index);
    tile_ptr(t)->decompress(zfp, block, b, shape ? shape[index] : 0);
  }

  // block index for (i, j)
  uint block(uint i, uint j) const { return (i / 4) + bx * (j / 4); }

  // tile id associated with given block index
  uint tile_id(uint block) const
  {
    uint x = block % bx;
    uint y = block / bx;
    uint xx = x / Tile2<Scalar>::bx;
    uint yy = y / Tile2<Scalar>::by;
    return xx + tx * yy;
  }

  // tile-local block id associated with given global block index
  uint block_id(uint block) const
  {
    uint x = block % bx;
    uint y = block / bx;
    uint xx = x & (Tile2<Scalar>::bx - 1);
    uint yy = y & (Tile2<Scalar>::by - 1);
    return xx + Tile2<Scalar>::bx * yy;
  }

  // convert flat index to (i, j)
  void ij(uint& i, uint& j, uint index) const
  {
    i = index % nx;
    index /= nx;
    j = index;
  }

  // number of cache lines corresponding to size (or suggested size if zero)
  static uint lines(size_t size, uint nx, uint ny)
  {
    uint n = size ? uint((size + sizeof(CacheLine) - 1) / sizeof(CacheLine)) : varray::lines(size_t((nx + 3) / 4) * size_t((ny + 3) / 4));
    return std::max(n, 1u);
  }

  mutable zfp::Cache<CacheLine> cache; // cache of decompressed blocks
};

typedef varray2<float> varray2f;
typedef varray2<double> varray2d;

}

#endif
