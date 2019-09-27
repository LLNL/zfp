#ifndef ZFP_VARRAY3_H
#define ZFP_VARRAY3_H

#include <cstddef>
#include <iterator>
#include <cstring>
#include "zfpvarray.h"
#include "zfpcodec.h"
#include "zfp/tile3.h"
#include "zfp/cache.h"

namespace zfp {

// compressed 3D array of scalars
template < typename Scalar, class Codec = zfp::codec<Scalar> >
class varray3 : public varray {
public:
  // forward declarations
  class reference;
  class pointer;
  class iterator;
  #include "zfp/vreference3.h"
  #include "zfp/vpointer3.h"
  #include "zfp/viterator3.h"

  typedef varray::storage storage;

  // default constructor
  varray3() : varray(3, Codec::type), tile(0) {}

  // constructor of nx * ny * nz array using prec bits of precision, at least
  // csize bytes of cache, and optionally initialized from flat array p
  varray3(uint nx, uint ny, uint nz, uint prec, const Scalar* p = 0, size_t csize = 0) :
    varray(3, Codec::type),
    cache(lines(csize, nx, ny, nz)),
    tile(0)
  {
    set_precision(prec);
    resize(nx, ny, nz);
    if (p)
      set(p);
  }

  // constructor of nx * ny * nz array using error tolerance tol, at least
  // csize bytes of cache, and optionally initialized from flat array p
  varray3(uint nx, uint ny, uint nz, double tol, const Scalar* p = 0, size_t csize = 0) :
    varray(3, Codec::type),
    cache(lines(csize, nx, ny, nz)),
    tile(0)
  {
    set_accuracy(tol);
    resize(nx, ny, nz);
    if (p)
      set(p);
  }

  // copy constructor--performs a deep copy
  varray3(const varray3& a) :
    varray()
  {
    deep_copy(a);
  }

  // virtual destructor
  virtual ~varray3()
  {
    free();
  }

  // assignment operator--performs a deep copy
  varray3& operator=(const varray3& a)
  {
    if (this != &a)
      deep_copy(a);
    return *this;
  }

  // total number of elements in array
  size_t size() const { return size_t(nx) * size_t(ny) * size_t(nz); }

  // array dimensions
  uint size_x() const { return nx; }
  uint size_y() const { return ny; }
  uint size_z() const { return nz; }

  // resize the array (all previously stored data will be lost)
  void resize(uint nx, uint ny, uint nz)
  {
    if (nx == 0 || ny == 0 || nz == 0)
      free();
    else {
      // precompute block dimensions
      this->nx = nx;
      this->ny = ny;
      this->nz = nz;
      bx = (nx + 3) / 4;
      by = (ny + 3) / 4;
      bz = (nz + 3) / 4;
      blocks = bx * by * bz;
      tx = (bx + Tile3<Scalar>::bx - 1) / Tile3<Scalar>::bx;
      ty = (by + Tile3<Scalar>::by - 1) / Tile3<Scalar>::by;
      tz = (bz + Tile3<Scalar>::bz - 1) / Tile3<Scalar>::bz;
      tiles = tx * ty * tz;
      alloc();
    }
  }

  // cache size in number of bytes
  size_t cache_size() const { return cache.size() * sizeof(CacheLine); }

  // set minimum cache size in bytes (array dimensions must be known)
  void set_cache_size(size_t csize)
  {
    flush_cache();
    cache.resize(lines(csize, nx, ny, nz));
  }

  // empty cache without compressing modified cached blocks
  void clear_cache() const { cache.clear(); }

  // flush cache by compressing all modified cached blocks
  void flush_cache() const
  {
    for (typename zfp::Cache<CacheLine>::const_iterator p = cache.first(); p; p++) {
      if (p->tag.index() && (true || p->tag.dirty())) {
        uint b = p->tag.index() - 1;
        encode(b, p->line->data());
      }
      cache.flush(p->line);
    }
  }

  virtual size_t storage_size(uint mask = ZFP_DATA_ALL) const
  {
    size_t size = varray::storage_size(mask);
    if (mask & ZFP_DATA_META)
      size += sizeof(varray3) - sizeof(varray);
    if (mask & ZFP_DATA_CACHE)
      size += cache.size() * (sizeof(CacheLine) + sizeof(typename Cache<CacheLine>::Tag));
    for (uint t = 0; t < tiles; t++)
      size += tile[t]->size(mask);
    return size;
  }

  storage element_storage(uint i, uint j, uint k) const
  {
    uint index = block(i, j, k);
    uint t = tile_id(index);
    uint b = block_id(index);
    return tile[t]->block_storage(zfp, b, shape(i, j, k));
  }

  // decompress array and store at p
  void get(Scalar* p) const
  {
    uint index = 0;
    for (uint z = 0; z < nz; z += 4, p += 4 * (ny - by))
      for (uint y = 0; y < ny; y += 4, p += 4 * (nx - bx))
        for (uint x = 0; x < nx; x += 4, p += 4, index++) {
          const CacheLine* line = cache.lookup(index + 1);
          if (line)
            line->get(p, 1, nx, nx * ny, shape(x, y, z));
          else {
            Scalar block[4 * 4 * 4];
            decode(index, block, false);
            uint sx, sy, sz;
            shape(sx, sy, sz, x, y, z);
            for (uint k = 0; k < sz; k++)
              for (uint j = 0; j < sy; j++)
                for (uint i = 0; i < sx; i++)
                  p[i + nx * (j + ny * k)] = block[i + 4 * (j + 4 * k)];
          }
        }
  }

  // initialize array by copying and compressing data stored at p
  void set(const Scalar* p)
  {
    uint index = 0;
    for (uint z = 0; z < nz; z += 4, p += 4 * (ny - by))
      for (uint y = 0; y < ny; y += 4, p += 4 * (nx - bx))
        for (uint x = 0; x < nx; x += 4, p += 4, index++) {
          Scalar block[4 * 4 * 4];
          uint sx, sy, sz;
          shape(sx, sy, sz, x, y, z);
          for (uint k = 0; k < sz; k++)
            for (uint j = 0; j < sy; j++)
              for (uint i = 0; i < sx; i++)
                block[i + 4 * (j + 4 * k)] = p[i + nx * (j + ny * k)];
          encode(index, block);
        }
    cache.clear();
  }

  // (i, j, k) accessors
  Scalar operator()(uint i, uint j, uint k) const { return get(i, j, k); }
  reference operator()(uint i, uint j, uint k) { return reference(this, i, j, k); }

  // flat index accessors
  Scalar operator[](uint index) const
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

  // sequential iterators
  iterator begin() { return iterator(this, 0, 0, 0); }
  iterator end() { return iterator(this, 0, 0, nz); }

protected:
  // cache line representing one block of decompressed values
  class CacheLine {
  public:
    Scalar operator()(uint i, uint j, uint k) const { return a[index(i, j, k)]; }
    Scalar& operator()(uint i, uint j, uint k) { return a[index(i, j, k)]; }
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

  // allocate memory for compressed data
  void alloc()
  {
    varray::alloc();
    if (tile) {
      for (uint t = 0; t < tiles; t++)
        delete tile[t];
      delete[] tile;
    }
    tile = new Tile3<Scalar>*[tiles];
    for (uint t = 0; t < tiles; t++)
      tile[t] = new Tile3<Scalar>(minbits);
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
  void deep_copy(const varray3& a)
  {
    // copy base class members
    varray::deep_copy(a);
    // copy cache
    cache = a.cache;
  }

  // inspector
  Scalar get(uint i, uint j, uint k) const
  {
    const CacheLine* p = line(i, j, k, false);
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
    typename zfp::Cache<CacheLine>::Tag t = cache.access(p, b + 1, write);
    uint c = t.index() - 1;
    if (c != b) {
      // write back occupied cache line (done even if not dirty)
      if (t.index())
        encode(c, p->data());
      // fetch cache line
      decode(b, p->data());
    }
    return p;
  }

  // encode block with given index
  void encode(uint index, const Scalar* block) const
  {
    uint t = tile_id(index);
    uint b = block_id(index);
    tile[t]->compress(zfp, block, b, shape(index));
  }

  // decode block with given index
  void decode(uint index, Scalar* block, bool cache = true) const
  {
    uint t = tile_id(index);
    uint b = block_id(index);
    tile[t]->decompress(zfp, block, b, shape(index), cache);
  }

  // block index for (i, j, k)
  uint block(uint i, uint j, uint k) const { return (i / 4) + bx * ((j / 4) + by * (k / 4)); }

  // tile id associated with given block index
  uint tile_id(uint block) const
  {
    uint x = block % bx; block /= bx;
    uint y = block % by; block /= by;
    uint z = block;
    uint xx = x / Tile3<Scalar>::bx;
    uint yy = y / Tile3<Scalar>::by;
    uint zz = z / Tile3<Scalar>::bz;
    return xx + tx * (yy + ty * zz);
  }

  // tile-local block id associated with given global block index
  uint block_id(uint block) const
  {
    uint x = block % bx; block /= bx;
    uint y = block % by; block /= by;
    uint z = block;
    uint xx = x % Tile3<Scalar>::bx;
    uint yy = y % Tile3<Scalar>::by;
    uint zz = z % Tile3<Scalar>::bz;
    return xx + Tile3<Scalar>::bx * (yy + Tile3<Scalar>::by * zz);
  }

  // shape (sx, sy, sz) of block containing array index (i, j, k)
  void shape(uint& sx, uint& sy, uint& sz, uint i, uint j, uint k) const
  {
    sx = -nx & (((i ^ nx) - 4) >> (CHAR_BIT * sizeof(uint) - 2));
    sy = -ny & (((j ^ ny) - 4) >> (CHAR_BIT * sizeof(uint) - 2));
    sz = -nz & (((k ^ nz) - 4) >> (CHAR_BIT * sizeof(uint) - 2));
  }

  // shape of block containing array index (i, j, k)
  uint shape(uint i, uint j, uint k) const
  {
    uint sx, sy, sz;
    shape(sx, sy, sz, i, j, k);
    return sx + 4 * sy + 16 * sz;
  }

  // shape of block with given global block index
  uint shape(uint block) const
  {
    uint i = 4 * (block % bx); block /= bx;
    uint j = 4 * (block % by); block /= by;
    uint k = 4 * block;
    return shape(i, j, k);
  }

  // convert flat index to (i, j)
  void ijk(uint& i, uint& j, uint& k, uint index) const
  {
    i = index % nx; index /= nx;
    j = index % ny; index /= ny;
    k = index;
  }

  // number of cache lines corresponding to size (or suggested size if zero)
  static uint lines(size_t size, uint nx, uint ny, uint nz)
  {
    uint n = size ? uint((size + sizeof(CacheLine) - 1) / sizeof(CacheLine)) : varray::lines(size_t((nx + 3) / 4) * size_t((ny + 3) / 4) * size_t((nz + 3) / 4));
    return std::max(n, 1u);
  }

  mutable zfp::Cache<CacheLine> cache; // cache of decompressed blocks
  Tile3<Scalar>** tile;                // tiles of compressed blocks
};

typedef varray3<float> varray3f;
typedef varray3<double> varray3d;

}

#endif
