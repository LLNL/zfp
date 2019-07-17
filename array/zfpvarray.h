#ifndef ZFP_VARRAY_H
#define ZFP_VARRAY_H

#include <algorithm>
#include <climits>
#include <cstring>
#include <stdexcept>
#include <string>
#include "zfp.h"
#include "zfp/memory.h"
#include "zfp/tile.h"

namespace zfp {

// abstract base class for variable-rate compressed array of scalars
class varray {
protected:
  // default constructor
  varray() :
    dims(0), type(zfp_type_none),
    nx(0), ny(0), nz(0),
    bx(0), by(0), bz(0),
    blocks(0),
    tiles(0), tile(0),
    zfp(0),
    shape(0)
  {}

  // generic array with 'dims' dimensions and scalar type 'type'
  varray(uint dims, zfp_type type) :
    dims(dims), type(type),
    nx(0), ny(0), nz(0),
    bx(0), by(0), bz(0),
    blocks(0),
    tiles(0), tile(0),
    zfp(zfp_stream_open(0)),
    shape(0)
  {}

  // copy constructor--performs a deep copy
  varray(const varray& a) :
    tile(0),
    zfp(0),
    shape(0)
  {
    deep_copy(a);
  }

  // assignment operator--performs a deep copy
  varray& operator=(const varray& a)
  {
    deep_copy(a);
    return *this;
  }

public:
  // public virtual destructor (can delete array through base class pointer)
  virtual ~varray()
  {
    free();
    zfp_stream_close(zfp);
  }

  // rate in bits per value
  double rate() const
  {
    size_t size = 0;
    for (uint t = 0; t < tiles; t++)
      size += tile[t]->size();
    return double(size) * CHAR_BIT / (nx * ny * nz);
  }

  // set precision in uncompressed bits per value
  uint set_precision(uint precision)
  {
    return zfp_stream_set_precision(zfp, precision);
  }

  // set compression rate in bits per value
  double set_accuracy(double tolerance)
  {
    return zfp_stream_set_accuracy(zfp, tolerance);
  }

  // enable reversible (lossless) mode
  void set_reversible()
  {
    zfp_stream_set_reversible(zfp);
  }

  // empty cache without compressing modified cached blocks
  virtual void clear_cache() const = 0;

  // flush cache by compressing all modified cached blocks
  virtual void flush_cache() const = 0;

  // number of bytes of compressed data
  size_t compressed_size() const
  {
    size_t size = 0;
    for (uint t = 0; t < tiles; t++)
      size += tile[t]->size();
    return size;
  }

  // dimensionality
  uint dimensionality() const { return dims; }

  // underlying scalar type
  zfp_type scalar_type() const { return type; }

protected:
  // number of values per block
  uint block_size() const { return 1u << (2 * dims); }

  // free memory associated with compressed data
  void free()
  {
    stream_close(zfp->stream);
    zfp_stream_set_bit_stream(zfp, 0);
    nx = ny = nz = 0;
    bx = by = bz = 0;
    blocks = 0;
    tiles = 0;
  }

  // perform a deep copy
  void deep_copy(const varray& a)
  {
    // copy metadata
    dims = a.dims;
    type = a.type;
    nx = a.nx;
    ny = a.ny;
    nz = a.nz;
    bx = a.bx;
    by = a.by;
    bz = a.bz;
    blocks = a.blocks;
    tiles = a.tiles;

//#warning "deep copy not implemented"
    // copy dynamically allocated data
    abort(); // need to copy tiles
  }

  // default number of cache lines for array with n blocks
  static uint lines(size_t n)
  {
    // compute m = O(sqrt(n))
    size_t m;
    for (m = 1; m * m < n; m *= 2);
    return static_cast<uint>(m);
  }

  uint dims;        // array dimensionality (1, 2, or 3)
  zfp_type type;    // scalar type
  uint nx, ny, nz;  // array dimensions
  uint bx, by, bz;  // array dimensions in number of blocks
  uint tx, ty, tz;  // array dimensions in number of tiles
  uint blocks;      // number of blocks
  uint tiles;       // number of tiles
  zfp::Tile** tile; // pointers to tiles
  zfp_stream* zfp;  // compression parameters and buffer shared among tiles
  uchar* shape;     // precomputed block dimensions (or null if uniform)
};

}

#endif
