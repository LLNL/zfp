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
    minbits(64),
    nx(0), ny(0), nz(0),
    bx(0), by(0), bz(0),
    tx(0), ty(0), tz(0),
    blocks(0),
    tiles(0),
    zfp(0)
  {}

  // generic array with 'dims' dimensions and scalar type 'type'
  varray(uint dims, zfp_type type) :
    dims(dims), type(type),
    minbits(64),
    nx(0), ny(0), nz(0),
    bx(0), by(0), bz(0),
    tx(0), ty(0), tz(0),
    blocks(0),
    tiles(0),
    zfp(zfp_stream_open(0))
  {}

  // copy constructor--performs a deep copy
  varray(const varray& a) :
    zfp(0)
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
  typedef Tile::storage storage;

  // public virtual destructor (can delete array through base class pointer)
  virtual ~varray()
  {
    free();
    if (zfp) {
      zfp_stream_close(zfp);
      zfp = 0;
    }
  }

  // total number of elements in array
  virtual size_t size() const = 0;

  // rate in bits per value
  virtual double rate(uint mask = ZFP_DATA_PAYLOAD) const
  {
    return double(storage_size(mask)) * CHAR_BIT / size();
  }

  // set precision in uncompressed bits per value
  uint set_precision(uint precision)
  {
    precision = zfp_stream_set_precision(zfp, precision);
    zfp->maxbits = minbits << 4;
    return precision;
  }

  // set compression rate in bits per value
  double set_accuracy(double tolerance)
  {
    tolerance = zfp_stream_set_accuracy(zfp, tolerance);
    zfp->maxbits = minbits << 4;
    return tolerance;
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

  // dimensionality
  uint dimensionality() const { return dims; }

  // underlying scalar type
  zfp_type scalar_type() const { return type; }

protected:
  // number of values per block
  uint block_size() const { return 1u << (2 * dims); }

  // allocate memory for bit stream buffer
  void alloc()
  {
    size_t bytes = (ZFP_MAX_BITS + CHAR_BIT - 1) / CHAR_BIT;
    uchar* buffer = (uchar*)zfp::allocate(bytes);
    stream_close(zfp->stream);
    zfp_stream_set_bit_stream(zfp, stream_open(buffer, bytes));
  }

  // free memory associated with compressed data
  void free()
  {
    if (zfp && zfp->stream) {
      zfp::deallocate((uchar*)stream_data(zfp->stream));
      stream_close(zfp->stream);
      zfp_stream_set_bit_stream(zfp, 0);
    }
    nx = ny = nz = 0;
    bx = by = bz = 0;
    tx = ty = tz = 0;
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

  // number of bytes of compressed data
  virtual size_t storage_size(uint mask = ZFP_DATA_PAYLOAD) const
  {
    size_t size = 0;
    if (mask & ZFP_DATA_META)
      size += sizeof(varray) + sizeof(*zfp); // + sizeof(*zfp->stream);
    return size;
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
  uint minbits;     // smallest non-empty block size in bits
  uint nx, ny, nz;  // array dimensions
  uint bx, by, bz;  // array dimensions in number of blocks
  uint tx, ty, tz;  // array dimensions in number of tiles
  uint blocks;      // number of blocks
  uint tiles;       // number of tiles
  zfp_stream* zfp;  // compression parameters and buffer shared among tiles
};

}

#endif
