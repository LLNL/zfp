#ifndef ZFP_CARRAY_H
#define ZFP_CARRAY_H

#include <algorithm>
#include <climits>
#include <cstring>
#include <stdexcept>
#include <string>

#include "zfp.h"
#include "zfp/memory.h"

namespace zfp {

// abstract base class for compressed array of scalars
class const_array {
protected:
  // default constructor
  const_array() :
    dims(0), type(zfp_type_none),
    nx(0), ny(0), nz(0),
    bx(0), by(0), bz(0),
    blocks(0),
    bytes(0), data(0),
    zfp(0),
    shape(0)
  {}

  // generic array with 'dims' dimensions and scalar type 'type'
  const_array(uint dims, zfp_type type) :
    dims(dims), type(type),
    nx(0), ny(0), nz(0),
    bx(0), by(0), bz(0),
    blocks(0),
    bytes(0), data(0),
    zfp(zfp_stream_open(0)),
    shape(0)
  {}

  // copy constructor--performs a deep copy
  const_array(const const_array& a) :
    data(0),
    zfp(0),
    shape(0)
  {
    deep_copy(a);
  }

  // assignment operator--performs a deep copy
  const_array& operator=(const const_array& a)
  {
    deep_copy(a);
    return *this;
  }

public:
  // public virtual destructor (can delete array through base class pointer)
  virtual ~const_array()
  {
    free();
    zfp_stream_close(zfp);
  }

  // rate in bits per value
  double rate() const { return CHAR_BIT * double(bytes) / (nx * ny * nz); }

  // set compression rate in bits per value
  double set_rate(double rate)
  {
    rate = zfp_stream_set_rate(zfp, rate, type, dims, 1);
    if (blocks)
      alloc();
    return rate;
  }

  // set precision in uncompressed bits per value
  uint set_precision(uint precision)
  {
    precision = zfp_stream_set_precision(zfp, precision);
    zfp->maxbits = maxbits;
    if (blocks)
      alloc(false);
    return precision;
  }

  // set compression rate in bits per value
  double set_accuracy(double tolerance)
  {
    tolerance = zfp_stream_set_accuracy(zfp, tolerance);
    zfp->maxbits = maxbits;
    if (blocks)
      alloc(false);
    return tolerance;
  }

  // enable reversible (lossless) mode
  void set_reversible()
  {
    zfp_stream_set_reversible(zfp);
    zfp->maxbits = maxbits;
    if (blocks)
      alloc(false);
  }

  // empty cache without compressing modified cached blocks
  virtual void clear_cache() const = 0;

  // number of bytes of compressed data
  size_t compressed_size() const { return bytes; }

  // pointer to compressed data for read or write access
  uchar* compressed_data() const { return data; }

  // dimensionality
  uint dimensionality() const { return dims; }

  // underlying scalar type
  zfp_type scalar_type() const { return type; }

protected:
  // number of values per block
  uint block_size() const { return 1u << (2 * dims); }

  // allocate memory for compressed data
  void alloc(bool clear = true)
  {
    zfp_field field;
    field.type = type;
    field.nx = nx;
    field.ny = ny;
    field.nz = nz;
    field.nw = 0;
    bytes = zfp_stream_maximum_size(zfp, &field);
    zfp::reallocate_aligned(data, bytes, 0x100u);
    if (clear)
      std::fill(data, data + bytes, 0);
    stream_close(zfp->stream);
    zfp_stream_set_bit_stream(zfp, stream_open(data, bytes));
    clear_cache();
  }

  // free memory associated with compressed data
  void free()
  {
#warning "implement"
    nx = ny = nz = 0;
    bx = by = bz = 0;
    blocks = 0;
    stream_close(zfp->stream);
    zfp_stream_set_bit_stream(zfp, 0);
    bytes = 0;
    zfp::deallocate_aligned(data);
    data = 0;
    zfp::deallocate(shape);
    shape = 0;
  }

  // perform a deep copy
  void deep_copy(const const_array& a)
  {
abort();
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
    bytes = a.bytes;

    // copy dynamically allocated data
    zfp::clone_aligned(data, a.data, bytes, 0x100u);
    if (zfp) {
      if (zfp->stream)
        stream_close(zfp->stream);
      zfp_stream_close(zfp);
    }
    zfp = zfp_stream_open(0);
    *zfp = *a.zfp;
    zfp_stream_set_bit_stream(zfp, stream_open(data, bytes));
    zfp::clone(shape, a.shape, blocks);
  }

  // default number of cache lines for array with n blocks
  static uint lines(size_t n)
  {
    // compute m = O(sqrt(n))
    size_t m;
    for (m = 1; m * m < n; m *= 2);
    return static_cast<uint>(m);
  }

  static const uint maxbits = 0x1000u; // maximum supported block size

  uint dims;           // array dimensionality (1, 2, or 3)
  zfp_type type;       // scalar type
  uint nx, ny, nz;     // array dimensions
  uint bx, by, bz;     // array dimensions in number of blocks
  uint blocks;         // number of blocks
  size_t bytes;        // total bytes of compressed data
  mutable uchar* data; // pointer to compressed data
  zfp_stream* zfp;     // compressed stream of blocks
  uchar* shape;        // precomputed block dimensions (or null if uniform)
};

}

#endif
