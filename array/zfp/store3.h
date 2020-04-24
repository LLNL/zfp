#ifndef ZFP_STORE3_H
#define ZFP_STORE3_H

#include "zfp/store.h"
#include "zfp/memory.h"

namespace zfp {

// compressed block store for 3D array
template <typename Scalar, class Codec>
class BlockStore3 : public BlockStore {
public:
  // default constructor
  BlockStore3() :
    nx(0), ny(0), nz(0),
    bx(0), by(0), bz(0)
  {}

  // block store for array of size nx * ny * nz and given rate
  BlockStore3(size_t nx, size_t ny, size_t nz, double rate) :
    nx(nx),
    ny(ny),
    nz(nz),
    bx((nx + 3) / 4),
    by((ny + 3) / 4),
    bz((nz + 3) / 4)
  {
    set_rate(rate);
  }

  // destructor
  ~BlockStore3() { free(); }

  // perform a deep copy
  void deep_copy(const BlockStore3& s)
  {
    free();
    BlockStore::deep_copy(s);
    nx = s.nx;
    ny = s.ny;
    nz = s.nz;
    bx = s.bx;
    by = s.by;
    bz = s.bz;
  }

  // rate in bits per value
  double rate() const { return double(bits_per_block) / block_size; }

  // set rate in bits per value
  double set_rate(double rate)
  {
    free();
    rate = Codec::nearest_rate(rate);
    bits_per_block = uint(rate * block_size);
    alloc(blocks(), true);
    return rate;
  }

  // resize array
  void resize(size_t nx, size_t ny, size_t nz, bool clear = true)
  {
    free();
    if (nx == 0 || ny == 0 || nz == 0) {
      this->nx = this->ny = this->nz = 0;
      bx = by = bz = 0;
    }
    else {
      this->nx = nx;
      this->ny = ny;
      this->nz = nz;
      bx = (nx + 3) / 4;
      by = (ny + 3) / 4;
      bz = (nz + 3) / 4;
      alloc(blocks(), clear);
    }
  }

  // total number of blocks
  size_t blocks() const { return bx * by * bz; }

  // array size in blocks
  size_t block_size_x() const { return bx; }
  size_t block_size_y() const { return by; }
  size_t block_size_z() const { return bz; }

  // flat block index for block (i, j, k)
  size_t block_index(size_t i, size_t j, size_t k) const { return (i / 4) + bx * ((j / 4) + by * (k / 4)); }

  // encoding of block dimensions
  uint block_shape(size_t block_index) const { return shape(block_index); }

  // encode contiguous block with given index
  size_t encode(Codec* codec, size_t block_index, const Scalar* block) const
  {
    return codec->encode_block(offset(block_index), shape(block_index), block);
  }

  // encode block with given index from strided array
  size_t encode(Codec* codec, size_t block_index, const Scalar* p, ptrdiff_t sx, ptrdiff_t sy, ptrdiff_t sz) const
  {
    return codec->encode_block_strided(offset(block_index), shape(block_index), p, sx, sy, sz);
  }

  // decode contiguous block with given index
  size_t decode(Codec* codec, size_t block_index, Scalar* block) const
  {
    return codec->decode_block(offset(block_index), shape(block_index), block);
  }

  // decode block with given index to strided array
  size_t decode(Codec* codec, size_t block_index, Scalar* p, ptrdiff_t sx, ptrdiff_t sy, ptrdiff_t sz) const
  {
    return codec->decode_block_strided(offset(block_index), shape(block_index), p, sx, sy, sz);
  }

protected:
  // shape of block with given global block index
  uint shape(size_t block_index) const
  {
    size_t i = 4 * (block_index % bx); block_index /= bx;
    size_t j = 4 * (block_index % by); block_index /= by;
    size_t k = 4 * block_index;
    uint mx = shape_code(i, nx);
    uint my = shape_code(j, ny);
    uint mz = shape_code(k, nz);
    return mx + 4 * (my + 4 * mz);
  }

  static const size_t block_size = 4 * 4 * 4; // block size in number of elements

  size_t nx, ny, nz; // array dimensions
  size_t bx, by, bz; // array dimensions in number of blocks
};

}

#endif
