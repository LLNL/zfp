#ifndef ZFP_STORE2_H
#define ZFP_STORE2_H

#include "zfp/store.h"
#include "zfp/memory.h"

namespace zfp {

// compressed block store for 2D array
template <typename Scalar, class Codec>
class BlockStore2 : public BlockStore {
public:
  // default constructor
  BlockStore2() :
    nx(0), ny(0),
    bx(0), by(0)
  {}

  // block store for array of size nx * ny and given rate
  BlockStore2(size_t nx, size_t ny, double rate) :
    nx(nx),
    ny(ny),
    bx((nx + 3) / 4),
    by((ny + 3) / 4)
  {
    set_rate(rate);
  }

  // destructor
  ~BlockStore2() { free(); }

  // perform a deep copy
  void deep_copy(const BlockStore2& s)
  {
    free();
    BlockStore::deep_copy(s);
    nx = s.nx;
    ny = s.ny;
    bx = s.bx;
    by = s.by;
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
  void resize(size_t nx, size_t ny, bool clear = true)
  {
    free();
    if (nx == 0 || ny == 0) {
      this->nx = this->ny = 0;
      bx = by = 0;
    }
    else {
      this->nx = nx;
      this->ny = ny;
      bx = (nx + 3) / 4;
      by = (ny + 3) / 4;
      alloc(blocks(), clear);
    }
  }

  // total number of blocks
  size_t blocks() const { return bx * by; }

  // array size in blocks
  size_t block_size_x() const { return bx; }
  size_t block_size_y() const { return by; }

  // flat block index for block (i, j)
  size_t block_index(size_t i, size_t j) const { return (i / 4) + bx * (j / 4); }

  // encoding of block dimensions
  uint block_shape(size_t block_index) const { return shape(block_index); }

  // encode contiguous block with given index
  size_t encode(Codec* codec, size_t block_index, const Scalar* block) const
  {
    return codec->encode_block(offset(block_index), shape(block_index), block);
  }

  // encode block with given index from strided array
  size_t encode(Codec* codec, size_t block_index, const Scalar* p, ptrdiff_t sx, ptrdiff_t sy) const
  {
    return codec->encode_block_strided(offset(block_index), shape(block_index), p, sx, sy);
  }

  // decode contiguous block with given index
  size_t decode(Codec* codec, size_t block_index, Scalar* block) const
  {
    return codec->decode_block(offset(block_index), shape(block_index), block);
  }

  // decode block with given index to strided array
  size_t decode(Codec* codec, size_t block_index, Scalar* p, ptrdiff_t sx, ptrdiff_t sy) const
  {
    return codec->decode_block_strided(offset(block_index), shape(block_index), p, sx, sy);
  }

protected:
  // shape of block with given global block index
  uint shape(size_t block_index) const
  {
    size_t i = 4 * (block_index % bx); block_index /= bx;
    size_t j = 4 * block_index;
    uint mx = shape_code(i, nx);
    uint my = shape_code(j, ny);
    return mx + 4 * my;
  }

  static const size_t block_size = 4 * 4; // block size in number of elements

  size_t nx, ny; // array dimensions
  size_t bx, by; // array dimensions in number of blocks
};

}

#endif
