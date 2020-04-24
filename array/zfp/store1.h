#ifndef ZFP_STORE1_H
#define ZFP_STORE1_H

#include "zfp/store.h"
#include "zfp/memory.h"

namespace zfp {

// compressed block store for 1D array
template <typename Scalar, class Codec>
class BlockStore1 : public BlockStore {
public:
  // default constructor
  BlockStore1() :
    nx(0),
    bx(0)
  {}

  // block store for array of size nx and given rate
  BlockStore1(size_t nx, double rate) :
    nx(nx),
    bx((nx + 3) / 4)
  {
    set_rate(rate);
  }

  // destructor
  ~BlockStore1() { free(); }

  // perform a deep copy
  void deep_copy(const BlockStore1& s)
  {
    free();
    BlockStore::deep_copy(s);
    nx = s.nx;
    bx = s.bx;
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
  void resize(size_t nx, bool clear = true)
  {
    free();
    if (nx == 0) {
      this->nx = 0;
      bx = 0;
    }
    else {
      this->nx = nx;
      bx = (nx + 3) / 4;
      alloc(blocks(), clear);
    }
  }

  // total number of blocks
  size_t blocks() const { return bx; }

  // array size in blocks
  size_t block_size_x() const { return bx; }

  // flat block index for block i
  size_t block_index(size_t i) const { return i / 4; }

  // encoding of block dimensions
  uint block_shape(size_t block_index) const { return shape(block_index); }

  // encode contiguous block with given index
  size_t encode(Codec* codec, size_t block_index, const Scalar* block) const
  {
    return codec->encode_block(offset(block_index), shape(block_index), block);
  }

  // encode block with given index from strided array
  size_t encode(Codec* codec, size_t block_index, const Scalar* p, ptrdiff_t sx) const
  {
    return codec->encode_block_strided(offset(block_index), shape(block_index), p, sx);
  }

  // decode contiguous block with given index
  size_t decode(Codec* codec, size_t block_index, Scalar* block) const
  {
    return codec->decode_block(offset(block_index), shape(block_index), block);
  }

  // decode block with given index to strided array
  size_t decode(Codec* codec, size_t block_index, Scalar* p, ptrdiff_t sx) const
  {
    return codec->decode_block_strided(offset(block_index), shape(block_index), p, sx);
  }

protected:
  // shape of block with given global block index
  uint shape(size_t block_index) const
  {
    size_t i = 4 * block_index;
    uint mx = shape_code(i, nx);
    return mx;
  }

  static const size_t block_size = 4; // block size in number of elements

  size_t nx; // array dimensions
  size_t bx; // array dimensions in number of blocks
};

}

#endif
