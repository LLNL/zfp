#ifndef ZFP_STORE4_H
#define ZFP_STORE4_H

#include "zfp/store.h"
#include "zfp/memory.h"

namespace zfp {

// compressed block store for 4D array
template <typename Scalar, class Codec>
class BlockStore4 : public BlockStore {
public:
  // default constructor
  BlockStore4() :
    nx(0), ny(0), nz(0), nw(0),
    bx(0), by(0), bz(0), bw(0)
  {}

  // block store for array of size nx * ny * nz * nw and given rate
  BlockStore4(size_t nx, size_t ny, size_t nz, size_t nw, double rate) :
    nx(nx),
    ny(ny),
    nz(nz),
    nw(nw),
    bx((nx + 3) / 4),
    by((ny + 3) / 4),
    bz((nz + 3) / 4),
    bw((nw + 3) / 4)
  {
    set_rate(rate);
  }

  // destructor
  ~BlockStore4() { free(); }

  // perform a deep copy
  void deep_copy(const BlockStore4& s)
  {
    free();
    BlockStore::deep_copy(s);
    nx = s.nx;
    ny = s.ny;
    nz = s.nz;
    nw = s.nw;
    bx = s.bx;
    by = s.by;
    bz = s.bz;
    bw = s.bw;
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
  void resize(size_t nx, size_t ny, size_t nz, size_t nw, bool clear = true)
  {
    free();
    if (nx == 0 || ny == 0 || nz == 0 || nw == 0) {
      this->nx = this->ny = this->nz = this->nw = 0;
      bx = by = bz = bw = 0;
    }
    else {
      this->nx = nx;
      this->ny = ny;
      this->nz = nz;
      this->nw = nw;
      bx = (nx + 3) / 4;
      by = (ny + 3) / 4;
      bz = (nz + 3) / 4;
      bw = (nw + 3) / 4;
      alloc(blocks(), clear);
    }
  }

  // total number of blocks
  size_t blocks() const { return bx * by * bz * bw; }

  // array size in blocks
  size_t block_size_x() const { return bx; }
  size_t block_size_y() const { return by; }
  size_t block_size_z() const { return bz; }
  size_t block_size_w() const { return bw; }

  // flat block index for block (i, j, k)
  size_t block_index(size_t i, size_t j, size_t k, size_t l) const { return (i / 4) + bx * ((j / 4) + by * ((k / 4) + bz * (l / 4))); }

  // encoding of block dimensions
  uint block_shape(size_t block_index) const { return shape(block_index); }

  // encode contiguous block with given index
  size_t encode(Codec* codec, size_t block_index, const Scalar* block) const
  {
    return codec->encode_block(offset(block_index), shape(block_index), block);
  }

  // encode block with given index from strided array
  size_t encode(Codec* codec, size_t block_index, const Scalar* p, ptrdiff_t sx, ptrdiff_t sy, ptrdiff_t sz, ptrdiff_t sw) const
  {
    return codec->encode_block_strided(offset(block_index), shape(block_index), p, sx, sy, sz, sw);
  }

  // decode contiguous block with given index
  size_t decode(Codec* codec, size_t block_index, Scalar* block) const
  {
    return codec->decode_block(offset(block_index), shape(block_index), block);
  }

  // decode block with given index to strided array
  size_t decode(Codec* codec, size_t block_index, Scalar* p, ptrdiff_t sx, ptrdiff_t sy, ptrdiff_t sz, ptrdiff_t sw) const
  {
    return codec->decode_block_strided(offset(block_index), shape(block_index), p, sx, sy, sz, sw);
  }

protected:
  // shape of block with given global block index
  uint shape(size_t block_index) const
  {
    size_t i = 4 * (block_index % bx); block_index /= bx;
    size_t j = 4 * (block_index % by); block_index /= by;
    size_t k = 4 * (block_index % bz); block_index /= bz;
    size_t l = 4 * block_index;
    uint mx = shape_code(i, nx);
    uint my = shape_code(j, ny);
    uint mz = shape_code(k, nz);
    uint mw = shape_code(l, nw);
    return mx + 4 * (my + 4 * (mz + 4 * mw));
  }

  static const size_t block_size = 4 * 4 * 4 * 4; // block size in number of elements

  size_t nx, ny, nz, nw; // array dimensions
  size_t bx, by, bz, bw; // array dimensions in number of blocks
};

}

#endif
