#ifndef ZFP_STORE4_HPP
#define ZFP_STORE4_HPP

#include "zfp/internal/array/store.hpp"

namespace zfp {
namespace internal {

// compressed block store for 4D array
template <typename Scalar, class Codec, class Index>
class BlockStore4 : public BlockStore<Codec, Index> {
public:
  // default constructor
  BlockStore4() :
    nx(0), ny(0), nz(0), nw(0),
    bx(0), by(0), bz(0), bw(0)
  {}

  // block store for array of size nx * ny * nz * nw and given configuration
  BlockStore4(size_t nx, size_t ny, size_t nz, size_t nw, const zfp_config& config)
  {
    set_size(nx, ny, nz, nw);
    this->set_config(config);
  }

  // perform a deep copy
  void deep_copy(const BlockStore4& s)
  {
    free();
    BlockStore<Codec, Index>::deep_copy(s);
    nx = s.nx;
    ny = s.ny;
    nz = s.nz;
    nw = s.nw;
    bx = s.bx;
    by = s.by;
    bz = s.bz;
    bw = s.bw;
  }

  // resize array
  void resize(size_t nx, size_t ny, size_t nz, size_t nw, bool clear = true)
  {
    free();
    set_size(nx, ny, nz, nw);
    if (blocks())
      alloc(clear);
  }

  // byte size of store data structure components indicated by mask
  virtual size_t size_bytes(uint mask = ZFP_DATA_ALL) const
  { 
    size_t size = 0;
    size += BlockStore<Codec, Index>::size_bytes(mask);
    if (mask & ZFP_DATA_META)
      size += sizeof(*this) - sizeof(BlockStore<Codec, Index>);
    return size;
  }

  // conservative buffer size 
  virtual size_t buffer_size() const
  {
    zfp_field* field = zfp_field_4d(0, codec.type, nx, ny, nz, nw);
    size_t size = codec.buffer_size(field);
    zfp_field_free(field);
    return size;
  }

  // number of elements per block
  virtual size_t block_size() const { return 4 * 4 * 4 * 4; }

  // total number of blocks
  virtual size_t blocks() const { return bx * by * bz * bw; }

  // array size in blocks
  size_t block_size_x() const { return bx; }
  size_t block_size_y() const { return by; }
  size_t block_size_z() const { return bz; }
  size_t block_size_w() const { return bw; }

  // flat block index for element (i, j, k, l)
  size_t block_index(size_t i, size_t j, size_t k, size_t l) const { return (i / 4) + bx * ((j / 4) + by * ((k / 4) + bz * (l / 4))); }

  // encoding of block dimensions
  uint block_shape(size_t block_index) const
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

  // encode contiguous block with given index
  size_t encode(size_t block_index, const Scalar* block)
  {
    size_t size = codec.encode_block(offset(block_index), block_shape(block_index), block);
    index.set_block_size(block_index, size);
    return size;
  }

  // encode block with given index from strided array
  size_t encode(size_t block_index, const Scalar* p, ptrdiff_t sx, ptrdiff_t sy, ptrdiff_t sz, ptrdiff_t sw)
  {
    size_t size = codec.encode_block_strided(offset(block_index), block_shape(block_index), p, sx, sy, sz, sw);
    index.set_block_size(block_index, size);
    return size;
  }

  // decode contiguous block with given index
  size_t decode(size_t block_index, Scalar* block) const
  {
    return codec.decode_block(offset(block_index), block_shape(block_index), block);
  }

  // decode block with given index to strided array
  size_t decode(size_t block_index, Scalar* p, ptrdiff_t sx, ptrdiff_t sy, ptrdiff_t sz, ptrdiff_t sw) const
  {
    return codec.decode_block_strided(offset(block_index), block_shape(block_index), p, sx, sy, sz, sw);
  }

protected:
  using BlockStore<Codec, Index>::alloc;
  using BlockStore<Codec, Index>::free;
  using BlockStore<Codec, Index>::offset;
  using BlockStore<Codec, Index>::shape_code;
  using BlockStore<Codec, Index>::index;
  using BlockStore<Codec, Index>::codec;

  // set array dimensions
  void set_size(size_t nx, size_t ny, size_t nz, size_t nw)
  {
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
    }
    index.resize(blocks());
  }

  size_t nx, ny, nz, nw; // array dimensions
  size_t bx, by, bz, bw; // array dimensions in number of blocks
};

} // internal
} // zfp

#endif
