#ifndef ZFP_STORE2_HPP
#define ZFP_STORE2_HPP

#include "zfp/internal/array/store.hpp"

namespace zfp {
namespace internal {

// compressed block store for 2D array
template <typename Scalar, class Codec, class Index>
class BlockStore2 : public BlockStore<Codec, Index> {
public:
  // default constructor
  BlockStore2() :
    nx(0), ny(0),
    bx(0), by(0)
  {}

  // block store for array of size nx * ny and given configuration
  BlockStore2(size_t nx, size_t ny, const zfp_config& config)
  {
    set_size(nx, ny);
    this->set_config(config);
  }

  // perform a deep copy
  void deep_copy(const BlockStore2& s)
  {
    free();
    BlockStore<Codec, Index>::deep_copy(s);
    nx = s.nx;
    ny = s.ny;
    bx = s.bx;
    by = s.by;
  }

  // resize array
  void resize(size_t nx, size_t ny, bool clear = true)
  {
    free();
    set_size(nx, ny);
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
    zfp_field* field = zfp_field_2d(0, codec.type, nx, ny);
    size_t size = codec.buffer_size(field);
    zfp_field_free(field);
    return size;
  }

  // number of elements per block
  virtual size_t block_size() const { return 4 * 4; }

  // total number of blocks
  virtual size_t blocks() const { return bx * by; }

  // array size in blocks
  size_t block_size_x() const { return bx; }
  size_t block_size_y() const { return by; }

  // flat block index for element (i, j)
  size_t block_index(size_t i, size_t j) const { return (i / 4) + bx * (j / 4); }

  // encoding of block dimensions
  uint block_shape(size_t block_index) const
  {
    size_t i = 4 * (block_index % bx); block_index /= bx;
    size_t j = 4 * block_index;
    uint mx = shape_code(i, nx);
    uint my = shape_code(j, ny);
    return mx + 4 * my;
  }

  // encode contiguous block with given index
  size_t encode(size_t block_index, const Scalar* block)
  {
    size_t size = codec.encode_block(offset(block_index), block_shape(block_index), block);
    index.set_block_size(block_index, size);
    return size;
  }

  // encode block with given index from strided array
  size_t encode(size_t block_index, const Scalar* p, ptrdiff_t sx, ptrdiff_t sy)
  {
    size_t size = codec.encode_block_strided(offset(block_index), block_shape(block_index), p, sx, sy);
    index.set_block_size(block_index, size);
    return size;
  }

  // decode contiguous block with given index
  size_t decode(size_t block_index, Scalar* block) const
  {
    return codec.decode_block(offset(block_index), block_shape(block_index), block);
  }

  // decode block with given index to strided array
  size_t decode(size_t block_index, Scalar* p, ptrdiff_t sx, ptrdiff_t sy) const
  {
    return codec.decode_block_strided(offset(block_index), block_shape(block_index), p, sx, sy);
  }

protected:
  using BlockStore<Codec, Index>::alloc;
  using BlockStore<Codec, Index>::free;
  using BlockStore<Codec, Index>::offset;
  using BlockStore<Codec, Index>::shape_code;
  using BlockStore<Codec, Index>::index;
  using BlockStore<Codec, Index>::codec;

  // set array dimensions
  void set_size(size_t nx, size_t ny)
  {
    if (nx == 0 || ny == 0) {
      this->nx = this->ny = 0;
      bx = by = 0;
    }
    else {
      this->nx = nx;
      this->ny = ny;
      bx = (nx + 3) / 4;
      by = (ny + 3) / 4;
    }
    index.resize(blocks());
  }

  size_t nx, ny; // array dimensions
  size_t bx, by; // array dimensions in number of blocks
};

} // internal
} // zfp

#endif
