#ifndef ZFP_BLOCK2_H
#define ZFP_BLOCK2_H

#include "zfpcodec.h"
#include "zfp/memory.h"

namespace zfp {

// compressed 2D array of scalars
template < typename Scalar, class Codec = zfp::codec<Scalar, 2> >
class BlockStorage2 {
public:
  // default constructor
  BlockStorage2() :
    nx(0), ny(0),
    bx(0), by(0),
    bits_per_block(0),
    data(0),
    bytes(0),
    codec(0)
  {}

  // block storage for array of size nx * ny and given rate
  BlockStorage2(uint nx, uint ny, double rate) :
    nx(nx),
    ny(ny),
    bx((nx + 3) / 4),
    by((ny + 3) / 4),
    bits_per_block(0),
    data(0),
    bytes(0),
    codec(0)
  {
    set_rate(rate);
  }

  // perform a deep copy
  void deep_copy(const BlockStorage2& s)
  {
    free();
    nx = s.nx;
    ny = s.ny;
    bx = s.bx;
    by = s.by;
    bits_per_block = s.bits_per_block;
    bytes = s.bytes;
    zfp::clone_aligned(data, s.data, s.bytes, ZFP_MEMORY_ALIGNMENT);
    codec = s.codec->clone();
  }

  // rate in bits per value
  double rate() const { return double(bits_per_block) / block_size; }

  // set rate in bits per value
  double set_rate(double rate)
  {
    free();
    rate = Codec::nearest_rate(rate);
    bits_per_block = uint(rate * block_size);
    alloc();
    return rate;
  }

  // resize array
  void resize(uint nx, uint ny, bool clear = true)
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
      alloc(clear);
    }
  }

  // number of bytes of compressed data
  size_t compressed_size() const { return bytes; }

  // pointer to compressed data for read or write access
  void* compressed_data() const { return data; }

  // total number of blocks
  size_t blocks() const { return size_t(bx) * size_t(by); }

  // array size in blocks
  size_t block_size_x() const { return bx; }
  size_t block_size_y() const { return by; }

  // flat block index for block (i, j)
  uint block_index(uint i, uint j) const { return (i / 4) + bx * (j / 4); }

  // encoding of block dimensions
  uint block_shape(uint block_index) const { return shape(block_index); }

  // encode contiguous block with given index
  size_t encode(uint block_index, const Scalar* block) const
  {
    return codec->encode_block(offset(block_index), shape(block_index), block);
  }

  // encode block with given index from strided array
  size_t encode(uint block_index, const Scalar* p, ptrdiff_t sx, ptrdiff_t sy) const
  {
    return codec->encode_block_strided(offset(block_index), shape(block_index), p, sx, sy);
  }

  // decode contiguous block with given index
  size_t decode(uint block_index, Scalar* block) const
  {
    return codec->decode_block(offset(block_index), shape(block_index), block);
  }

  // decode block with given index to strided array
  size_t decode(uint block_index, Scalar* p, ptrdiff_t sx, ptrdiff_t sy) const
  {
    return codec->decode_block_strided(offset(block_index), shape(block_index), p, sx, sy);
  }

protected:
  // allocate memory for persistent block storage
  void alloc(bool clear = true)
  {
    size_t words = (blocks() * bits_per_block + CHAR_BIT * sizeof(uint64) - 1) / (CHAR_BIT * sizeof(uint64));
    bytes = words * sizeof(uint64);
    zfp::reallocate_aligned(data, bytes, ZFP_MEMORY_ALIGNMENT);
    if (clear)
      std::fill(static_cast<uint64*>(data), static_cast<uint64*>(data) + words, uint64(0));
    codec = new Codec(data, bytes);
    codec->set_rate(rate());
  }

  // free block storage
  void free()
  {
    if (codec) {
      delete codec;
      codec = 0;
    }
    if (data) {
      zfp::deallocate_aligned(data);
      data = 0;
      bytes = 0;
    }
  }

  // bit offset to block storage
  size_t offset(uint block_index) const { return block_index * bits_per_block; }

  // shape (sx, sy) of block containing array index (i, j)
  void shape(uint& sx, uint& sy, uint i, uint j) const
  { 
    // right operand is 0x3 if i or j is in a partial block; otherwise 0x0
    sx = -nx & (((i ^ nx) - 4) >> (CHAR_BIT * sizeof(uint) - 2));
    sy = -ny & (((j ^ ny) - 4) >> (CHAR_BIT * sizeof(uint) - 2));
  }

  // shape of block containing array index (i, j)
  uint shape(uint i, uint j) const
  {
    uint sx, sy;
    shape(sx, sy, i, j);
    return sx + 4 * sy;
  }

  // shape of block with given global block index
  uint shape(uint block_index) const
  {
    uint i = 4 * (block_index % bx); block_index /= bx;
    uint j = 4 * block_index;
    return shape(i, j);
  }

  static const size_t block_size = 4 * 4; // block size in number of elements

  uint nx, ny;         // array dimensions
  uint bx, by;         // array dimensions in number of blocks
  uint bits_per_block; // number of bits of compressed storage per block
  void* data;          // pointer to compressed blocks
  size_t bytes;        // compressed data size
  Codec* codec;        // compression codec
};

}

#endif
