#ifndef ZFP_BLOCK1_H
#define ZFP_BLOCK1_H

#include "zfpcodec.h"
#include "zfp/memory.h"

namespace zfp {

// compressed 2D array of scalars
template < typename Scalar, class Codec = zfp::codec<Scalar, 1> >
class BlockStore1 {
public:
  // default constructor
  BlockStore1() :
    nx(0),
    bx(0),
    bits_per_block(0),
    data(0),
    bytes(0)
  {}

  // block store for array of size nx and given rate
  BlockStore1(uint nx, double rate) :
    nx(nx),
    bx((nx + 3) / 4),
    bits_per_block(0),
    data(0),
    bytes(0)
  {
    set_rate(rate);
  }

  // perform a deep copy
  void deep_copy(const BlockStore1& s)
  {
    free();
    nx = s.nx;
    bx = s.bx;
    bits_per_block = s.bits_per_block;
    bytes = s.bytes;
    zfp::clone_aligned(data, s.data, s.bytes, ZFP_MEMORY_ALIGNMENT);
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
  void resize(uint nx, bool clear = true)
  {
    free();
    if (nx == 0) {
      this->nx = 0;
      bx = 0;
    }
    else {
      this->nx = nx;
      bx = (nx + 3) / 4;
      alloc(clear);
    }
  }

  // number of bytes of compressed data
  size_t compressed_size() const { return bytes; }

  // pointer to compressed data for read or write access
  void* compressed_data() const { return data; }

  // total number of blocks
  size_t blocks() const { return size_t(bx); }

  // array size in blocks
  size_t block_size_x() const { return bx; }

  // flat block index for block i
  uint block_index(uint i) const { return i / 4; }

  // encoding of block dimensions
  uint block_shape(uint block_index) const { return shape(block_index); }

  // encode contiguous block with given index
  size_t encode(Codec* codec, uint block_index, const Scalar* block) const
  {
    return codec->encode_block(offset(block_index), shape(block_index), block);
  }

  // encode block with given index from strided array
  size_t encode(Codec* codec, uint block_index, const Scalar* p, ptrdiff_t sx) const
  {
    return codec->encode_block_strided(offset(block_index), shape(block_index), p, sx);
  }

  // decode contiguous block with given index
  size_t decode(Codec* codec, uint block_index, Scalar* block) const
  {
    return codec->decode_block(offset(block_index), shape(block_index), block);
  }

  // decode block with given index to strided array
  size_t decode(Codec* codec, uint block_index, Scalar* p, ptrdiff_t sx) const
  {
    return codec->decode_block_strided(offset(block_index), shape(block_index), p, sx);
  }

protected:
  // allocate memory for persistent block store
  void alloc(bool clear = true)
  {
    size_t words = (blocks() * bits_per_block + CHAR_BIT * sizeof(uint64) - 1) / (CHAR_BIT * sizeof(uint64));
    bytes = words * sizeof(uint64);
    zfp::reallocate_aligned(data, bytes, ZFP_MEMORY_ALIGNMENT);
    if (clear)
      std::fill(static_cast<uint64*>(data), static_cast<uint64*>(data) + words, uint64(0));
  }

  // free block store
  void free()
  {
    if (data) {
      zfp::deallocate_aligned(data);
      data = 0;
      bytes = 0;
    }
  }

  // bit offset to block store
  size_t offset(uint block_index) const { return block_index * bits_per_block; }

  // shape of block with given global block index
  uint shape(uint block_index) const
  {
    uint i = 4 * block_index;
    uint sx = -nx & (((i ^ nx) - 4) >> (CHAR_BIT * sizeof(uint) - 2));
    return sx;
  }

  static const size_t block_size = 4; // block size in number of elements

  uint nx;             // array dimensions
  uint bx;             // array dimensions in number of blocks
  uint bits_per_block; // number of bits of compressed storage per block
  void* data;          // pointer to compressed blocks
  size_t bytes;        // compressed data size
};

}

#endif
