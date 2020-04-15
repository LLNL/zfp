#ifndef ZFP_BLOCK3_H
#define ZFP_BLOCK3_H

#include "zfpcodec.h"
#include "zfp/memory.h"

namespace zfp {

// compressed 3D array of scalars
template < typename Scalar, class Codec = zfp::codec<Scalar, 3> >
class BlockStore3 {
public:
  // default constructor
  BlockStore3() :
    nx(0), ny(0), nz(0),
    bx(0), by(0), bz(0),
    bits_per_block(0),
    data(0),
    bytes(0),
    codec(0)
  {}

  // block store for array of size nx * ny * nz and given rate
  BlockStore3(uint nx, uint ny, uint nz, double rate) :
    nx(nx),
    ny(ny),
    nz(nz),
    bx((nx + 3) / 4),
    by((ny + 3) / 4),
    bz((nz + 3) / 4),
    bits_per_block(0),
    data(0),
    bytes(0),
    codec(0)
  {
    set_rate(rate);
  }

  // perform a deep copy
  void deep_copy(const BlockStore3& s)
  {
    free();
    nx = s.nx;
    ny = s.ny;
    nz = s.nz;
    bx = s.bx;
    by = s.by;
    bz = s.bz;
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
  void resize(uint nx, uint ny, uint nz, bool clear = true)
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
      alloc(clear);
    }
  }

  // number of bytes of compressed data
  size_t compressed_size() const { return bytes; }

  // pointer to compressed data for read or write access
  void* compressed_data() const { return data; }

  // total number of blocks
  size_t blocks() const { return size_t(bx) * size_t(by) * size_t(bz); }

  // array size in blocks
  size_t block_size_x() const { return bx; }
  size_t block_size_y() const { return by; }
  size_t block_size_z() const { return bz; }

  // flat block index for block (i, j, k)
  uint block_index(uint i, uint j, uint k) const { return (i / 4) + bx * ((j / 4) + by * (k / 4)); }

  // encoding of block dimensions
  uint block_shape(uint block_index) const { return shape(block_index); }

  // encode contiguous block with given index
  size_t encode(uint block_index, const Scalar* block) const
  {
    return codec->encode_block(offset(block_index), shape(block_index), block);
  }

  // encode block with given index from strided array
  size_t encode(uint block_index, const Scalar* p, ptrdiff_t sx, ptrdiff_t sy, ptrdiff_t sz) const
  {
    return codec->encode_block_strided(offset(block_index), shape(block_index), p, sx, sy, sz);
  }

  // decode contiguous block with given index
  size_t decode(uint block_index, Scalar* block) const
  {
    return codec->decode_block(offset(block_index), shape(block_index), block);
  }

  // decode block with given index to strided array
  size_t decode(uint block_index, Scalar* p, ptrdiff_t sx, ptrdiff_t sy, ptrdiff_t sz) const
  {
    return codec->decode_block_strided(offset(block_index), shape(block_index), p, sx, sy, sz);
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
    codec = new Codec(data, bytes);
    codec->set_rate(rate());
  }

  // free block store
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

  // bit offset to block store
  size_t offset(uint block_index) const { return block_index * bits_per_block; }

  // shape (sx, sy, sz) of block containing array index (i, j, k)
  void shape(uint& sx, uint& sy, uint& sz, uint i, uint j, uint k) const
  { 
    // right operand is 0x3 if i, j, or k is in a partial block; otherwise 0x0
    sx = -nx & (((i ^ nx) - 4) >> (CHAR_BIT * sizeof(uint) - 2));
    sy = -ny & (((j ^ ny) - 4) >> (CHAR_BIT * sizeof(uint) - 2));
    sz = -nz & (((k ^ nz) - 4) >> (CHAR_BIT * sizeof(uint) - 2));
  }

  // shape of block containing array index (i, j, k)
  uint shape(uint i, uint j, uint k) const
  {
    uint sx, sy, sz;
    shape(sx, sy, sz, i, j, k);
    return sx + 4 * (sy + 4 * sz);
  }

  // shape of block with given global block index
  uint shape(uint block_index) const
  {
    uint i = 4 * (block_index % bx); block_index /= bx;
    uint j = 4 * (block_index % by); block_index /= by;
    uint k = 4 * block_index;
    return shape(i, j, k);
  }

  static const size_t block_size = 4 * 4 * 4; // block size in number of elements

  uint nx, ny, nz;     // array dimensions
  uint bx, by, bz;     // array dimensions in number of blocks
  uint bits_per_block; // number of bits of compressed storage per block
  void* data;          // pointer to compressed blocks
  size_t bytes;        // compressed data size
  Codec* codec;        // compression codec
};

}

#endif
