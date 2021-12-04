#ifndef ZFP_TILE3_H
#define ZFP_TILE3_H

#include "tile.h"

namespace zfp {
namespace internal {

// tile of 16x16x16 variable-rate 4x4x4 blocks
template <typename Scalar, class Codec>
class Tile3 : public Tile<Scalar, Codec> {
public:
  Tile3(size_t minbits = 256) :
    Tile<Scalar, Codec>(minbits)
  {}

#warning "reorder arguments?"
  // allocate compressed data and compress block with tile-local index 'id'
  size_t encode(const Codec& codec, const Scalar* block, ptrdiff_t sx, ptrdiff_t sy, ptrdiff_t sz, size_t id, uchar shape = 0)
  {
    size_t bits = encode_block(codec, block, sx, sy, sz, shape);
    store_block(id, static_cast<const word*>(codec.data()), bits);
    return bits;
  }

  // decompress block with tile-local index 'id' and free compressed data
  size_t decode(const Codec& codec, Scalar* block, ptrdiff_t sx, ptrdiff_t sy, ptrdiff_t sz, size_t id, uchar shape = 0, bool cache_block = true)
  {
    size_t bits = 0;
    offset p = pos[id];
    assert(p != cached);
    if (cache_block)
      pos[id] = cached;
    if (p == null) {
      // empty block; fill with zeros
      // TODO: handle partial blocks and strides
      assert(!sx && !sy && !sz && !shape);
      std::fill(block, block + block_size, Scalar(0));
    }
    else {
      // decompress block
      bits = decode_block(codec, block, sx, sy, sz, p, shape);
      if (cache_block) {
        // free space occupied by compressed data
        size_t words = word_size(bits);
        deallocate(p, words);
      }
    }
#if DEBUG
    print_lists();
#endif
    return bits;
  }

  static const size_t bx = 16;                // number of blocks per tile along x
  static const size_t by = 16;                // number of blocks per tile along y
  static const size_t bz = 16;                // number of blocks per tile along z
  static const size_t block_size = 4 * 4 * 4; // number of scalars per block

protected:
  using Tile<Scalar, Codec>::word_size;
  using Tile<Scalar, Codec>::offset_bits;
  using Tile<Scalar, Codec>::capacity;
  using Tile<Scalar, Codec>::deallocate;
  using Tile<Scalar, Codec>::store_block;
  using Tile<Scalar, Codec>::null;
  using Tile<Scalar, Codec>::cached;
  using Tile<Scalar, Codec>::data;
  using Tile<Scalar, Codec>::pos;

  typedef typename Tile<Scalar, Codec>::offset offset;
  typedef typename Tile<Scalar, Codec>::word word;

  // compress block to temporary storage and return its storage size in bits
  size_t encode_block(const Codec& codec, const Scalar* block, ptrdiff_t sx, ptrdiff_t sy, ptrdiff_t sz, uchar shape = 0) const
  {
    return (sx || sy || sz)
             ? codec.encode_block_strided(0, shape, block, sx, sy, sz)
             : codec.encode_block(0, shape, block);
  }

  // decompress block stored at offset p
  size_t decode_block(const Codec& codec, Scalar* block, ptrdiff_t sx, ptrdiff_t sy, ptrdiff_t sz, offset p, uchar shape = 0) const
  {
    // TODO: maintain tile-local codec to avoid expensive deep copy
    Codec c = codec;
    c.open(data, capacity());
#warning "why not copy compressed data to buffer?"
// can also copy from buffer only up to max slot size; no codec maxbits needed
// though this may preclude reversible compression
// other codecs may require special care and not support bit stream truncation
    return (sx || sy || sz)
             ? c.decode_block_strided(offset_bits(p), shape, block, sx, sy, sz)
             : c.decode_block(offset_bits(p), shape, block);
  }
};

} // internal
} // zfp

#endif
