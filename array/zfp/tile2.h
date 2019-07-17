#ifndef ZFP_TILE2_H
#define ZFP_TILE2_H

#include "tile.h"

namespace zfp {

// tile of 4K variable-rate 4x4 blocks
template <typename Scalar>
class Tile2 : public Tile {
public:
  Tile2(uint minrate = 4u) :
    Tile(4 * 4 * minrate)
  {}

  // allocate compressed data and compress block with tile-local index 'id'
  void compress(zfp_stream* zfp, const Scalar* block, uint id, uchar shape = 0)
  {
    offset p = null;
    // compress block and determine its size in bits
    stream_rewind(zfp->stream);
    uint bits = zfp::codec<Scalar>::encode_block_2(zfp, block, shape);
    // if block is empty, no storage is needed; otherwise, find space for it
    if (bits > 1u) {
      // flush stream and determine block size in words
      bits += stream_flush(zfp->stream);
      size_t words = word_size(bits);
      // allocate memory for and copy block to persistent storage
      p = allocate(words);
      const word* buffer = static_cast<word*>(stream_data(zfp->stream));
      std::copy(buffer, buffer + words, data + offset_words(p));
    }
    pos[id] = p;
  }

  // decompress block with tile-local index 'id' and free compressed data
  void decompress(zfp_stream* zfp, Scalar* block, uint id, uchar shape = 0)
  {
    offset p = pos[id];
    if (p == null) {
      // empty block; fill with zeros
      std::fill(block, block + 4 * 4, Scalar(0));
    }
    else {
      void* buffer = stream_data(zfp->stream);
      size_t size = stream_capacity(zfp->stream);
      stream_reopen(zfp->stream, data, capacity());
      stream_rseek(zfp->stream, offset_bits(p));
      uint bits = zfp::codec<Scalar>::decode_block_2(zfp, block, shape);
      size_t words = word_size(bits);
      uint s = slot_size(words);
      enqueue_slot(s, p);
      stream_reopen(zfp->stream, buffer, size);
    }
  }

  static const uint bx = 64; // number of blocks per tile along x
  static const uint by = 64; // number of blocks per tile along y

protected:
/*
  using Tile::blocks;
  using Tile::null;
  using Tile::zfp;
  using Tile::stream;
  using Tile::pos;
  using Tile::head;
  using Tile::quantum;
  using Tile::capacity;
  using Tile::size;
  using Tile::data;
  using Tile::buffer;
*/
};

}

#endif
