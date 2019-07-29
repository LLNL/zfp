#ifndef ZFP_TILE2_H
#define ZFP_TILE2_H

#include <cmath>
#include "tile.h"

namespace zfp {

// tile of 64x64 variable-rate 4x4 blocks
template <typename Scalar>
class Tile2 : public Tile {
public:
  Tile2(double minrate = 4u) :
    Tile((uint)std::lrint(4 * 4 * minrate))
  {}

#ifdef DEBUG
  void fragmentation(zfp_stream* zfp)
  {
    uchar* slot = new uchar[0x10000];
    std::fill(slot, slot + 0x10000, 0xff);
    void* buffer = stream_data(zfp->stream);
    size_t size = stream_capacity(zfp->stream);
    size_t usage = 0;
    size_t zeros = 0;
    stream_reopen(zfp->stream, data, capacity());
    // process compressed blocks
    for (uint b = 0; b < blocks; b++) {
      offset p = pos[b];
      if (p < 0xfffe) {
        Scalar block[16];
        stream_rseek(zfp->stream, offset_bits(p));
        uint bits = zfp::codec<Scalar>::decode_block_2(zfp, block, 0);
        bits += stream_align(zfp->stream);
        uint words = word_size(bits);
        uint s = slot_size(words);
        usage += quantum_bytes() << s;
        for (uint i = 0; i < (1u << s); i++)
          slot[p + i] = s;
      }
      else if (p == null)
        zeros++;
    }
    // process free slots
    for (uint s = 0; s < sizes; s++) {
      for (offset p = head[s]; p != null; p = get_next_slot(p))
        for (uint i = 0; i < (1u << s); i++)
          slot[p + i] = 0x80 + s;
    }
    for (uint i = 0; i < 0x10000; i++)
      printf("%02x ", slot[i]);
    printf("\n");
    stream_reopen(zfp->stream, buffer, size);
    delete[] slot;
    printf("bytes=%zu vs %zu\n", bytes, usage);
    printf("words=%zu\n", totwords);
    printf("zeros=%zu\n", zeros);
  }
#endif

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
      const word* buffer = static_cast<const word*>(stream_data(zfp->stream));
      std::copy(buffer, buffer + words, data + offset_words(p));
    }
    pos[id] = p;
  }

  // decompress block with tile-local index 'id' and free compressed data
  void decompress(zfp_stream* zfp, Scalar* block, uint id, uchar shape = 0)
  {
    offset p = pos[id];
    pos[id] = cached;
    if (p == null) {
      // empty block; fill with zeros
      std::fill(block, block + 4 * 4, Scalar(0));
    }
    else {
      // save current buffer
      void* buffer = stream_data(zfp->stream);
      size_t size = stream_capacity(zfp->stream);
      // decompress block to cache
      stream_reopen(zfp->stream, data, capacity());
      stream_rseek(zfp->stream, offset_bits(p));
      uint bits = zfp::codec<Scalar>::decode_block_2(zfp, block, shape);
      bits += stream_align(zfp->stream);
      // free space occupied by compressed data
      size_t words = word_size(bits);
      deallocate(p, words);
      // restore buffer
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
