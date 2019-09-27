#ifndef ZFP_TILE2_H
#define ZFP_TILE2_H

#include "tile.h"

namespace zfp {

// tile of 64x64 variable-rate 4x4 blocks
template <typename Scalar>
class Tile2 : public Tile {
public:
  Tile2(uint minbits = 64) :
    Tile(minbits)
  {}

#ifdef DEBUG
  void fragmentation(zfp_stream* zfp, FILE* file = stderr) const
  {
    uchar* slot = new uchar[0x10000];
    std::fill(slot, slot + 0x10000, 0xff);
    size_t usage = 0;
    size_t zeros = 0;
    // process compressed blocks
    for (uint b = 0; b < blocks; b++) {
      offset p = pos[b];
      if (p == null)
        zeros++;
      else if (p != cached) {
        Scalar block[block_size];
        uint bits = decode_block(zfp, block, p);
        uint words = word_size(bits);
        uint s = slot_size(words);
        usage += quantum_bytes() << s;
        for (uint i = 0; i < (1u << s); i++)
          slot[p + i] = s;
      }
    }
    // process free slots
    for (uint s = 0; s < sizes; s++) {
      for (offset p = head[s]; p != null; p = get_next_slot(p))
        for (uint i = 0; i < (1u << s); i++)
          slot[p + i] = 0x80 + s;
    }
    // print 
    for (uint i = 0; i < 0x10000; i++)
      fprintf(file, "%02x ", slot[i]);
    fprintf(file, "\n");
    delete[] slot;
    fprintf(file, "bytes=%zu vs %zu\n", bytes, usage);
    fprintf(file, "zeros=%zu\n", zeros);
  }
#endif

  // allocate compressed data and compress block with tile-local index 'id'
  void compress(zfp_stream* zfp, const Scalar* block, uint id, uchar shape = 0)
  {
    offset p = null;
    // compress block
    uint bits = encode_block(zfp, block, shape);
    // if block is empty, no storage is needed; otherwise, find space for it
    if (bits > 1u) {
      // allocate memory for compressed block
      size_t words = word_size(bits);
      assert(words <= (quantum_words() << 4));
      p = allocate(words);
      assert(p != null);
      assert(p != cached);
      // copy compressed data to persistent storage
      const word* buffer = static_cast<const word*>(stream_data(zfp->stream));
      std::copy(buffer, buffer + words, data + offset_words(p));
    }
    pos[id] = p;
  }

  // decompress block with tile-local index 'id' and free compressed data
  void decompress(zfp_stream* zfp, Scalar* block, uint id, uchar shape = 0, bool cache = true)
  {
    offset p = pos[id];
    assert(p != cached);
    if (cache)
      pos[id] = cached;
    if (p == null) {
      // empty block; fill with zeros
      std::fill(block, block + block_size, Scalar(0));
    }
    else {
      // decompress block
      uint bits = decode_block(zfp, block, p, shape);
      if (cache) {
        // free space occupied by compressed data
        size_t words = word_size(bits);
        deallocate(p, words);
      }
    }
  }

  // storage class for block with given id
  Tile::storage block_storage(zfp_stream* zfp, uint id, uchar shape = 0) const
  {
    offset p = pos[id];
    if (p == null)
      return storage_empty;
    if (p == cached)
      return storage_cached;
    if (p & 1u)
      return storage_xs;
    Scalar block[block_size];
    uint bits = decode_block(zfp, block, p, shape);
    return storage(storage_xs + slot_size(word_size(bits)));
  }

  static const uint bx = 64;            // number of blocks per tile along x
  static const uint by = 64;            // number of blocks per tile along y
  static const uint block_size = 4 * 4; // number of scalars per block

protected:
  // compress block to beginning of stream and return its storage size in bits
  uint encode_block(zfp_stream* zfp, const Scalar* block, uchar shape = 0) const
  {
    // compress block to temporary storage
    stream_rewind(zfp->stream);
    uint bits = zfp::codec<Scalar>::encode_block_2(zfp, block, shape);
    // if block is non-empty (i.e., stored), make sure stream is flushed
    if (bits > 1u)
      bits += stream_flush(zfp->stream);
    return bits;
  }

  // decompress block stored at offset p
  uint decode_block(zfp_stream* zfp, Scalar* block, offset p, uchar shape = 0) const
  {
    // save current buffer
    void* buffer = stream_data(zfp->stream);
    size_t size = stream_capacity(zfp->stream);
    // decompress block to cache
    stream_reopen(zfp->stream, data, capacity());
    stream_rseek(zfp->stream, offset_bits(p));
    uint bits = zfp::codec<Scalar>::decode_block_2(zfp, block, shape);
    bits += stream_align(zfp->stream);
    // restore buffer
    stream_reopen(zfp->stream, buffer, size);
    return bits;
  }
};

}

#endif
