#ifndef ZFP_STORE_H
#define ZFP_STORE_H

#include <climits>

namespace zfp {

// base class for block store
class BlockStore {
public:
  // number of bytes of compressed data
  size_t compressed_size() const { return bytes; }

  // pointer to compressed data for read or write access
  void* compressed_data() const { return data; }

protected:
  // protected default constructor
  BlockStore() :
    bits_per_block(0),
    data(0),
    bytes(0)
  {}

  // perform a deep copy
  void deep_copy(const BlockStore& s)
  {
    bits_per_block = s.bits_per_block;
    bytes = s.bytes;
    zfp::clone_aligned(data, s.data, s.bytes, ZFP_MEMORY_ALIGNMENT);
  }

  // allocate memory for persistent block store
  void alloc(size_t blocks, bool clear)
  {
    size_t words = (blocks * bits_per_block + CHAR_BIT * sizeof(uint64) - 1) / (CHAR_BIT * sizeof(uint64));
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
  size_t offset(size_t block_index) const { return block_index * bits_per_block; }

  // shape 0 <= m <= 3 of block containing index i, 0 <= i <= n - 1
  static uint shape_code(size_t i, size_t n)
  {
    // handle partial blocks efficiently using no conditionals
    size_t m = i ^ n;               // m < 4 iff partial block
    m -= 4;                         // m < 0 iff partial block
    m >>= CHAR_BIT * sizeof(m) - 2; // m = 3 iff partial block; otherwise m = 0
    m &= -n;                        // m = 4 - w
    return static_cast<uint>(m);
  }

  uint bits_per_block; // number of bits of compressed storage per block
  void* data;          // pointer to compressed blocks
  size_t bytes;        // compressed data size
};

}

#endif
