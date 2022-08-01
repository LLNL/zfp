#ifndef ZFP_INDEX_HPP
#define ZFP_INDEX_HPP

#include <algorithm>
#include "zfp/internal/array/memory.hpp"

namespace zfp {
namespace index {

// implicit block index (fixed-size blocks; 0 bits/block; 64-bit offsets) -----
class implicit {
public:
  // constructor
  implicit(size_t blocks) :
    bits_per_block(0)
  {
    resize(blocks);
  }

  // destructor
  ~implicit() {}

  // byte size of index data structure components indicated by mask
  size_t size_bytes(uint mask = ZFP_DATA_ALL) const
  {
    size_t size = 0;
    if (mask & ZFP_DATA_META)
      size += sizeof(*this);
    return size;
  }

  // range of offsets spanned by indexed data in bits
  bitstream_size range() const { return block_offset(blocks); }

  // bit size of given block
  size_t block_size(size_t /*block_index*/) const { return bits_per_block; }

  // bit offset of given block
  bitstream_offset block_offset(size_t block_index) const { return block_index * bits_per_block; }

  // reset index
  void clear() {}

  // resize index in number of blocks
  void resize(size_t blocks) { this->blocks = blocks; }

  // flush any buffered data
  void flush() {}

  // set bit size of all blocks
  void set_block_size(size_t size) { bits_per_block = size; }

  // set bit size of given block (ignored for performance reasons)
  void set_block_size(size_t /*block_index*/, size_t /*size*/) {}

  // does not support variable rate
  static bool has_variable_rate() { return false; }

protected:
  size_t blocks;         // number of blocks
  size_t bits_per_block; // fixed number of bits per block
};

// verbatim block index (64 bits/block; 64-bit offsets) -----------------------
class verbatim {
public:
  // constructor for given nbumber of blocks
  verbatim(size_t blocks) :
    data(0)
  {
    resize(blocks);
  }

  // destructor
  ~verbatim() { zfp::internal::deallocate(data); }

  // assignment operator--performs a deep copy
  verbatim& operator=(const verbatim& index)
  {
    if (this != &index)
      deep_copy(index);
    return *this;
  }

  // byte size of index data structure components indicated by mask
  size_t size_bytes(uint mask = ZFP_DATA_ALL) const
  {
    size_t size = 0;
    if (mask & ZFP_DATA_INDEX)
      size += capacity() * sizeof(*data);
    if (mask & ZFP_DATA_META)
      size += sizeof(*this);
    return size;
  }

  // range of offsets spanned by indexed data in bits
  bitstream_size range() const { return block_offset(blocks); }

  // bit size of given block
  size_t block_size(size_t block_index) const { return static_cast<size_t>(block_offset(block_index + 1) - block_offset(block_index)); }

  // bit offset of given block
  bitstream_offset block_offset(size_t block_index) const { return static_cast<bitstream_offset>(data[block_index]); }

  // reset index
  void clear() { block = 0; }

  // resize index in number of blocks
  void resize(size_t blocks)
  {
    this->blocks = blocks;
    zfp::internal::reallocate(data, capacity() * sizeof(*data));
    *data = 0;
    clear();
  }

  // flush any buffered data
  void flush() {}

  // set bit size of all blocks
  void set_block_size(size_t size)
  {
    clear();
    while (block < blocks)
      set_block_size(block, size);
    clear();
  }

  // set bit size of given block (in sequential order)
  void set_block_size(size_t block_index, size_t size)
  {
    if (block_index != block)
      throw zfp::exception("zfp index supports only sequential build");
    if (block == blocks)
      throw zfp::exception("zfp index overflow");
    data[block + 1] = data[block] + size;
    block++;
  }

  // supports variable rate
  static bool has_variable_rate() { return true; }

protected:
  // capacity of data array
  size_t capacity() const { return blocks + 1; }

  // make a deep copy of index
  void deep_copy(const verbatim& index)
  {
    zfp::internal::clone(data, index.data, index.capacity());
    blocks = index.blocks;
    block = index.block;
  }

  uint64* data;  // block offset array
  size_t blocks; // number of blocks
  size_t block;  // current block index
};

// hybrid block index (4 blocks/chunk; 24 bits/block; 44-bit offsets) ---------
class hybrid4 {
public:
  // constructor for given number of blocks
  hybrid4(size_t blocks) :
    data(0)
  {
    resize(blocks);
  }

  // destructor
  ~hybrid4() { zfp::internal::deallocate(data); }

  // assignment operator--performs a deep copy
  hybrid4& operator=(const hybrid4& index)
  {
    if (this != &index)
      deep_copy(index);
    return *this;
  }

  // byte size of index data structure components indicated by mask
  size_t size_bytes(uint mask = ZFP_DATA_ALL) const
  {
    size_t size = 0;
    if (mask & ZFP_DATA_INDEX)
      size += capacity() * sizeof(*data);
    if (mask & ZFP_DATA_META)
      size += sizeof(*this);
    return size;
  }

  // range of offsets spanned by indexed data in bits
  bitstream_size range() const { return end; }

  // bit size of given block
  size_t block_size(size_t block_index) const
  {
    size_t chunk = block_index / 4;
    size_t which = block_index % 4;
    return which == 3u
             ? static_cast<size_t>(block_offset(block_index + 1) - block_offset(block_index))
             : static_cast<size_t>(data[chunk].lo[which + 1] - data[chunk].lo[which]);
  }

  // bit offset of given block
  bitstream_offset block_offset(size_t block_index) const
  {
    // if index is being built, point offset to end
    if (block_index == block)
      return end;
    // index has already been built; decode offset
    size_t chunk = block_index / 4;
    size_t which = block_index % 4;
    return (bitstream_offset(data[chunk].hi) << shift) + data[chunk].lo[which];
  }

  // reset index
  void clear()
  {
    block = 0;
    ptr = 0;
    end = 0;
  }

  void resize(size_t blocks)
  {
    this->blocks = blocks;
    zfp::internal::reallocate(data, capacity() * sizeof(*data));
    clear();
  }

  // flush any buffered data
  void flush()
  {
    while (block & 0x3u)
      set_block_size(block, 0);
  }

  // set bit size of all blocks
  void set_block_size(size_t size)
  {
    clear();
    while (block < blocks)
      set_block_size(block, size);
    flush();
    clear();
  }

  // set bit size of given block (in sequential order)
  void set_block_size(size_t block_index, size_t size)
  {
    // ensure block_index is next in sequence
    if (block_index != block)
      throw zfp::exception("zfp index supports only sequential build");
    // ensure block index is within bounds, but allow 0-size blocks for padding 
    if (block >= blocks && size)
      throw zfp::exception("zfp index overflow");
    // ensure block size is valid
    if (size > ZFP_MAX_BITS)
      throw zfp::exception("zfp block size is too large for hybrid4 index");
    // advance end pointer
    end += size;
    // buffer chunk of 4 block sizes at a time
    size_t chunk = block / 4;
    size_t which = block % 4;
    buffer[which] = size;
    if (which == 3u) {
      // chunk is complete; encode it
      if (ptr >> (32 + shift))
        throw zfp::exception("zfp block offset is too large for hybrid4 index");
      // store high bits
      data[chunk].hi = static_cast<uint32>(ptr >> shift);
      bitstream_offset base = bitstream_offset(data[chunk].hi) << shift;
      // store low bits
      for (uint k = 0; k < 4; k++) {
        data[chunk].lo[k] = static_cast<uint16>(ptr - base);
        ptr += buffer[k];
      }
    }
    block++;
  }

  // supports variable rate
  static bool has_variable_rate() { return true; }

protected:
  // chunk record encoding 4 block offsets
  typedef struct {
    uint32 hi;    // 32 most significant bits of 44-bit base offset
    uint16 lo[4]; // 16-bit offsets from base
  } record;

  // capacity of data array
  size_t capacity() const { return (blocks + 3) / 4; }

  // make a deep copy of index
  void deep_copy(const hybrid4& index)
  {
    zfp::internal::clone(data, index.data, index.capacity());
    blocks = index.blocks;
    block = index.block;
    ptr = index.ptr;
    end = index.end;
    std::copy(index.buffer, index.buffer + 4, buffer);
  }

  static const uint shift = 12; // number of bits to shift hi bits

  record* data;         // block offset array
  size_t blocks;        // number of blocks
  size_t block;         // current block index
  bitstream_offset end; // offset to last block
  bitstream_offset ptr; // offset to current chunk of blocks
  size_t buffer[4];     // bit sizes 4 blocks to be stored together
};

// hybrid block index (8 blocks/chunk; 16 bits/block; 86-14dims bit offsets) --
template <uint dims>
class hybrid8 {
public:
  // constructor for given number of blocks
  hybrid8(size_t blocks) :
    data(0)
  {
    resize(blocks);
  }

  // destructor
  ~hybrid8() { zfp::internal::deallocate(data); }

  // assignment operator--performs a deep copy
  hybrid8& operator=(const hybrid8& index)
  {
    if (this != &index)
      deep_copy(index);
    return *this;
  }

  // byte size of index data structure components indicated by mask
  size_t size_bytes(uint mask = ZFP_DATA_ALL) const
  {
    size_t size = 0;
    if (mask & ZFP_DATA_INDEX)
      size += capacity() * sizeof(*data);
    if (mask & ZFP_DATA_META)
      size += sizeof(*this);
    return size;
  }

  // range of offsets spanned by indexed data in bits
  bitstream_size range() const { return end; }

  // bit size of given block
  size_t block_size(size_t block_index) const
  {
    size_t chunk = block_index / 8;
    size_t which = block_index % 8;
    return which == 7u
             ? static_cast<size_t>(block_offset(block_index + 1) - block_offset(block_index))
             : size(data[2 * chunk + 0], data[2 * chunk + 1], static_cast<uint>(which));
  }

  // bit offset of given block
  bitstream_offset block_offset(size_t block_index) const
  {
    // if index is being built, point offset to end
    if (block_index == block)
      return end;
    // index has already been built; decode offset
    size_t chunk = block_index / 8;
    size_t which = block_index % 8;
    return offset(data[2 * chunk + 0], data[2 * chunk + 1], static_cast<uint>(which));
  }

  // reset index
  void clear()
  {
    block = 0;
    ptr = 0;
    end = 0;
  }

  void resize(size_t blocks)
  {
    this->blocks = blocks;
    zfp::internal::reallocate(data, capacity() * sizeof(*data));
    clear();
  }

  // flush any buffered data
  void flush()
  {
    while (block & 0x7u)
      set_block_size(block, 0);
  }

  // set bit size of all blocks
  void set_block_size(size_t size)
  {
    clear();
    while (block < blocks)
      set_block_size(block, size);
    flush();
    clear();
  }

  // set bit size of given block (in sequential order)
  void set_block_size(size_t block_index, size_t size)
  {
    // ensure block_index is next in sequence
    if (block_index != block)
      throw zfp::exception("zfp index supports only sequential build");
    // ensure block index is within bounds, but allow 0-size blocks for padding 
    if (block >= blocks && size)
      throw zfp::exception("zfp index overflow");
    // ensure block size is valid
    if (size >> (hbits + lbits))
      throw zfp::exception("zfp block size is too large for hybrid8 index");
    // advance end pointer
    end += size;
    // buffer chunk of 8 block sizes at a time
    size_t chunk = block / 8;
    size_t which = block % 8;
    buffer[which] = size;
    if (which == 7u) {
      // partition chunk offset into low and high bits
      uint64 h = ptr >> lbits;
      uint64 l = ptr - (h << lbits);
      uint64 hi = h << (7 * hbits);
      uint64 lo = l << (7 * lbits);
      // make sure base offset does not overflow
      if ((hi >> (7 * hbits)) != h)
        throw zfp::exception("zfp block offset is too large for hybrid8 index");
      // store sizes of blocks 0-6
      for (uint k = 0; k < 7; k++) {
        size = buffer[k];
        ptr += size;
        // partition block size into hbits high and lbits low bits
        h = size >> lbits;
        l = size - (h << lbits);
        hi += h << ((6 - k) * hbits);
        lo += l << ((6 - k) * lbits);
      }
      ptr += buffer[7];
      data[2 * chunk + 0] = hi;
      data[2 * chunk + 1] = lo;
    }
    block++;
  }

  // supports variable rate
  static bool has_variable_rate() { return true; }

protected:
  // capacity of data array
  size_t capacity() const { return 2 * ((blocks + 7) / 8); }

  // make a deep copy of index
  void deep_copy(const hybrid8& index)
  {
    zfp::internal::clone(data, index.data, index.capacity());
    blocks = index.blocks;
    block = index.block;
    ptr = index.ptr;
    end = index.end;
    std::copy(index.buffer, index.buffer + 8, buffer);
  }

  // kth size in chunk, 0 <= k <= 6
  static size_t size(uint64 h, uint64 l, uint k)
  {
    // extract high and low bits
    h >>= (6 - k) * hbits; h &= (UINT64C(1) << hbits) - 1;
    l >>= (6 - k) * lbits; l &= (UINT64C(1) << lbits) - 1;
    // combine base offset with high and low bits
    return static_cast<size_t>((h << lbits) + l);
  }

  // kth offset in chunk, 0 <= k <= 7
  static bitstream_offset offset(uint64 h, uint64 l, uint k)
  {
    // extract all but lowest (8 * hbits) bits
    uint64 base = h >> (8 * hbits);
    h -= base << (8 * hbits);
    // add LSBs of base offset and k block sizes
    h = hsum(h >> ((7 - k) * hbits));
    l = lsum(l >> ((7 - k) * lbits));
    // combine base offset with high and low bits
    return static_cast<bitstream_offset>((((base << hbits) + h) << lbits) + l);
  }

  // sum of (up to) eight packed 8-bit numbers (efficient version of sum8)
  static uint64 lsum(uint64 x)
  {
    // reduce in parallel
    uint64 y = x & UINT64C(0xff00ff00ff00ff00);
    x -= y;
    x += y >> 8;
    x += x >> 16;
    x += x >> 32;
    return x & UINT64C(0xffff);
  }

  // sum of (up to) eight packed h-bit numbers
  static uint64 hsum(uint64 x) { return sum8(x, hbits); }

  // compute sum of eight packed n-bit values (1 <= n <= 8)
  static uint64 sum8(uint64 x, uint n)
  {
    // bit masks for extracting terms of sums
    uint64 m3 = ~UINT64C(0) << (4 * n);
    uint64 m2 = m3 ^ (m3 << (4 * n));
    uint64 m1 = m2 ^ (m2 >> (2 * n));
    uint64 m0 = m1 ^ (m1 >> (1 * n));
    uint64 y;
    // perform summations in parallel
    y = x & m0; x -= y; x += y >> n; n *= 2; // four summations
    y = x & m1; x -= y; x += y >> n; n *= 2; // two summations
    y = x & m2; x -= y; x += y >> n; n *= 2; // final summation
    return x;
  }

  static const uint lbits = 8;              // 64 bits partitioned into 8
  static const uint hbits = 2 * (dims - 1); // log2(4^d * maxprec / 2^lbits)

  uint64* data;         // block offset array
  size_t blocks;        // number of blocks
  size_t block;         // current block index
  bitstream_offset end; // offset to last block
  bitstream_offset ptr; // offset to current set of blocks
  size_t buffer[8];     // sizes of 8 blocks to be stored together
};

} // index
} // zfp

#endif
