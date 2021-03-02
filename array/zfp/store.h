#ifndef ZFP_STORE_H
#define ZFP_STORE_H

#include <climits>
#include <cmath>
#include "zfp/memory.h"
#include "zfp/index.h"

namespace zfp {

// base class for block store
template <class Codec, class Index = zfp::internal::ImplicitIndex>
class BlockStore {
public:
  // compression mode
  zfp_mode mode() const { return codec.mode(); }

  // rate in bits per value (supported in all mode)
  double rate() const { return codec.rate(); }

  // precision in uncompressed bits per value (fixed-precision mode only)
  uint precision() const { return codec.precision(); }

  // accuracy as absolute error tolerance (fixed-accuracy mode only)
  double accuracy() const { return codec.accuracy(); }

  // set fixed rate in compressed bits per value
  double set_rate(double rate)
  {
    rate = codec.set_rate(rate);
    index.set_block_size(codec.maxbits());
    alloc(true);
    return rate;
  }

  // set precision in uncompressed bits per value
  uint set_precision(uint precision)
  {
    set_variable_rate();
    precision = codec.set_precision(precision);
    index.clear();
    alloc(true);
    return precision;
  }

  // set accuracy as absolute error tolerance
  double set_accuracy(double tolerance)
  {
    set_variable_rate();
    tolerance = codec.set_accuracy(tolerance);
    index.clear();
    alloc(true);
    return tolerance;
  }

  // enable reversible (lossless) mode
  void set_reversible()
  {
    set_variable_rate();
    codec.set_reversible();
    index.clear();
    alloc(true);
  }

  // set compression mode and parameters
  void set_config(const zfp_config& config)
  {
    switch (config.mode) {
      case zfp_mode_fixed_rate:
        set_rate(std::fabs(config.arg.rate));
        break;
      case zfp_mode_fixed_precision:
        set_precision(config.arg.precision);
        break;
      case zfp_mode_fixed_accuracy:
        set_accuracy(config.arg.tolerance);
        break;
      case zfp_mode_reversible:
        set_reversible();
        break;
      default:
        throw zfp::exception("zfp compression mode not supported by array");
    }
  }

  // shrink buffer to match size of compressed data
  void compact()
  {
    size_t size = zfp::round_up(index.data_size(), codec.alignment() * CHAR_BIT) / CHAR_BIT;
    if (bytes > size) {
      zfp::reallocate_aligned(data, size, ZFP_MEMORY_ALIGNMENT, bytes);
      bytes = size;
    }
  }

  // byte size of store data structure components indicated by mask
  size_t size_bytes(uint mask = ZFP_DATA_ALL) const
  {
    size_t size = 0;
    size += index.size_bytes(mask);
    size += codec.size_bytes(mask);
    if (mask & ZFP_DATA_PAYLOAD)
      size += bytes;
    if (mask & ZFP_DATA_META)
      size += sizeof(*this);
    return size;
  }

  // number of bytes of compressed data
  size_t compressed_size() const { return bytes; }

  // pointer to compressed data for read or write access
  void* compressed_data() const { return data; }

  // reset block index
  void clear_index() { index.clear(); }

  // flush any buffered block index data
  void flush_index() { index.flush(); }

protected:
  // protected default constructor
  BlockStore() :
    data(0),
    bytes(0),
    index(0)
  {}

  // destructor
  virtual ~BlockStore() { free(); }

  // buffer size in bytes needed for current codec settings
  virtual size_t buffer_size() const = 0;

  // number of elements per block
  virtual size_t block_size() const = 0;

  // total number of blocks
  virtual size_t blocks() const = 0;

  // ensure variable rate is supported
  void set_variable_rate()
  {
    if (!index.variable_rate())
      throw zfp::exception("zfp index does not support variable rate");
  }

  // perform a deep copy
  void deep_copy(const BlockStore& s)
  {
    free();
    zfp::clone_aligned(data, s.data, s.bytes, ZFP_MEMORY_ALIGNMENT);
    bytes = s.bytes;
    index = s.index;
    codec = s.codec;
    codec.open(data, bytes);
  }

  // allocate memory for block store
  void alloc(bool clear)
  {
    free();
    bytes = buffer_size();
    zfp::reallocate_aligned(data, bytes, ZFP_MEMORY_ALIGNMENT);
    if (clear)
      std::fill(static_cast<uint8*>(data), static_cast<uint8*>(data) + bytes, uint8(0));
    codec.open(data, bytes);
  }

  // free block store
  void free()
  {
    if (data) {
      zfp::deallocate_aligned(data);
      data = 0;
      bytes = 0;
      codec.close();
    }
  }

  // bit offset to block store
  size_t offset(size_t block_index) const { return index.block_offset(block_index); }

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

  void* data;   // pointer to compressed blocks
  size_t bytes; // compressed data size
  Index index;  // block index (size and offset)
  Codec codec;  // compression codec
};

}

#endif
