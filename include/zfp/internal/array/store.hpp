#ifndef ZFP_STORE_HPP
#define ZFP_STORE_HPP

#include <climits>
#include <cmath>
#include "zfp/internal/array/memory.hpp"

namespace zfp {
namespace internal {

// base class for block store
template <class Codec, class Index>
class BlockStore {
public:
  // compression mode
  zfp_mode mode() const { return codec.mode(); }

  // rate in bits per value (fixed-rate mode only)
  double rate() const { return codec.rate(); }

  // precision in uncompressed bits per value (fixed-precision mode only)
  uint precision() const { return codec.precision(); }

  // accuracy as absolute error tolerance (fixed-accuracy mode only)
  double accuracy() const { return codec.accuracy(); }

  // compression parameters (all compression modes)
  void params(uint* minbits, uint* maxbits, uint* maxprec, int* minexp) const { codec.params(minbits, maxbits, maxprec, minexp); }

  // enable reversible (lossless) mode
  void set_reversible()
  {
    set_variable_rate();
    codec.set_reversible();
    clear();
  }

  // set fixed rate in compressed bits per value with optional word alignment
  double set_rate(double rate, bool align)
  {
    rate = codec.set_rate(rate, align);
    uint maxbits;
    codec.params(0, &maxbits, 0, 0);
    index.set_block_size(maxbits);
    alloc(true);
    return rate;
  }

  // set precision in uncompressed bits per value
  uint set_precision(uint precision)
  {
    set_variable_rate();
    precision = codec.set_precision(precision);
    clear();
    return precision;
  }

  // set accuracy as absolute error tolerance
  double set_accuracy(double tolerance)
  {
    set_variable_rate();
    tolerance = codec.set_accuracy(tolerance);
    clear();
    return tolerance;
  }

  // set expert mode compression parameters
  bool set_params(uint minbits, uint maxbits, uint maxprec, int minexp)
  {
    if (minbits != maxbits)
      set_variable_rate();
    bool status = codec.set_params(minbits, maxbits, maxprec, minexp);
    clear();
    return status;
  }

  // set compression mode and parameters
  void set_config(const zfp_config& config)
  {
    switch (config.mode) {
      case zfp_mode_reversible:
        set_reversible();
        break;
      case zfp_mode_fixed_rate:
        if (config.arg.rate < 0)
          set_rate(-config.arg.rate, true);
        else
          set_rate(+config.arg.rate, false);
        break;
      case zfp_mode_fixed_precision:
        set_precision(config.arg.precision);
        break;
      case zfp_mode_fixed_accuracy:
        set_accuracy(config.arg.tolerance);
        break;
      case zfp_mode_expert:
        set_params(config.arg.expert.minbits, config.arg.expert.maxbits, config.arg.expert.maxprec, config.arg.expert.minexp);
        break;
      default:
        throw zfp::exception("zfp compression mode not supported by array");
    }
  }

  // clear store and reallocate memory for buffer
  void clear()
  {
    index.clear();
    alloc(true);
  }

  // flush any buffered block index data
  void flush() { index.flush(); }

  // shrink buffer to match size of compressed data
  void compact()
  {
    size_t size = zfp::internal::round_up(index.range(), codec.alignment() * CHAR_BIT) / CHAR_BIT;
    if (bytes > size) {
      codec.close();
      zfp::internal::reallocate_aligned(data, size, ZFP_MEMORY_ALIGNMENT, bytes);
      bytes = size;
      codec.open(data, bytes);
    }
  }

  // increment private view reference count (for thread safety)
  void reference()
  {
#ifdef _OPENMP
    #pragma omp critical(references)
    {
      references++;
      codec.set_thread_safety(references > 1);
    }
#endif
  }

  // decrement private view reference count (for thread safety)
  void unreference()
  {
#ifdef _OPENMP
    #pragma omp critical(references)
    {
      references--;
      codec.set_thread_safety(references > 1);
    }
#endif
  }

  // byte size of store data structure components indicated by mask
  virtual size_t size_bytes(uint mask = ZFP_DATA_ALL) const
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

protected:
  // protected default constructor
  BlockStore() :
    data(0),
    bytes(0),
    references(0),
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
    if (!index.has_variable_rate())
      throw zfp::exception("zfp index does not support variable rate");
  }

  // perform a deep copy
  void deep_copy(const BlockStore& s)
  {
    free();
    zfp::internal::clone_aligned(data, s.data, s.bytes, ZFP_MEMORY_ALIGNMENT);
    bytes = s.bytes;
    references = s.references;
    index = s.index;
    codec = s.codec;
    codec.open(data, bytes);
  }

  // allocate memory for block store
  void alloc(bool clear)
  {
    free();
    bytes = buffer_size();
    zfp::internal::reallocate_aligned(data, bytes, ZFP_MEMORY_ALIGNMENT);
    if (clear)
      std::fill(static_cast<uchar*>(data), static_cast<uchar*>(data) + bytes, uchar(0));
    codec.open(data, bytes);
  }

  // free block store
  void free()
  {
    if (data) {
      zfp::internal::deallocate_aligned(data);
      data = 0;
      bytes = 0;
      codec.close();
    }
  }

  // bit offset to block store
  bitstream_offset offset(size_t block_index) const { return index.block_offset(block_index); }

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

  void* data;        // pointer to compressed blocks
  size_t bytes;      // compressed data size
  size_t references; // private view references to array (for thread safety)
  Index index;       // block index (size and offset)
  Codec codec;       // compression codec
};

} // internal
} // zfp

#endif
