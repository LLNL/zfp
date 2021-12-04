#ifndef ZFP_TILE_STORE_H
#define ZFP_TILE_STORE_H

#include <climits>
#include "zfp/exception.h"
#include "zfp/memory.h"

namespace zfp {
namespace internal {

// base class for tile store
template <class Codec>
class TileStore {
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
    codec.set_reversible();
    clear();
  }

  // set fixed rate in compressed bits per value with optional word alignment
  double set_rate(double rate, bool align)
  {
    rate = codec.set_rate(rate, align);
    clear();
    return rate;
  }

  // TODO: limit variable-rate modes to not overflow maximum slot size

  // set precision in uncompressed bits per value
  uint set_precision(uint precision)
  {
    precision = codec.set_precision(precision);
    clear();
    return precision;
  }

  // set accuracy as absolute error tolerance
  double set_accuracy(double tolerance)
  {
    tolerance = codec.set_accuracy(tolerance);
    clear();
    return tolerance;
  }

  // set expert mode compression parameters
  bool set_params(uint minbits, uint maxbits, uint maxprec, int minexp)
  {
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
    // TODO: clear each tile; pure virtual here?
    alloc();
  }

  // flush any buffered block index data
  void flush() {}

  // shrink buffer to match size of compressed data
  void compact()
  {
    // TODO: compact each tile; pure virtual here?
    abort();
  }

  // byte size of store data structure components indicated by mask
  virtual size_t size_bytes(uint mask = ZFP_DATA_ALL) const
  {
    size_t size = 0;
    size += codec.size_bytes(mask);
    if (mask & ZFP_DATA_META) {
      size += sizeof(*this);
      if (buffer)
        size += buffer_size();
    }
    return size;
  }

#if 0
  // number of bytes of compressed data
  size_t compressed_size() const { return 0; }

  // pointer to compressed data for read or write access
  void* compressed_data() const { return 0; }
#endif

protected:
  // protected default constructor
  TileStore() :
    buffer(0)
  {}

  // destructor
  virtual ~TileStore() { TileStore::free(); }

  // buffer size in bytes needed for current codec settings
  virtual size_t buffer_size() const = 0;

  // number of elements per block
  virtual size_t block_size() const = 0;

  // total number of blocks
  virtual size_t blocks() const = 0;

  // total number of tiles
  virtual size_t tiles() const = 0;

  // allocate memory for single-block buffer
  virtual void alloc()
  {
    TileStore::free();
    size_t bytes = buffer_size();
    buffer = zfp::allocate_aligned(bytes, ZFP_MEMORY_ALIGNMENT);
    codec.open(buffer, bytes);
  }

  // free single-block buffer
  virtual void free()
  {
    if (buffer) {
      zfp::deallocate_aligned(buffer);
      buffer = 0;
      codec.close();
    }
  }

  // perform a deep copy
  void deep_copy(const TileStore& /*s*/)
  {
    // TODO: copy each tile
    abort();
#if 0
    free();
    zfp::clone_aligned(data, s.data, s.bytes, ZFP_MEMORY_ALIGNMENT);
    bytes = s.bytes;
    index = s.index;
    codec = s.codec;
    codec.open(data, bytes);
#endif
  }

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

  void* buffer; // buffer sufficient for a single compressed block
  Codec codec;  // compression codec
};

} // internal
} // zfp

#endif
