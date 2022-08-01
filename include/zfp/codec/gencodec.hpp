#ifndef ZFP_GENERIC_CODEC_HPP
#define ZFP_GENERIC_CODEC_HPP

// This CODEC allows interfacing with the zfp::array classes via a user-facing
// scalar type, ExternalType (e.g., double), while storing data in memory using
// a possibly less precise scalar type, InternalType (e.g., float).  Using
// zfp's caching mechanism, blocks of data may reside for some time in cache
// as ExternalType.  This potentially allows a sequence of more precise
// operations to be performed on the data before it is down-converted to
// InternalType and stored to memory.  When ExternalType = InternalType, this
// CODEC allows defining arrays that support the full zfp array API but use
// uncompressed storage.  To use this CODEC, pass it as the Codec template
// parameter to a zfp::array class of matching dimensionality.

#include <algorithm>
#include <climits>
#include <cstring>
#include "zfp.h"
#include "zfp/internal/array/memory.hpp"
#include "zfp/internal/array/traits.hpp"

namespace zfp {
namespace codec {

// abstract base class for storing 1D-4D uncompressed blocks of scalars
template <
  uint dims,                           // data dimensionality (1-4)
  typename ExternalType,               // scalar type exposed through array API
  typename InternalType = ExternalType // scalar type used for storage
>
class generic_base {
protected:
  // default constructor
  generic_base() :
    bytes(0),
    buffer(0)
  {}

public:
  // conservative buffer size for current codec settings
  size_t buffer_size(const zfp_field* field) const
  {
    return zfp_field_blocks(field) * block_size * sizeof(InternalType);
  }

  // open 
  void open(void* data, size_t size)
  {
    bytes = size;
    buffer = static_cast<InternalType*>(data);
  }

  // close bit stream
  void close()
  {
    bytes = 0;
    buffer = 0;
  }

  // pointer to beginning of bit stream
  void* data() const { return static_cast<void*>(buffer); }

  // compression mode
  zfp_mode mode() const { return zfp_mode_fixed_rate; }

  // rate in compressed bits/value (equals precision)
  double rate() const { return static_cast<double>(precision()); }

  // precision in uncompressed bits/value
  uint precision() const { return internal_size_bits; }

  // accuracy as absolute error tolerance (unsupported)
  double accuracy() const { return -1; }

  // compression parameters (all compression modes)
  void params(uint* minbits, uint* maxbits, uint* maxprec, int* minexp) const
  {
    if (minbits)
      *minbits = block_size_bits;
    if (maxbits)
      *maxbits = block_size_bits;
    if (maxprec)
      *maxprec = precision();
    if (minexp)
      *minexp = ZFP_MIN_EXP;
  }

  // enable reversible (lossless) mode
  void set_reversible()
  {
    throw zfp::exception("zfp generic codec does not support reversible mode");
  }

  // set rate in compressed bits/value (equals precision)
  double set_rate(double rate, bool)
  {
    return static_cast<double>(set_precision(static_cast<uint>(rate)));
  }

  // set precision in uncompressed bits/value (must equal InternalType width)
  uint set_precision(uint precision)
  {
    if (precision != internal_size_bits)
      throw zfp::exception("zfp generic codec precision mismatch");
    return precision;
  }

  // set accuracy as absolute error tolerance
  double set_accuracy(double)
  {
    throw zfp::exception("zfp generic codec does not support fixed-accuracy mode");
    return -1;
  }

  // set expert mode parameters
  bool set_params(uint, uint, uint, int)
  {
    throw zfp::exception("zfp generic codec does not support expert mode");
    return false;
  }

  // set thread safety mode (not required by this codec)
  void set_thread_safety(bool) {}

  // byte size of codec data structure components indicated by mask
  size_t size_bytes(uint mask = ZFP_DATA_ALL) const
  {
    size_t size = 0;
    if (mask & ZFP_DATA_META)
      size += sizeof(*this);
    return size;
  }

  // unit of allocated data in bytes
  static size_t alignment() { return sizeof(InternalType); }

  static const zfp_type type = zfp::internal::trait<ExternalType>::type; // scalar type

  // zfp::codec::generic_base::header class for array (de)serialization
  #include "zfp/internal/codec/genheader.hpp"

protected:
  // pointer to beginning of block
  InternalType* begin(bitstream_offset offset) const
  {
    if (offset % internal_size_bits)
      throw zfp::exception("zfp generic codec bit offset alignment error");
    return buffer + offset / internal_size_bits;
  }

  // store full contiguous block to memory
  size_t encode_block(bitstream_offset offset, const ExternalType* block) const
  {
    InternalType* ptr = begin(offset);
    for (size_t n = block_size; n--;)
      *ptr++ = static_cast<InternalType>(*block++);
    return block_size_bits;
  }

  // load full contiguous block from memory
  size_t decode_block(bitstream_offset offset, ExternalType* block) const
  {
    const InternalType* ptr = begin(offset);
    for (size_t n = block_size; n--;)
      *block++ = static_cast<ExternalType>(*ptr++);
    return block_size_bits;
  }

  // constants associated with template arguments
  static const size_t internal_size_bits = sizeof(InternalType) * CHAR_BIT;
  static const size_t block_size = 1u << (2 * dims);
  static const size_t block_size_bits = block_size * internal_size_bits;

  size_t bytes;         // number of bytes of storage
  InternalType* buffer; // pointer to storage managed by block store
};

// 1D codec
template <typename ExternalType, typename InternalType = ExternalType>
class generic1 : public generic_base<1, ExternalType, InternalType> {
public:
  // encode contiguous 1D block
  size_t encode_block(bitstream_offset offset, uint shape, const ExternalType* block) const
  {
    return shape ? encode_block_strided(offset, shape, block, 1)
                 : encode_block(offset, block);
  }

  // decode contiguous 1D block
  size_t decode_block(bitstream_offset offset, uint shape, ExternalType* block) const
  {
    return shape ? decode_block_strided(offset, shape, block, 1)
                 : decode_block(offset, block);
  }

  // encode 1D block from strided storage
  size_t encode_block_strided(bitstream_offset offset, uint shape, const ExternalType* p, ptrdiff_t sx) const
  {
    InternalType* q = begin(offset);
    size_t nx = 4;
    if (shape) {
      nx -= shape & 3u; shape >>= 2;
    }
    for (size_t x = 0; x < nx; x++, p += sx, q++)
      *q = static_cast<InternalType>(*p);
    return block_size_bits;
  }

  // decode 1D block to strided storage
  size_t decode_block_strided(bitstream_offset offset, uint shape, ExternalType* p, ptrdiff_t sx) const
  {
    const InternalType* q = begin(offset);
    size_t nx = 4;
    if (shape) {
      nx -= shape & 3u; shape >>= 2;
    }
    for (size_t x = 0; x < nx; x++, p += sx, q++)
      *p = static_cast<ExternalType>(*q);
    return block_size_bits;
  }

protected:
  using generic_base<1, ExternalType, InternalType>::begin;
  using generic_base<1, ExternalType, InternalType>::encode_block;
  using generic_base<1, ExternalType, InternalType>::decode_block;
  using generic_base<1, ExternalType, InternalType>::block_size_bits;
};

// 2D codec
template <typename ExternalType, typename InternalType = ExternalType>
class generic2 : public generic_base<2, ExternalType, InternalType> {
public:
  // encode contiguous 2D block
  size_t encode_block(bitstream_offset offset, uint shape, const ExternalType* block) const
  {
    return shape ? encode_block_strided(offset, shape, block, 1, 4)
                 : encode_block(offset, block);
  }

  // decode contiguous 2D block
  size_t decode_block(bitstream_offset offset, uint shape, ExternalType* block) const
  {
    return shape ? decode_block_strided(offset, shape, block, 1, 4)
                 : decode_block(offset, block);
  }

  // encode 2D block from strided storage
  size_t encode_block_strided(bitstream_offset offset, uint shape, const ExternalType* p, ptrdiff_t sx, ptrdiff_t sy) const
  {
    InternalType* q = begin(offset);
    size_t nx = 4;
    size_t ny = 4;
    if (shape) {
      nx -= shape & 3u; shape >>= 2;
      ny -= shape & 3u; shape >>= 2;
    }
    for (size_t y = 0; y < ny; y++, p += sy - (ptrdiff_t)nx * sx, q += 4 - nx)
      for (size_t x = 0; x < nx; x++, p += sx, q++)
        *q = static_cast<InternalType>(*p);
    return block_size_bits;
  }

  // decode 2D block to strided storage
  size_t decode_block_strided(bitstream_offset offset, uint shape, ExternalType* p, ptrdiff_t sx, ptrdiff_t sy) const
  {
    const InternalType* q = begin(offset);
    size_t nx = 4;
    size_t ny = 4;
    if (shape) {
      nx -= shape & 3u; shape >>= 2;
      ny -= shape & 3u; shape >>= 2;
    }
    for (size_t y = 0; y < ny; y++, p += sy - (ptrdiff_t)nx * sx, q += 4 - nx)
      for (size_t x = 0; x < nx; x++, p += sx, q++)
        *p = static_cast<ExternalType>(*q);
    return block_size_bits;
  }

protected:
  using generic_base<2, ExternalType, InternalType>::begin;
  using generic_base<2, ExternalType, InternalType>::encode_block;
  using generic_base<2, ExternalType, InternalType>::decode_block;
  using generic_base<2, ExternalType, InternalType>::block_size_bits;
};

// 3D codec
template <typename ExternalType, typename InternalType = ExternalType>
class generic3 : public generic_base<3, ExternalType, InternalType> {
public:
  // encode contiguous 3D block
  size_t encode_block(bitstream_offset offset, uint shape, const ExternalType* block) const
  {
    return shape ? encode_block_strided(offset, shape, block, 1, 4, 16)
                 : encode_block(offset, block);
  }

  // decode contiguous 3D block
  size_t decode_block(bitstream_offset offset, uint shape, ExternalType* block) const
  {
    return shape ? decode_block_strided(offset, shape, block, 1, 4, 16)
                 : decode_block(offset, block);
  }

  // encode 3D block from strided storage
  size_t encode_block_strided(bitstream_offset offset, uint shape, const ExternalType* p, ptrdiff_t sx, ptrdiff_t sy, ptrdiff_t sz) const
  {
    InternalType* q = begin(offset);
    size_t nx = 4;
    size_t ny = 4;
    size_t nz = 4;
    if (shape) {
      nx -= shape & 3u; shape >>= 2;
      ny -= shape & 3u; shape >>= 2;
      nz -= shape & 3u; shape >>= 2;
    }
    for (size_t z = 0; z < nz; z++, p += sz - (ptrdiff_t)ny * sy, q += 16 - 4 * ny)
      for (size_t y = 0; y < ny; y++, p += sy - (ptrdiff_t)nx * sx, q += 4 - nx)
        for (size_t x = 0; x < nx; x++, p += sx, q++)
          *q = static_cast<InternalType>(*p);
    return block_size_bits;
  }

  // decode 3D block to strided storage
  size_t decode_block_strided(bitstream_offset offset, uint shape, ExternalType* p, ptrdiff_t sx, ptrdiff_t sy, ptrdiff_t sz) const
  {
    const InternalType* q = begin(offset);
    size_t nx = 4;
    size_t ny = 4;
    size_t nz = 4;
    if (shape) {
      nx -= shape & 3u; shape >>= 2;
      ny -= shape & 3u; shape >>= 2;
      nz -= shape & 3u; shape >>= 2;
    }
    for (size_t z = 0; z < nz; z++, p += sz - (ptrdiff_t)ny * sy, q += 16 - 4 * ny)
      for (size_t y = 0; y < ny; y++, p += sy - (ptrdiff_t)nx * sx, q += 4 - nx)
        for (size_t x = 0; x < nx; x++, p += sx, q++)
          *p = static_cast<ExternalType>(*q);
    return block_size_bits;
  }

protected:
  using generic_base<3, ExternalType, InternalType>::begin;
  using generic_base<3, ExternalType, InternalType>::encode_block;
  using generic_base<3, ExternalType, InternalType>::decode_block;
  using generic_base<3, ExternalType, InternalType>::block_size_bits;
};

// 4D codec
template <typename ExternalType, typename InternalType = ExternalType>
class generic4 : public generic_base<4, ExternalType, InternalType> {
public:
  // encode contiguous 4D block
  size_t encode_block(bitstream_offset offset, uint shape, const ExternalType* block) const
  {
    return shape ? encode_block_strided(offset, shape, block, 1, 4, 16, 64)
                 : encode_block(offset, block);
  }

  // decode contiguous 4D block
  size_t decode_block(bitstream_offset offset, uint shape, ExternalType* block) const
  {
    return shape ? decode_block_strided(offset, shape, block, 1, 4, 16, 64)
                 : decode_block(offset, block);
  }

  // encode 4D block from strided storage
  size_t encode_block_strided(bitstream_offset offset, uint shape, const ExternalType* p, ptrdiff_t sx, ptrdiff_t sy, ptrdiff_t sz, ptrdiff_t sw) const
  {
    InternalType* q = begin(offset);
    size_t nx = 4;
    size_t ny = 4;
    size_t nz = 4;
    size_t nw = 4;
    if (shape) {
      nx -= shape & 3u; shape >>= 2;
      ny -= shape & 3u; shape >>= 2;
      nz -= shape & 3u; shape >>= 2;
      nw -= shape & 3u; shape >>= 2;
    }
    for (size_t w = 0; w < nw; w++, p += sw - (ptrdiff_t)nz * sz, q += 64 - 16 * nz)
      for (size_t z = 0; z < nz; z++, p += sz - (ptrdiff_t)ny * sy, q += 16 - 4 * ny)
        for (size_t y = 0; y < ny; y++, p += sy - (ptrdiff_t)nx * sx, q += 4 - nx)
          for (size_t x = 0; x < nx; x++, p += sx, q++)
            *q = static_cast<InternalType>(*p);
    return block_size_bits;
  }

  // decode 4D block to strided storage
  size_t decode_block_strided(bitstream_offset offset, uint shape, ExternalType* p, ptrdiff_t sx, ptrdiff_t sy, ptrdiff_t sz, ptrdiff_t sw) const
  {
    const InternalType* q = begin(offset);
    size_t nx = 4;
    size_t ny = 4;
    size_t nz = 4;
    size_t nw = 4;
    if (shape) {
      nx -= shape & 3u; shape >>= 2;
      ny -= shape & 3u; shape >>= 2;
      nz -= shape & 3u; shape >>= 2;
      nw -= shape & 3u; shape >>= 2;
    }
    for (size_t w = 0; w < nw; w++, p += sw - (ptrdiff_t)nz * sz, q += 64 - 16 * nz)
      for (size_t z = 0; z < nz; z++, p += sz - (ptrdiff_t)ny * sy, q += 16 - 4 * ny)
        for (size_t y = 0; y < ny; y++, p += sy - (ptrdiff_t)nx * sx, q += 4 - nx)
          for (size_t x = 0; x < nx; x++, p += sx, q++)
            *p = static_cast<ExternalType>(*q);
    return block_size_bits;
  }

protected:
  using generic_base<4, ExternalType, InternalType>::begin;
  using generic_base<4, ExternalType, InternalType>::encode_block;
  using generic_base<4, ExternalType, InternalType>::decode_block;
  using generic_base<4, ExternalType, InternalType>::block_size_bits;
};

} // codec
} // zfp

#endif
