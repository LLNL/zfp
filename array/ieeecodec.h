#ifndef ZFP_IEEE_CODEC_H
#define ZFP_IEEE_CODEC_H

#include "zfp.h"

namespace zfp {

template <typename Scalar>
struct traits;

template <>
struct traits<float> {
  static const zfp_type type = zfp_type_float;
};

template <>
struct traits<double> {
  static const zfp_type type = zfp_type_double;
};

// IEEE-754 codec with arithmetic type AType and storage type SType
template <typename AType, typename SType>
class ieee_codec {
private:
  static SType* wpointer(bitstream* stream)
  {
    return static_cast<SType*>(stream_data(stream)) + (stream_wtell(stream) / (CHAR_BIT * sizeof(SType)));
  }

  static const SType* rpointer(bitstream* stream)
  {
    return static_cast<const SType*>(stream_data(stream)) + (stream_rtell(stream) / (CHAR_BIT * sizeof(SType)));
  }

public:
  // encode contiguous 1D block
  static void encode_block_1(zfp_stream* zfp, const AType* block, uint shape)
  {
    encode_block_strided_1(zfp, block, shape, 1);
  }

  // encode 1D block from strided storage
  static void encode_block_strided_1(zfp_stream* zfp, const AType* p, uint shape, int sx)
  {
    SType* ptr = wpointer(zfp->stream);
    if (shape) {
      uint nx = 4 - (shape & 3u); shape >>= 2;
      for (uint x = 0; x < nx; x++)
        *ptr++ = static_cast<SType>(p[x * sx]);
    }
    else
      for (uint x = 0; x < 4; x++)
        *ptr++ = static_cast<SType>(p[x * sx]);
  }

  // encode contiguous 2D block
  static void encode_block_2(zfp_stream* zfp, const AType* block, uint shape)
  {
    encode_block_strided_2(zfp, block, shape, 1, 4);
  }

  // encode 2D block from strided storage
  static void encode_block_strided_2(zfp_stream* zfp, const AType* p, uint shape, int sx, int sy)
  {
    SType* ptr = wpointer(zfp->stream);
    if (shape) {
      uint nx = 4 - (shape & 3u); shape >>= 2;
      uint ny = 4 - (shape & 3u); shape >>= 2;
      for (uint y = 0; y < ny; y++)
        for (uint x = 0; x < nx; x++)
          ptr[x + 4 * y] = static_cast<SType>(p[x * sx + y * sy]);
    }
    else
      for (uint y = 0; y < 4; y++)
        for (uint x = 0; x < 4; x++)
          *ptr++ = static_cast<SType>(p[x * sx + y * sy]);
  }

  // encode contiguous 3D block
  static void encode_block_3(zfp_stream* zfp, const AType* block, uint shape)
  {
    encode_block_strided_3(zfp, block, shape, 1, 4, 16);
  }

  // encode 3D block from strided storage
  static void encode_block_strided_3(zfp_stream* zfp, const AType* p, uint shape, int sx, int sy, int sz)
  {
    SType* ptr = wpointer(zfp->stream);
    if (shape) {
      uint nx = 4 - (shape & 3u); shape >>= 2;
      uint ny = 4 - (shape & 3u); shape >>= 2;
      uint nz = 4 - (shape & 3u); shape >>= 2;
      for (uint z = 0; z < nz; z++)
        for (uint y = 0; y < ny; y++)
          for (uint x = 0; x < nx; x++)
            ptr[x + 4 * (y + 4 * z)] = static_cast<SType>(p[x * sx + y * sy + z * sz]);
    }
    else
      for (uint z = 0; z < 4; z++)
        for (uint y = 0; y < 4; y++)
          for (uint x = 0; x < 4; x++)
            *ptr++ = static_cast<SType>(p[x * sx + y * sy + z * sz]);
  }

  // decode contiguous 1D block
  static void decode_block_1(zfp_stream* zfp, AType* block, uint shape)
  {
    decode_block_strided_1(zfp, block, shape, 1);
  }

  // decode 1D block to strided storage
  static void decode_block_strided_1(zfp_stream* zfp, AType* p, uint shape, int sx)
  {
    const SType* ptr = rpointer(zfp->stream);
    if (shape) {
      uint nx = 4 - (shape & 3u); shape >>= 2;
      for (uint x = 0; x < nx; x++)
        p[x * sx] = static_cast<AType>(*ptr++);
    }
    else
      for (uint x = 0; x < 4; x++)
        p[x * sx] = static_cast<AType>(*ptr++);
  }

  // decode contiguous 2D block
  static void decode_block_2(zfp_stream* zfp, AType* block, uint shape)
  {
    decode_block_strided_2(zfp, block, shape, 1, 4);
  }

  // decode 2D block to strided storage
  static void decode_block_strided_2(zfp_stream* zfp, AType* p, uint shape, int sx, int sy)
  {
    const SType* ptr = rpointer(zfp->stream);
    if (shape) {
      uint nx = 4 - (shape & 3u); shape >>= 2;
      uint ny = 4 - (shape & 3u); shape >>= 2;
      for (uint y = 0; y < ny; y++)
        for (uint x = 0; x < nx; x++)
          p[x * sx + y * sy] = static_cast<AType>(ptr[x + 4 * y]);
    }
    else
      for (uint y = 0; y < 4; y++)
        for (uint x = 0; x < 4; x++)
          p[x * sx + y * sy] = static_cast<AType>(*ptr++);
  }

  // decode contiguous 3D block
  static void decode_block_3(zfp_stream* zfp, AType* block, uint shape)
  {
    decode_block_strided_3(zfp, block, shape, 1, 4, 16);
  }

  // decode 3D block to strided storage
  static void decode_block_strided_3(zfp_stream* zfp, AType* p, uint shape, int sx, int sy, int sz)
  {
    const SType* ptr = rpointer(zfp->stream);
    if (shape) {
      uint nx = 4 - (shape & 3u); shape >>= 2;
      uint ny = 4 - (shape & 3u); shape >>= 2;
      uint nz = 4 - (shape & 3u); shape >>= 2;
      for (uint z = 0; z < nz; z++)
        for (uint y = 0; y < ny; y++)
          for (uint x = 0; x < nx; x++)
            p[x * sx + y * sy + z * sz] = static_cast<AType>(ptr[x + 4 * (y + 4 * z)]);
    }
    else
      for (uint z = 0; z < 4; z++)
        for (uint y = 0; y < 4; y++)
          for (uint x = 0; x < 4; x++)
            p[x * sx + y * sy + z * sz] = static_cast<AType>(*ptr++);
  }

  static const zfp_type type = traits<AType>::type;
};

}

#endif
