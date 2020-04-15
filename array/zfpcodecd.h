// double-precision 1D codec
template <>
class codec<double, 1> : public codec_base<double, 1, zfp_type_double> {
public:
  // constructor takes pre-allocated buffer of compressed blocks
  codec(void* data, size_t size) : codec_base(data, size) {}

  // encode contiguous 1D block
  size_t encode_block(size_t offset, uint shape, const double* block)
  {
    size_t size;
    stream_wseek(zfp->stream, offset);
    if (shape) {
      uint nx = 4 - (shape & 3u); shape >>= 2;
      size = zfp_encode_partial_block_strided_double_1(zfp, block, nx, 1);
    }
    else
      size = zfp_encode_block_double_1(zfp, block);
    size += zfp_stream_flush(zfp);
    return size;
  }

  // encode 1D block from strided storage
  size_t encode_block_strided(size_t offset, uint shape, const double* p, ptrdiff_t sx)
  {
    size_t size;
    stream_wseek(zfp->stream, offset);
    if (shape) {
      uint nx = 4 - (shape & 3u); shape >>= 2;
      size = zfp_encode_partial_block_strided_double_1(zfp, p, nx, sx);
    }
    else
      size = zfp_encode_block_strided_double_1(zfp, p, sx);
    size += zfp_stream_flush(zfp);
    return size;
  }

  // decode contiguous 1D block
  size_t decode_block(size_t offset, uint shape, double* block)
  {
    size_t size;
    stream_rseek(zfp->stream, offset);
    if (shape) {
      uint nx = 4 - (shape & 3u); shape >>= 2;
      size = zfp_decode_partial_block_strided_double_1(zfp, block, nx, 1);
    }
    else
      size = zfp_decode_block_double_1(zfp, block);
    size += zfp_stream_align(zfp);
    return size;
  }

  // decode 1D block to strided storage
  size_t decode_block_strided(size_t offset, uint shape, double* p, ptrdiff_t sx)
  {
    size_t size;
    stream_rseek(zfp->stream, offset);
    if (shape) {
      uint nx = 4 - (shape & 3u); shape >>= 2;
      size = zfp_decode_partial_block_strided_double_1(zfp, p, nx, sx);
    }
    else
      size = zfp_decode_block_strided_double_1(zfp, p, sx);
    size += zfp_stream_align(zfp);
    return size;
  }
};

// double-precision 2D codec
template <>
class codec<double, 2> : public codec_base<double, 2, zfp_type_double> {
public:
  // constructor takes pre-allocated buffer of compressed blocks
  codec(void* data, size_t size) : codec_base(data, size) {}

  // clone object
  codec* clone() const { return 0; }

  // encode contiguous 2D block
  size_t encode_block(size_t offset, uint shape, const double* block)
  {
    size_t size;
    stream_wseek(zfp->stream, offset);
    if (shape) {
      uint nx = 4 - (shape & 3u); shape >>= 2;
      uint ny = 4 - (shape & 3u); shape >>= 2;
      size = zfp_encode_partial_block_strided_double_2(zfp, block, nx, ny, 1, 4);
    }
    else
      size = zfp_encode_block_double_2(zfp, block);
    size += zfp_stream_flush(zfp);
    return size;
  }

  // encode 2D block from strided storage
  size_t encode_block_strided(size_t offset, uint shape, const double* p, ptrdiff_t sx, ptrdiff_t sy)
  {
    size_t size;
    stream_wseek(zfp->stream, offset);
    if (shape) {
      uint nx = 4 - (shape & 3u); shape >>= 2;
      uint ny = 4 - (shape & 3u); shape >>= 2;
      size = zfp_encode_partial_block_strided_double_2(zfp, p, nx, ny, sx, sy);
    }
    else
      size = zfp_encode_block_strided_double_2(zfp, p, sx, sy);
    size += zfp_stream_flush(zfp);
    return size;
  }

  // decode contiguous 2D block
  size_t decode_block(size_t offset, uint shape, double* block)
  {
    size_t size;
    stream_rseek(zfp->stream, offset);
    if (shape) {
      uint nx = 4 - (shape & 3u); shape >>= 2;
      uint ny = 4 - (shape & 3u); shape >>= 2;
      size = zfp_decode_partial_block_strided_double_2(zfp, block, nx, ny, 1, 4);
    }
    else
      size = zfp_decode_block_double_2(zfp, block);
    size += zfp_stream_align(zfp);
    return size;
  }

  // decode 2D block to strided storage
  size_t decode_block_strided(size_t offset, uint shape, double* p, ptrdiff_t sx, ptrdiff_t sy)
  {
    size_t size;
    stream_rseek(zfp->stream, offset);
    if (shape) {
      uint nx = 4 - (shape & 3u); shape >>= 2;
      uint ny = 4 - (shape & 3u); shape >>= 2;
      size = zfp_decode_partial_block_strided_double_2(zfp, p, nx, ny, sx, sy);
    }
    else
      size = zfp_decode_block_strided_double_2(zfp, p, sx, sy);
    size += zfp_stream_align(zfp);
    return size;
  }
};

// double-precision 3D codec
template <>
class codec<double, 3> : public codec_base<double, 3, zfp_type_double> {
public:
  // constructor takes pre-allocated buffer of compressed blocks
  codec(void* data, size_t size) : codec_base(data, size) {}

  // clone object
  codec* clone() const { return 0; }

  // encode contiguous 3D block
  size_t encode_block(size_t offset, uint shape, const double* block)
  {
    size_t size;
    stream_wseek(zfp->stream, offset);
    if (shape) {
      uint nx = 4 - (shape & 3u); shape >>= 2;
      uint ny = 4 - (shape & 3u); shape >>= 2;
      uint nz = 4 - (shape & 3u); shape >>= 2;
      size = zfp_encode_partial_block_strided_double_3(zfp, block, nx, ny, nz, 1, 4, 16);
    }
    else
      size = zfp_encode_block_double_3(zfp, block);
    size += zfp_stream_flush(zfp);
    return size;
  }

  // encode 3D block from strided storage
  size_t encode_block_strided(size_t offset, uint shape, const double* p, ptrdiff_t sx, ptrdiff_t sy, ptrdiff_t sz)
  {
    size_t size;
    stream_wseek(zfp->stream, offset);
    if (shape) {
      uint nx = 4 - (shape & 3u); shape >>= 2;
      uint ny = 4 - (shape & 3u); shape >>= 2;
      uint nz = 4 - (shape & 3u); shape >>= 2;
      size = zfp_encode_partial_block_strided_double_3(zfp, p, nx, ny, nz, sx, sy, sz);
    }
    else
      size = zfp_encode_block_strided_double_3(zfp, p, sx, sy, sz);
    size += zfp_stream_flush(zfp);
    return size;
  }

  // decode contiguous 3D block
  size_t decode_block(size_t offset, uint shape, double* block)
  {
    size_t size;
    stream_rseek(zfp->stream, offset);
    if (shape) {
      uint nx = 4 - (shape & 3u); shape >>= 2;
      uint ny = 4 - (shape & 3u); shape >>= 2;
      uint nz = 4 - (shape & 3u); shape >>= 2;
      size = zfp_decode_partial_block_strided_double_3(zfp, block, nx, ny, nz, 1, 4, 16);
    }
    else
      size = zfp_decode_block_double_3(zfp, block);
    size += zfp_stream_align(zfp);
    return size;
  }

  // decode 3D block to strided storage
  size_t decode_block_strided(size_t offset, uint shape, double* p, ptrdiff_t sx, ptrdiff_t sy, ptrdiff_t sz)
  {
    size_t size;
    stream_rseek(zfp->stream, offset);
    if (shape) {
      uint nx = 4 - (shape & 3u); shape >>= 2;
      uint ny = 4 - (shape & 3u); shape >>= 2;
      uint nz = 4 - (shape & 3u); shape >>= 2;
      size = zfp_decode_partial_block_strided_double_3(zfp, p, nx, ny, nz, sx, sy, sz);
    }
    else
      size = zfp_decode_block_strided_double_3(zfp, p, sx, sy, sz);
    size += zfp_stream_align(zfp);
    return size;
  }
};
