#ifndef ZFP_COMPRESS_H
#define ZFP_COMPRESS_H

#include <cstddef>

namespace ZFP {

typedef unsigned int uint;

size_t            // byte size of compressed bit stream (zero upon error)
compress(
  const void* in, // uncompressed floating-point data
  void* out,      // compressed bit stream (must be allocated and large enough)
  bool dp,        // double-precision data?
  uint nx,        // array x dimensions
  uint ny,        // array y dimensions (zero if 1D)
  uint nz,        // array z dimensions (zero if 1D or 2D)
  uint minbits,   // minimum number of bits to store per block
  uint maxbits,   // maximum number of bits to store per block
  uint maxprec,   // maximum number of bits to store per value
  int minexp      // minimum bitplane number (soft error tolerance = 2^minexp)
);

bool              // true upon success
decompress(
  const void* in, // compressed bit stream
  void* out,      // decompressed floating-point data
  bool dp,        // double-precision data?
  uint nx,        // array x dimensions
  uint ny,        // array y dimensions (zero if 1D)
  uint nz,        // array z dimensions (zero if 1D or 2D)
  uint minbits,   // minimum number of bits stored per block
  uint maxbits,   // maximum number of bits stored per block
  uint maxprec,   // maximum number of bits stored per value
  int minexp      // minimum bitplane number (soft error tolerance = 2^minexp)
);

}

#endif
