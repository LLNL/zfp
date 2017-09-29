/* measure the throughput of encoding and decoding 3D blocks of doubles */

#include <limits.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "zfp.h"

/* example 3D block of (reinterpreted) doubles */
static const uint64 block[] = {
UINT64C(0xbf7c3a7bb8495ca9),
UINT64C(0xbf79f9d9058ffdaf),
UINT64C(0xbf77c7abd0b61999),
UINT64C(0xbf75a42c806bd1da),
UINT64C(0xbf738f8f740b8ea8),
UINT64C(0xbf718a050399fef8),
UINT64C(0xbf6f2772ff8c30fe),
UINT64C(0xbf6b59aa63d22f68),
UINT64C(0xbf67aaf8b80cff9e),
UINT64C(0xbf641b9e71983592),
UINT64C(0xbf60abd3f723f2b7),
UINT64C(0xbf5ab7934169cc04),
UINT64C(0xbf54574f6f4897d3),
UINT64C(0xbf4c6e39da7fb99b),
UINT64C(0xbf40ae5826a893d1),
UINT64C(0xbf25bce8e19d48e1),
UINT64C(0x3f253bfed65904d7),
UINT64C(0x3f3f18ab46a04cf3),
UINT64C(0x3f4948e7cb74278b),
UINT64C(0x3f51427b51aeec2e),
UINT64C(0x3f55a0716d8b4b6b),
UINT64C(0x3f59be96aeaac56f),
UINT64C(0x3f5d9d3ba7bfd327),
UINT64C(0x3f609e608469e93e),
UINT64C(0x3f624ecbcfa3832c),
UINT64C(0x3f63e0202ae84b4d),
UINT64C(0x3f6552a61a3f4812),
UINT64C(0x3f66a6ae305af268),
UINT64C(0x3f67dc910e9935bc),
UINT64C(0x3f68f4af65036ff7),
UINT64C(0x3f69ef71f24e7182),
UINT64C(0x3f6acd4983da7d43),
UINT64C(0x3f6b8eaef5b348a0),
UINT64C(0x3f6c3423328ffb7a),
UINT64C(0x3f6cbe2f33d33034),
UINT64C(0x3f6d2d64018af3ac),
UINT64C(0x3f6d825ab270c540),
UINT64C(0x3f6dbdb46be996cc),
UINT64C(0x3f6de01a6205cca9),
UINT64C(0x3f6dea3dd7813daf),
UINT64C(0x3f6ddcd81dc33335),
UINT64C(0x3f6db8aa94de690f),
UINT64C(0x3f6d7e7eab910d8f),
UINT64C(0x3f6d2f25df44c187),
UINT64C(0x3f6ccb79bc0e9844),
UINT64C(0x3f6c545bdcaf1795),
UINT64C(0x3f6bcab5ea9237c4),
UINT64C(0x3f6b2f799dcf639b),
UINT64C(0x3f6a83a0bd297862),
UINT64C(0x3f69c82d1e0ec5de),
UINT64C(0x3f68fe28a4990e53),
UINT64C(0x3f6826a5438d8685),
UINT64C(0x3f6742bcfc5cd5b2),
UINT64C(0x3f665391df231599),
UINT64C(0x3f655a4e0aa7d278),
UINT64C(0x3f645823ac5e0b09),
UINT64C(0x3f634e4d00643085),
UINT64C(0x3f623e0c518426a3),
UINT64C(0x3f6128abf933439a),
UINT64C(0x3f600f7e5f92501c),
UINT64C(0x3f5de7bbf6db0eb7),
UINT64C(0x3f5bae5aa4792e11),
UINT64C(0x3f5975adf0453ea2),
UINT64C(0x3f57409b1fdc65c4),
};

int main(int argc, char* argv[])
{
  uint blocks = 0x200000;
  double rate = 1;
  zfp_field* field;
  uint insize;
  zfp_stream* zfp;
  bitstream* stream;
  void* buffer;
  size_t bytes;
  clock_t c;
  double time;
  uint i;

  switch (argc) {
    case 3:
      sscanf(argv[2], "%u", &blocks);
      /* FALLTHROUGH */
    case 2:
      sscanf(argv[1], "%lf", &rate);
      break;
  }

  /* declare array to compress */
  field = zfp_field_3d(NULL, zfp_type_double, 4, 4, 4 * blocks);
  insize = blocks * sizeof(block);

  /* allocate storage for compressed bit stream */
  zfp = zfp_stream_open(NULL);
  zfp_stream_set_rate(zfp, rate, zfp_field_type(field), zfp_field_dimensionality(field), 0);
  bytes = zfp_stream_maximum_size(zfp, field);
  buffer = malloc(bytes);
  stream = stream_open(buffer, bytes);
  zfp_stream_set_bit_stream(zfp, stream);
  zfp_field_free(field);

  /* compress */
  c = clock();
  for (i = 0; i < blocks; i++)
    zfp_encode_block_double_3(zfp, (const double*)block);
  zfp_stream_flush(zfp);
  time = (double)(clock() - c) / CLOCKS_PER_SEC;
  printf("encode in=%u out=%u %.0f MB/s\n", insize, (uint)stream_size(stream), insize / (1024 * 1024 * time));

  /* decompress */
  zfp_stream_rewind(zfp);
  c = clock();
  for (i = 0; i < blocks; i++) {
    double a[64];
    zfp_decode_block_double_3(zfp, a);
  }
  time = (double)(clock() - c) / CLOCKS_PER_SEC;
  printf("decode in=%u out=%u %.0f MB/s\n", (uint)stream_size(stream), insize, insize / (1024 * 1024 * time));

  zfp_stream_close(zfp);
  stream_close(stream);
  free(buffer);

  return 0;
}
