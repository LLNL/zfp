/*
This simple example shows how zfp can be used to compress 8-bit color images
stored in the PPM image format.  This lossy compressor employs two common image
compression strategies: (1) transformation to the YCoCg color space, which
decorrelates color bands, and (2) chroma subsampling, which reduces spatial
resolution in the Co and Cg chrominance bands.  The single command-line argument
selects one of two compression modes: if a positive rate (in bits/pixel) is
specified, fixed-rate mode is selected; a negative integer argument, -p, sets
the precision to p in fixed-precision mode.  Rate allocation in fixed-rate mode
assigns more bits to luma than to chroma components due to the relatively higher
information content in luma after chroma subsampling.

The YCoCg transform employed here has been adapted to avoid range expansion and
potential overflow.  Chroma subsampling is achieved by performing zfp's forward
decorrelating transform and then zeroing all but the four lowest-sequency
coefficients, effectively reducing each chroma block to a bilinear approximation.

Because only four chroma coefficients per 4x4 pixel block are retained, an
alternative to zeroing and then encoding the remaining twelve zero-valued
coefficients is to treat the chroma block as being one-dimensional, with only
four values, and then compressing it using zfp's 1D codec.  The dimensionality
of chroma blocks (1 or 2) is specified at compile time via the PPM_CHROMA macro.

NOTE: To keep this example simple, only images whose dimensions are multiples
of four are supported.
*/

#ifdef PPM_CHROMA
  #if PPM_CHROMA != 1 && PPM_CHROMA != 2
    #error "compile with PPM_CHROMA=1 or PPM_CHROMA=2"
  #endif
#else
  /* default */
  #define PPM_CHROMA 2
#endif

#include <limits.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "zfp.h"

/* clamp values to 31-bit range */
static void
clamp(int32* block, uint n)
{
  uint i;
  for (i = 0; i < n; i++) {
    if (block[i] < 1 - (1 << 30))
      block[i] = 1 - (1 << 30);
    if (block[i] > (1 << 30) - 1)
      block[i] = (1 << 30) - 1;
  }
}

/* convert 2D block from RGB to YCoCg color space */
static void
rgb2ycocg(int32 ycocg[3][16], /*const*/ int32 rgb[3][16])
{
  uint i;
  for (i = 0; i < 16; i++) {
    int32 r, g, b;
    int32 y, co, cg, t;
    /* fetch RGB values */
    r = rgb[0][i];
    g = rgb[1][i];
    b = rgb[2][i];
    /* perform range-preserving YCoCg forward transform */
    co = (r - b) >> 1;
    t = b + co;
    cg = (g - t) >> 1;
    y = t + cg;
    /* store YCoCg values */
    ycocg[0][i] = y;
    ycocg[1][i] = co;
    ycocg[2][i] = cg;
  }
}

/* convert 2D block from YCoCg to RGB color space */
static void
ycocg2rgb(int32 rgb[3][16], /*const*/ int32 ycocg[3][16])
{
  uint i;
  for (i = 0; i < 16; i++) {
    int32 r, g, b;
    int32 y, co, cg, t;
    /* fetch YCoCg values */
    y = ycocg[0][i];
    co = ycocg[1][i];
    cg = ycocg[2][i];
    /* perform range-preserving YCoCg inverse transform */
    t = y - cg;
    g = (cg << 1) + t;
    b = t - co;
    r = (co << 1) + b;
    /* store RGB values */
    rgb[0][i] = r;
    rgb[1][i] = g;
    rgb[2][i] = b;
  }
}

/* perform partial forward decorrelating transform */
static void
fwd_lift(int32* p, uint s)
{
  int32 x, y, z, w;
  x = *p; p += s;
  y = *p; p += s;
  z = *p; p += s;
  w = *p; p += s;

  x += w; x >>= 1; w -= x;
  z += y; z >>= 1; y -= z;
  x += z; x >>= 1; z -= x;
  w += y; w >>= 1; y -= w;
  w += y >> 1; y -= w >> 1;

  p -= s; *p = w;
  p -= s; *p = z;
  p -= s; *p = y;
  p -= s; *p = x;
}

/* perform partial inverse decorrelating transform */
static void
inv_lift(int32* p, uint s)
{
  int32 x, y, z, w;
  x = *p; p += s;
  y = *p; p += s;
  z = *p; p += s;
  w = *p; p += s;

  y += w >> 1; w -= y >> 1;
  y += w; w <<= 1; w -= y;
  z += x; x <<= 1; x -= z;
  y += z; z <<= 1; z -= y;
  w += x; x <<= 1; x -= w;

  p -= s; *p = w;
  p -= s; *p = z;
  p -= s; *p = y;
  p -= s; *p = x;
}

/* perform chroma subsampling by discarding high-frequency components */
static void
chroma_downsample(int32* block)
{
  uint i, j;
  /* perform forward decorrelating transform */
  for (j = 0; j < 4; j++)
    fwd_lift(block + 4 * j, 1);
  for (i = 0; i < 4; i++)
    fwd_lift(block + 1 * i, 4);
#if PPM_CHROMA == 1
  /* keep only the four lowest-sequency coefficients */
  block[2] = block[4];
  block[3] = block[5];
  for (i = 4; i < 16; i++)
    block[i] = 0;
  /* reconstruct as 1D block */
  inv_lift(block, 1);
  /* clamp values to 31 bits to avoid overflow */
  clamp(block, 4);
#else
  /* zero out all but four lowest-sequency coefficients */
  for (j = 0; j < 4; j++)
    for (i = 0; i < 4; i++)
      if (i >= 2 || j >= 2)
        block[i + 4 * j] = 0;
  /* perform inverse decorrelating transform */
  for (i = 0; i < 4; i++)
    inv_lift(block + 1 * i, 4);
  for (j = 0; j < 4; j++)
    inv_lift(block + 4 * j, 1);
  /* clamp values to 31 bits to avoid overflow */
  clamp(block, 16);
#endif
}

/* reconstruct 2D chroma block */
static void
chroma_upsample(int32* block)
{
#if PPM_CHROMA == 1
  uint i, j;
  /* obtain 1D block coefficients */
  fwd_lift(block, 1);
  /* reorganize and initialize remaining 2D block coefficients */
  block[4] = block[2];
  block[5] = block[3];
  block[2] = 0;
  block[3] = 0;
  for (i = 6; i < 16; i++)
    block[i] = 0;
  /* perform inverse decorrelating transform */
  for (i = 0; i < 4; i++)
    inv_lift(block + 1 * i, 4);
  for (j = 0; j < 4; j++)
    inv_lift(block + 4 * j, 1);
  /* clamp values to 31 bits to avoid overflow */
  clamp(block, 16);
#else
  /* clamp values to 31 bits to avoid overflow */
  clamp(block, 16);
#endif
}

int main(int argc, char* argv[])
{
  double rate = 0;
  uint nx, ny;
  uint x, y;
  uint k;
  char line[0x100];
  uchar* image;
  zfp_field* field;
  zfp_stream* zfp[3];
  bitstream* stream;
  void* buffer;
  size_t bytes;
  size_t size;

  switch (argc) {
    case 2:
      if (sscanf(argv[1], "%lf", &rate) != 1)
        goto usage;
      break;
    default:
    usage:
      fprintf(stderr, "Usage: ppm <rate|-precision> <input.ppm >output.ppm\n");
      return EXIT_FAILURE;
  }

  /* read ppm header */
  if (!fgets(line, sizeof(line), stdin) || strcmp(line, "P6\n") ||
      !fgets(line, sizeof(line), stdin) || sscanf(line, "%u%u", &nx, &ny) != 2 ||
      !fgets(line, sizeof(line), stdin) || strcmp(line, "255\n")) {
    fprintf(stderr, "error opening image\n");
    return EXIT_FAILURE;
  }
  if ((nx & 3u) || (ny & 3u)) {
    fprintf(stderr, "image dimensions must be multiples of four\n");
    return EXIT_FAILURE;
  }

  /* read image data */
  image = malloc(3 * nx * ny);
  if (!image) {
    fprintf(stderr, "error allocating memory\n");
    return EXIT_FAILURE;
  }
  if (fread(image, sizeof(*image), 3 * nx * ny, stdin) != 3 * nx * ny) {
    fprintf(stderr, "error reading image\n");
    return EXIT_FAILURE;
  }

  /* initialize compressed streams */
  for (k = 0; k < 3; k++)
    zfp[k] = zfp_stream_open(NULL);
  if (rate < 0) {
    /* use fixed-precision mode */
    for (k = 0; k < 3; k++)
      zfp_stream_set_precision(zfp[k], (uint)floor(0.5 - rate));
  }
  else {
    /* assign higher rate to luminance than to chrominance components */
#if PPM_CHROMA == 1
    double chroma_rate = floor(8 * rate / 3 + 0.5) / 4;
    double luma_rate = rate - chroma_rate / 2;
    zfp_stream_set_rate(zfp[0], luma_rate, zfp_type_int32, 2, zfp_false);
    zfp_stream_set_rate(zfp[1], chroma_rate, zfp_type_int32, 1, zfp_false);
    zfp_stream_set_rate(zfp[2], chroma_rate, zfp_type_int32, 1, zfp_false);
#else
    double chroma_rate = floor(8 * rate / 3 + 0.5) / 16;
    double luma_rate = rate - 2 * chroma_rate;
    zfp_stream_set_rate(zfp[0], luma_rate, zfp_type_int32, 2, zfp_false);
    zfp_stream_set_rate(zfp[1], chroma_rate, zfp_type_int32, 2, zfp_false);
    zfp_stream_set_rate(zfp[2], chroma_rate, zfp_type_int32, 2, zfp_false);
#endif
  }

  /* determine size of compressed buffer */
  bytes = 0;
  field = zfp_field_2d(image, zfp_type_int32, nx, ny);
  for (k = 0; k < 3; k++)
    bytes += zfp_stream_maximum_size(zfp[k], field);
  zfp_field_free(field);

  /* allocate buffer and initialize bit stream */
  buffer = malloc(bytes);
  if (!buffer) {
    fprintf(stderr, "error allocating memory\n");
    return EXIT_FAILURE;
  }
  stream = stream_open(buffer, bytes);

  /* the three zfp streams share a single bit stream */
  for (k = 0; k < 3; k++)
    zfp_stream_set_bit_stream(zfp[k], stream);

  /* compress image */
  for (y = 0; y < ny; y += 4)
    for (x = 0; x < nx; x += 4) {
      uchar block[3][16];
      int32 rgb[3][16];
      int32 ycocg[3][16];
      uint i, j, k;
      /* fetch R, G, and B blocks */
      for (k = 0; k < 3; k++)
        for (j = 0; j < 4; j++)
          for (i = 0; i < 4; i++)
            block[k][i + 4 * j] = image[k + 3 * (x + i + nx * (y + j))];
      /* promote to 32-bit integers */
      for (k = 0; k < 3; k++)
        zfp_promote_uint8_to_int32(rgb[k], block[k], 2);
      /* perform color space transform */
      rgb2ycocg(ycocg, rgb);
      /* chroma subsample the Co and Cg bands */
      for (k = 1; k < 3; k++)
        chroma_downsample(ycocg[k]);
      /* compress the Y, Co, and Cg blocks */
#if PPM_CHROMA == 1
      zfp_encode_block_int32_2(zfp[0], ycocg[0]);
      zfp_encode_block_int32_1(zfp[1], ycocg[1]);
      zfp_encode_block_int32_1(zfp[2], ycocg[2]);
#else
      for (k = 0; k < 3; k++)
        zfp_encode_block_int32_2(zfp[k], ycocg[k]);
#endif
    }

  zfp_stream_flush(zfp[0]);
  size = zfp_stream_compressed_size(zfp[0]);
  fprintf(stderr, "%u compressed bytes (%.2f bits/pixel)\n", (uint)size, (double)size * CHAR_BIT / (nx * ny));

  /* decompress image */
  zfp_stream_rewind(zfp[0]);
  for (y = 0; y < ny; y += 4)
    for (x = 0; x < nx; x += 4) {
      uchar block[3][16];
      int32 rgb[3][16];
      int32 ycocg[3][16];
      uint i, j, k;
      /* decompress the Y, Co, and Cg blocks */
#if PPM_CHROMA == 1
      zfp_decode_block_int32_2(zfp[0], ycocg[0]);
      zfp_decode_block_int32_1(zfp[1], ycocg[1]);
      zfp_decode_block_int32_1(zfp[2], ycocg[2]);
#else
      for (k = 0; k < 3; k++)
        zfp_decode_block_int32_2(zfp[k], ycocg[k]);
#endif
      /* reconstruct Co and Cg chroma bands */
      for (k = 1; k < 3; k++)
        chroma_upsample(ycocg[k]);
      /* perform color space transform */
      ycocg2rgb(rgb, ycocg);
      /* demote to 8-bit integers */
      for (k = 0; k < 3; k++)
        zfp_demote_int32_to_uint8(block[k], rgb[k], 2);
      /* store R, G, and B blocks */
      for (k = 0; k < 3; k++)
        for (j = 0; j < 4; j++)
          for (i = 0; i < 4; i++)
            image[k + 3 * (x + i + nx * (y + j))] = block[k][i + 4 * j];
    }

  /* clean up */
  for (k = 0; k < 3; k++)
    zfp_stream_close(zfp[k]);
  stream_close(stream);
  free(buffer);

  /* output reconstructed image */
  printf("P6\n");
  printf("%u %u\n", nx, ny);
  printf("255\n");
  if (fwrite(image, sizeof(*image), 3 * nx * ny, stdout) != 3 * nx * ny) {
    fprintf(stderr, "error writing image\n");
    return EXIT_FAILURE;
  }
  free(image);

  return 0;
}
