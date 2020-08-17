/* simple example that shows how zfp can be used to compress ppm color images */

#include <limits.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "zfp.h"

/* convert from RGB to YCoCg color space */
static void
rgb2ycocg(int32 ycocg[3][16], const int32 rgb[3][16])
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
    co = (r - b) / 2;
    t = b + co;
    cg = (g - t) / 2;
    y = t + cg;
    /* store YCoCg values */
    ycocg[0][i] = y;
    ycocg[1][i] = co;
    ycocg[2][i] = cg;
  }
}

/* convert from YCoCg to RGB color space */
static void
ycocg2rgb(int32 rgb[3][16], const int32 ycocg[3][16])
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
    g = 2 * cg + t;
    b = t - co;
    r = 2 * co + b;
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
chroma_subsample(int32* block)
{
  uint i, j;
  /* perform forward decorrelating transform */
  for (j = 0; j < 4; j++)
    fwd_lift(block + 4 * j, 1);
  for (i = 0; i < 4; i++)
    fwd_lift(block + i, 4);
  /* zero out all but four lowest-sequency coefficients */
  for (j = 0; j < 4; j++)
    for (i = 0; i < 4; i++)
      if (i >= 2 || j >= 2)
        block[i + 4 * j] = 0;
  /* perform inverse decorrelating transform */
  for (i = 0; i < 4; i++)
    inv_lift(block + i, 4);
  for (j = 0; j < 4; j++)
    inv_lift(block + 4 * j, 1);
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
      fprintf(stderr, "Usage: pgm <rate|-precision> <input.ppm >output.ppm\n");
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
  if (fread(image, sizeof(*image), 3 * nx * ny, stdin) != 3 * nx * ny) {
    fprintf(stderr, "error reading image\n");
    return EXIT_FAILURE;
  }

  /* create input array */
  field = zfp_field_2d(image, zfp_type_int32, nx, ny);

  /* initialize compressed stream */
  for (k = 0; k < 3; k++)
    zfp[k] = zfp_stream_open(NULL);
  if (rate < 0) {
    for (k = 0; k < 3; k++)
      zfp_stream_set_precision(zfp[k], (uint)floor(0.5 - rate));
  }
  else {
    /* assign higher rate to luminance than to chrominance components */
    zfp_stream_set_rate(zfp[0], rate * 2, zfp_type_int32, 2, 0);
    zfp_stream_set_rate(zfp[1], rate / 2, zfp_type_int32, 2, 0);
    zfp_stream_set_rate(zfp[2], rate / 2, zfp_type_int32, 2, 0);
  }
  bytes = 0;
  for (k = 0; k < 3; k++)
    bytes += zfp_stream_maximum_size(zfp[k], field);
  buffer = malloc(bytes);
  stream = stream_open(buffer, bytes);
  for (k = 0; k < 3; k++)
    zfp_stream_set_bit_stream(zfp[k], stream);
  zfp_field_free(field);

  /* compress */
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
        chroma_subsample(ycocg[i]);
      /* compress the Y, Co, and Cg blocks */
      for (k = 0; k < 3; k++)
        zfp_encode_block_int32_2(zfp[k], ycocg[k]);
    }

  zfp_stream_flush(zfp[0]);
  size = zfp_stream_compressed_size(zfp[0]);
  fprintf(stderr, "%u compressed bytes (%.2f bps)\n", (uint)size, (double)size * CHAR_BIT / (3 * nx * ny));

  /* decompress */
  zfp_stream_rewind(zfp[0]);
  for (y = 0; y < ny; y += 4)
    for (x = 0; x < nx; x += 4) {
      uchar block[3][16];
      int32 rgb[3][16];
      int32 ycocg[3][16];
      uint i, j, k;
      /* decompress the Y, Co, and Cg blocks */
      for (k = 0; k < 3; k++)
        zfp_decode_block_int32_2(zfp[k], ycocg[k]);
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
  for (k = 0; k < 3; k++)
    zfp_stream_close(zfp[k]);
  stream_close(stream);
  free(buffer);

  /* output reconstructed image */
  printf("P6\n");
  printf("%u %u\n", nx, ny);
  printf("255\n");
  fwrite(image, sizeof(*image), 3 * nx * ny, stdout);
  free(image);

  return 0;
}
