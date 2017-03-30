/* example illustrating in-place compression and decompression */

#include <limits.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "zfp.h"

/* compress and decompress contiguous blocks */
static int
process(double* buffer, uint blocks, double tolerance)
{
  zfp_stream* zfp;   /* compressed stream */
  bitstream* stream; /* bit stream to write to or read from */
  size_t* offset;    /* per-block bit offset in compressed stream */
  double* ptr;       /* pointer to block being processed */
  size_t bufsize;    /* byte size of uncompressed storage */
  size_t zfpsize;    /* byte size of compressed stream */
  uint minbits;      /* min bits per block */
  uint maxbits;      /* max bits per block */
  uint maxprec;      /* max precision */
  int minexp;        /* min bit plane encoded */
  uint bits;         /* size of compressed block */
  uint i;

  /* maintain offset to beginning of each variable-length block */
  offset = malloc(blocks * sizeof(size_t));

  /* associate bit stream with same storage as input */
  bufsize = blocks * 4 * 4 * sizeof(*buffer);
  stream = stream_open(buffer, bufsize);

  /* allocate meta data for a compressed stream */
  zfp = zfp_stream_open(stream);

  /* set tolerance for fixed-accuracy mode */
  zfp_stream_set_accuracy(zfp, tolerance);

  /* set maxbits to guard against prematurely overwriting the input */
  zfp_stream_params(zfp, &minbits, &maxbits, &maxprec, &minexp);
  maxbits = 4 * 4 * sizeof(*buffer) * CHAR_BIT;
  zfp_stream_set_params(zfp, minbits, maxbits, maxprec, minexp);

  /* compress one block at a time in sequential order */
  ptr = buffer;
  for (i = 0; i < blocks; i++) {
    offset[i] = stream_wtell(stream);
    bits = zfp_encode_block_double_2(zfp, ptr);
    if (!bits) {
      fprintf(stderr, "compression failed\n");
      return 0;
    }
    printf("block #%u offset=%4u size=%4u\n", i, (uint)offset[i], bits);
    ptr += 4 * 4;
  }
  /* important: flush any buffered compressed bits */
  stream_flush(stream);

  /* print out size */
  zfpsize = stream_size(stream);
  printf("compressed %u bytes to %u bytes\n", (uint)bufsize, (uint)zfpsize);

  /* decompress one block at a time in reverse order */
  for (i = blocks; i--;) {
    ptr -= 4 * 4;
    stream_rseek(stream, offset[i]);
    if (!zfp_decode_block_double_2(zfp, ptr)) {
      fprintf(stderr, "decompression failed\n");
      return 0;
    }
  }

  /* clean up */
  zfp_stream_close(zfp);
  stream_close(stream);
  free(offset);

  return 1;
}

int main(int argc, char* argv[])
{
  double tolerance = 1e-6;
  double* array;
  double* buffer;
  uint bx = 2;
  uint by = 4;
  uint nx = 4 * bx;
  uint ny = 4 * by;
  uint blocks = bx * by;
  uint x, y;
  uint i, j, k;
  int status;

  switch (argc) {
    case 2:
      if (sscanf(argv[1], "%lf", &tolerance) != 1)
        goto usage;
      /* FALLTHROUGH */
    case 1:
      break;
    default:
    usage:
      fprintf(stderr, "Usage: inline [tolerance]\n");
      return EXIT_FAILURE;
  }

  printf("tolerance=%g\n", tolerance);

  /* initialize array to be compressed */
  printf("original %ux%u array:\n", nx, ny);
  array = malloc(nx * ny * sizeof(double));
  for (y = 0; y < ny; y++) {
    for (x = 0; x < nx; x++) {
      double u = 2 * (x + 0.5) / nx;
      double v = asin(1.0) * (y + 0.5);
      double f = exp(-u * u) * sin(v) / v;
      printf("%9.6f%c", f, x == nx - 1 ? '\n' : ' ');
      array[x + nx * y] = f;
    }
  }

  /* reorganize array into 4x4 blocks */
  buffer = malloc(blocks * 4 * 4 * sizeof(double));
  for (k = 0; k < blocks; k++)
    for (j = 0; j < 4; j++)
      for (i = 0; i < 4; i++) {
        uint x = 4 * (k & 1) + i;
        uint y = 4 * (k / 2) + j;
        buffer[i + 4 * (j + 4 * k)] = array[x + nx * y];
      }

  status = process(buffer, blocks, tolerance);
  if (status) {
    /* reorganize blocks into array */
    for (k = 0; k < blocks; k++)
      for (j = 0; j < 4; j++)
        for (i = 0; i < 4; i++) {
          uint x = 4 * (k & 1) + i;
          uint y = 4 * (k / 2) + j;
          array[x + nx * y] = buffer[i + 4 * (j + 4 * k)];
        }

    /* print out modified array*/
    printf("decompressed %ux%u array:\n", nx, ny);
    for (y = 0; y < ny; y++)
      for (x = 0; x < nx; x++)
        printf("%9.6f%c", array[x + nx * y], x == nx - 1 ? '\n' : ' ');
  }

  free(buffer);
  free(array);

  return status ? EXIT_SUCCESS : EXIT_FAILURE;
}
