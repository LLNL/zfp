/* minimal code example showing how to call the zfp (de)compressor */

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "zfp.h"

/* compress or decompress array */
static int
compress(double* array, size_t nx, size_t ny, size_t nz, double tolerance, zfp_bool decompress)
{
  int status = 0;    /* return value: 0 = success */
  zfp_type type;     /* array scalar type */
  zfp_field* field;  /* array meta data */
  zfp_stream* zfp;   /* compressed stream */
  void* buffer;      /* storage for compressed stream */
  size_t bufsize;    /* byte size of compressed buffer */
  bitstream* stream; /* bit stream to write to or read from */
  size_t zfpsize;    /* byte size of compressed stream */

  /* allocate meta data for the 3D array a[nz][ny][nx] */
  type = zfp_type_double;
  field = zfp_field_3d(array, type, nx, ny, nz);

  /* allocate meta data for a compressed stream */
  zfp = zfp_stream_open(NULL);

  /* set compression mode and parameters via one of four functions */
/*  zfp_stream_set_reversible(zfp); */
/*  zfp_stream_set_rate(zfp, rate, type, zfp_field_dimensionality(field), zfp_false); */
/*  zfp_stream_set_precision(zfp, precision); */
  zfp_stream_set_accuracy(zfp, tolerance);

  /* allocate buffer for compressed data */
  bufsize = zfp_stream_maximum_size(zfp, field);
  buffer = malloc(bufsize);

  /* associate bit stream with allocated buffer */
  stream = stream_open(buffer, bufsize);
  zfp_stream_set_bit_stream(zfp, stream);
  zfp_stream_rewind(zfp);

  /* compress or decompress entire array */
  if (decompress) {
    /* read compressed stream and decompress and output array */
    zfpsize = fread(buffer, 1, bufsize, stdin);
    if (!zfp_decompress(zfp, field)) {
      fprintf(stderr, "decompression failed\n");
      status = EXIT_FAILURE;
    }
    else
      fwrite(array, sizeof(double), zfp_field_size(field, NULL), stdout);
  }
  else {
    /* compress array and output compressed stream */
    zfpsize = zfp_compress(zfp, field);
    if (!zfpsize) {
      fprintf(stderr, "compression failed\n");
      status = EXIT_FAILURE;
    }
    else
      fwrite(buffer, 1, zfpsize, stdout);
  }

  /* clean up */
  zfp_field_free(field);
  zfp_stream_close(zfp);
  stream_close(stream);
  free(buffer);
  free(array);

  return status;
}

int main(int argc, char* argv[])
{
  /* use -d to decompress rather than compress data */
  zfp_bool decompress = (argc == 2 && !strcmp(argv[1], "-d"));

  /* allocate 100x100x100 array of doubles */
  size_t nx = 100;
  size_t ny = 100;
  size_t nz = 100;
  double* array = malloc(nx * ny * nz * sizeof(double));

  if (!decompress) {
    /* initialize array to be compressed */
    size_t i, j, k;
    for (k = 0; k < nz; k++)
      for (j = 0; j < ny; j++)
        for (i = 0; i < nx; i++) {
          double x = 2.0 * i / nx;
          double y = 2.0 * j / ny;
          double z = 2.0 * k / nz;
          array[i + nx * (j + ny * k)] = exp(-(x * x + y * y + z * z));
        }
  }

  /* compress or decompress array */
  return compress(array, nx, ny, nz, 1e-3, decompress);
}
