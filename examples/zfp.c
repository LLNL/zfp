/* C89 version of zfp.cpp that illustrates how to call the compressor from C */

#include <limits.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "zfp.h"

#define PI 3.14159265358979323846

#define min(x, y) ((x) < (y) ? (x) : (y))
#define max(x, y) ((x) > (y) ? (x) : (y))

static double
func(uint x, uint y, uint z, uint nx, uint ny, uint nz)
{
#if RANDOM_FIELD
  return drand48();
#else
  double fx = 2 * (x + 0.5) / nx - 1;
  double fy = 2 * (y + 0.5) / ny - 1;
  double fz = 2 * (z + 0.5) / nz - 1;
#if 1
  return cos(PI * fx) * cos(PI * fy) * cos(PI * fz);
#else
  return exp(-8 * (fx * fx + fy * fy + fz * fz));
#endif
#endif
}

void usage()
{
  fprintf(stderr, "Usage: zfp [options] <nx> [ny nz infile outfile]\n");
  fprintf(stderr, "  nx, ny, nz : grid dimensions (set nz = 0 for 2D, ny = nz = 0 for 1D)\n");
  fprintf(stderr, "  infile : optional floating-point input file to compress\n");
  fprintf(stderr, "  outfile : optional output file for reconstructed data\n");
  fprintf(stderr, "Options:\n");
  fprintf(stderr, "  -f : single precision (float type)\n");
  fprintf(stderr, "  -d : double precision (double type)\n");
  fprintf(stderr, "  -r <rate> : fixed rate (# compressed bits per floating-point value)\n");
  fprintf(stderr, "  -p <precision> : fixed precision (# uncompressed bits per value)\n");
  fprintf(stderr, "  -a <tolerance> : fixed accuracy (absolute error tolerance)\n");
  fprintf(stderr, "  -c <minbits> <maxbits> <maxprec> <minexp> : advanced usage\n");
  fprintf(stderr, "      minbits : min # bits per 4^d values in d dimensions\n");
  fprintf(stderr, "      maxbits : max # bits per 4^d values in d dimensions\n");
  fprintf(stderr, "      maxprec : max # bits of precision per value (0 for full)\n");
  fprintf(stderr, "      minexp : min bit plane # coded (-1074 for all bit planes)\n");
  fprintf(stderr, "Examples:\n");
  fprintf(stderr, "  zfp -f -r 16 100 100 100 : 2x fixed-rate compression of 100x100x100 floats\n");
  fprintf(stderr, "  zfp -d -r 32 1000000 : 2x fixed-rate compression of stream of 1M doubles\n");
  fprintf(stderr, "  zfp -d -p 32 1000 1000 : 32-bit precision compression of 1000x1000 doubles\n");
  fprintf(stderr, "  zfp -d -a 1e-9 1000000 : compression of 1M doubles with < 1e-9 error\n");
  fprintf(stderr, "  zfp -d -c 64 64 0 -1074 1000000 : 4x fixed-rate compression of 1M doubles\n");
  exit(EXIT_FAILURE);
}

int main(int argc, char* argv[])
{
  /* default settings */
  uint type = 0;
  uint nx = 0;
  uint ny = 0;
  uint nz = 0;
  double rate = 0;
  uint precision = 0;
  double tolerance = 0;
  uint minbits = 0;
  uint maxbits = 0;
  uint maxprec = 0;
  int minexp = INT_MIN;
  char* inpath = 0;
  char* outpath = 0;
  char mode = 0;

  /* local variables */
  int i;
  int dp;
  zfp_params params;
  uint mx;
  uint my;
  uint mz;
  size_t typesize;
  size_t insize;
  void* f;
  float* ff;
  double* fd;
  void* g;
  float* gf;
  double* gd;
  size_t outsize;
  unsigned char* zip;
  double e;
  double fmin;
  double fmax;
  double emax;
  double nrmse;
  double psnr;

  /* parse command-line arguments */
  for (i = 1; i < argc; i++)
    if (argv[i][0] == '-')
      switch (argv[i][1]) {
        case 'a':
          if (++i == argc || sscanf(argv[i], "%lf", &tolerance) != 1)
            usage();
          mode = 'a';
          break;
        case 'c':
          if (++i == argc || sscanf(argv[i], "%u", &minbits) != 1 ||
              ++i == argc || sscanf(argv[i], "%u", &maxbits) != 1 ||
              ++i == argc || sscanf(argv[i], "%u", &maxprec) != 1 ||
              ++i == argc || sscanf(argv[i], "%d", &minexp) != 1)
            usage();
          mode = 'c';
          break;
        case 'd':
          type = ZFP_TYPE_DOUBLE;
          break;
        case 'f':
          type = ZFP_TYPE_FLOAT;
          break;
        case 'p':
          if (++i == argc || sscanf(argv[i], "%u", &precision) != 1)
            usage();
          mode = 'p';
          break;
        case 'r':
          if (++i == argc || sscanf(argv[i], "%lf", &rate) != 1)
            usage();
          mode = 'r';
          break;
        default:
          usage();
          break;
      }
    else {
      if (sscanf(argv[i++], "%u", &nx) != 1)
        usage();
      if (i < argc && sscanf(argv[i++], "%u", &ny) != 1)
        usage();
      if (i < argc && sscanf(argv[i++], "%u", &nz) != 1)
        usage();
      if (i < argc)
        inpath = argv[i++];
      if (i < argc)
        outpath = argv[i++];
      if (i != argc)
        usage();
    }

  /* make sure we know floating-point type */
  switch (type) {
    case ZFP_TYPE_FLOAT:
      dp = 0;
      break;
    case ZFP_TYPE_DOUBLE:
      dp = 1;
      break;
    default:
      fprintf(stderr, "must specify single or double precision via -f or -d\n");
      return EXIT_FAILURE;
  }

  /* set array type and size */
  zfp_init(&params);
  params.type = type;
  params.nx = nx;
  params.ny = ny;
  params.nz = nz;

  /* set compression mode */
  switch (mode) {
    case 'a':
      zfp_set_accuracy(&params, tolerance);
      break;
    case 'p':
      zfp_set_precision(&params, precision);
      break;
    case 'r':
      zfp_set_rate(&params, rate);
      break;
    case 'c':
      params.minbits = minbits;
      params.maxbits = maxbits;
      params.maxprec = maxprec;
      params.minexp = minexp;
      break;
    default:
      fprintf(stderr, "must specify compression parameters via -a, -c, -p, or -r\n");
      return EXIT_FAILURE;
  }

  /* effective array dimensions */
  mx = max(nx, 1u);
  my = max(ny, 1u);
  mz = max(nz, 1u);

  /* size of floating-point type in bytes */
  typesize = dp ? sizeof(double) : sizeof(float);

  /* allocate space for uncompressed and compressed fields */
  insize = mx * my * mz * typesize;
  outsize = zfp_estimate_compressed_size(&params);
  if (!outsize) {
    fprintf(stderr, "invalid compression parameters\n");
    return EXIT_FAILURE;
  }
  f = malloc(insize);
  ff = (float*)f;
  fd = (double*)f;
  g = malloc(insize);
  gf = (float*)g;
  gd = (double*)g;
  zip = malloc(outsize);

  /* initialize uncompressed field */
  if (inpath) {
    /* read from file */
    FILE* file = fopen(inpath, "rb");
    if (!file) {
      fprintf(stderr, "cannot open file\n");
      return EXIT_FAILURE;
    }
    if (fread(f, typesize, mx * my * mz, file) != mx * my * mz) {
      fprintf(stderr, "cannot read file\n");
      return EXIT_FAILURE;
    }
    fclose(file);
  }
  else {
    /* evaluate function */
    uint x, y, z;
    for (z = 0; z < mz; z++)
      for (y = 0; y < my; y++)
        for (x = 0; x < mx; x++) {
          double val = func(x, y, z, mx, my, mz);
          if (dp)
            fd[x + mx * (y + my * z)] = val;
          else
            ff[x + mx * (y + my * z)] = (float)val;
        }
  }

  /* compress data */
  outsize = zfp_compress(&params, f, zip, outsize);
  if (outsize == 0) {
    fprintf(stderr, "compression failed\n");
    return EXIT_FAILURE;
  }
  rate = CHAR_BIT * (double)outsize / (mx * my * mz);

#if IT_SEEMS_TOO_GOOD_TO_BE_TRUE
  /* for skeptics: relocate compressed data */
  {
    unsigned char* copy = malloc(outsize);
    memcpy(copy, zip, outsize);
    free(zip);
    zip = copy;
  }
#endif

  /* decompress data */
  if (!zfp_decompress(&params, g, zip, outsize)) {
    fprintf(stderr, "decompression failed\n");
    return EXIT_FAILURE;
  }

  /* write reconstructed data */
  if (outpath) {
    FILE* file = fopen(outpath, "wb");
    if (!file) {
      fprintf(stderr, "cannot create file\n");
      return EXIT_FAILURE;
    }
    if (fwrite(g, typesize, mx * my * mz, file) != mx * my * mz) {
      fprintf(stderr, "cannot write file\n");
      return EXIT_FAILURE;
    }
    fclose(file);
  }

  /* compute error */
  e = 0;
  fmin = dp ? fd[0] : ff[0];
  fmax = fmin;
  emax = 0;
  for (i = 0; (uint)i < mx * my * mz; i++) {
    double d = fabs(dp ? fd[i] - gd[i] : ff[i] - gf[i]);
    double val = dp ? fd[i] : ff[i];
    emax = max(emax, d);
    e += d * d;
    fmin = min(fmin, val);
    fmax = max(fmax, val);
  }
  e = sqrt(e / (mx * my * mz));
  nrmse = e / (fmax - fmin);
  psnr = 20 * log10((fmax - fmin) / (2 * e));
  
  fprintf(stderr, "in=%lu out=%lu ratio=%.3g rate=%.4g rmse=%.4g nrmse=%.4g maxe=%.4g psnr=%.2f\n", (unsigned long)insize, (unsigned long)outsize, (double)insize / outsize, rate, e, nrmse, emax, psnr);

  /* clean up */
  free(f);
  free(g);
  free(zip);

  return EXIT_SUCCESS;
}
