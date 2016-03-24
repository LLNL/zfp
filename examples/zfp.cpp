#include <algorithm>
#include <climits>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <iomanip>
#include "zfp.h"

#define PI 3.14159265358979323846

static double
func(uint x, uint y, uint z, uint nx, uint ny, uint nz)
{
  double fx = 2 * (x + 0.5) / nx - 1;
  double fy = 2 * (y + 0.5) / ny - 1;
  double fz = 2 * (z + 0.5) / nz - 1;
#if RANDOM_FIELD
  return drand48();
#elif 1
  return cos(PI * fx) * cos(PI * fy) * cos(PI * fz);
#else
  return exp(-8 * (fx * fx + fy * fy + fz * fz));
#endif
}

void usage()
{
  std::cerr << "Usage: zfp [options] <nx> [ny nz infile outfile]" << std::endl;
  std::cerr << "  nx, ny, nz : grid dimensions (set nz = 0 for 2D, ny = nz = 0 for 1D)" << std::endl;
  std::cerr << "  infile : optional floating-point input file to compress" << std::endl;
  std::cerr << "  outfile : optional output file for reconstructed data" << std::endl;
  std::cerr << "Options:" << std::endl;
  std::cerr << "  -f : single precision (float type)" << std::endl;
  std::cerr << "  -d : double precision (double type)" << std::endl;
  std::cerr << "  -r <rate> : fixed rate (# compressed bits per floating-point value)" << std::endl;
  std::cerr << "  -p <precision> : fixed precision (# uncompressed bits per value)" << std::endl;
  std::cerr << "  -a <tolerance> : fixed accuracy (absolute error tolerance)" << std::endl;
  std::cerr << "  -c <minbits> <maxbits> <maxprec> <minexp> : advanced usage" << std::endl;
  std::cerr << "      minbits : min # bits per 4^d values in d dimensions" << std::endl;
  std::cerr << "      maxbits : max # bits per 4^d values in d dimensions" << std::endl;
  std::cerr << "      maxprec : max # bits of precision per value (0 for full)" << std::endl;
  std::cerr << "      minexp : min bit plane # coded (-1074 for all bit planes)" << std::endl;
  std::cerr << "Examples:" << std::endl;
  std::cerr << "  zfp -f -r 16 100 100 100 : 2x fixed-rate compression of 100x100x100 floats" << std::endl;
  std::cerr << "  zfp -d -r 32 1000000 : 2x fixed-rate compression of stream of 1M doubles" << std::endl;
  std::cerr << "  zfp -d -p 32 1000 1000 : 32-bit precision compression of 1000x1000 doubles" << std::endl;
  std::cerr << "  zfp -d -a 1e-9 1000000 : compression of 1M doubles with < 1e-9 error" << std::endl;
  std::cerr << "  zfp -d -c 64 64 0 -1074 1000000 : 4x fixed-rate compression of 1M doubles" << std::endl;
  exit(EXIT_FAILURE);
}

int main(int argc, char* argv[])
{
  // default settings
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

  // parse command-line arguments
  for (int i = 1; i < argc; i++)
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

  // make sure we know floating-point type
  bool dp;
  switch (type) {
    case ZFP_TYPE_FLOAT:
      dp = false;
      break;
    case ZFP_TYPE_DOUBLE:
      dp = true;
      break;
    default:
      std::cerr << "must specify single or double precision via -f or -d" << std::endl;
      return EXIT_FAILURE;
  }

  // set array type and size
  zfp_params params;
  zfp_init(&params);
  params.type = type;
  params.nx = nx;
  params.ny = ny;
  params.nz = nz;

  // set compression mode
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
      std::cerr << "must specify compression parameters via -a, -c, -p, or -r" << std::endl;
      return EXIT_FAILURE;
  }

  // effective array dimensions
  uint mx = std::max(nx, 1u);
  uint my = std::max(ny, 1u);
  uint mz = std::max(nz, 1u);

  // size of floating-point type in bytes
  size_t typesize = dp ? sizeof(double) : sizeof(float);

  // allocate space for uncompressed and compressed fields
  size_t insize = mx * my * mz * typesize;
  size_t outsize = zfp_estimate_compressed_size(&params);
  if (!outsize) {
    std::cerr << "invalid compression parameters" << std::endl;
    return EXIT_FAILURE;
  }
  void* f = new unsigned char[insize];
  float* ff = static_cast<float*>(f);
  double* fd = static_cast<double*>(f);
  void* g = new unsigned char[insize];
  float* gf = static_cast<float*>(g);
  double* gd = static_cast<double*>(g);
  unsigned char* zip = new unsigned char[outsize];

  // initialize uncompressed field
  if (inpath) {
    // read from file
    FILE* file = fopen(inpath, "rb");
    if (!file) {
      std::cerr << "cannot open file" << std::endl;
      return EXIT_FAILURE;
    }
    if (fread(f, typesize, mx * my * mz, file) != mx * my * mz) {
      std::cerr << "cannot read file" << std::endl;
      return EXIT_FAILURE;
    }
    fclose(file);
  }
  else {
    // evaluate function
    for (uint z = 0; z < mz; z++)
      for (uint y = 0; y < my; y++)
        for (uint x = 0; x < mx; x++) {
          double val = func(x, y, z, mx, my, mz);
          if (dp)
            fd[x + mx * (y + my * z)] = val;
          else
            ff[x + mx * (y + my * z)] = float(val);
        }
  }

  // compress data
  outsize = zfp_compress(&params, f, zip, outsize);
  if (outsize == 0) {
    std::cerr << "compression failed" << std::endl;
    return EXIT_FAILURE;
  }
  rate = CHAR_BIT * double(outsize) / (mx * my * mz);

#if IT_SEEMS_TOO_GOOD_TO_BE_TRUE
  // for skeptics: relocate compressed data
  unsigned char* copy = new unsigned char[outsize];
  std::copy(zip, zip + outsize, copy);
  delete[] zip;
  zip = copy;
#endif

  // decompress data
  if (!zfp_decompress(&params, g, zip, outsize)) {
    std::cerr << "decompression failed" << std::endl;
    return EXIT_FAILURE;
  }

  // write reconstructed data
  if (outpath) {
    FILE* file = fopen(outpath, "wb");
    if (!file) {
      std::cerr << "cannot create file" << std::endl;
      return EXIT_FAILURE;
    }
    if (fwrite(g, typesize, mx * my * mz, file) != mx * my * mz) {
      std::cerr << "cannot write file" << std::endl;
      return EXIT_FAILURE;
    }
    fclose(file);
  }

  // compute error
  double e = 0;
  double fmin = dp ? fd[0] : ff[0];
  double fmax = fmin;
  double emax = 0;
  for (uint i = 0; i < mx * my * mz; i++) {
    double d = fabs(dp ? fd[i] - gd[i] : ff[i] - gf[i]);
    emax = std::max(emax, d);
    e += d * d;
    double val = dp ? fd[i] : ff[i];
    fmin = std::min(fmin, val);
    fmax = std::max(fmax, val);
  }
  e = sqrt(e / (mx * my * mz));
  double nrmse = e / (fmax - fmin);
  double psnr = 20 * log10((fmax - fmin) / (2 * e));
  
  std::cerr << "in=" << insize << " out=" << outsize << " ratio=" << std::setprecision(3) << double(insize) / outsize << " rate=" << std::setprecision(4) << rate << " rmse=" << e << " nrmse=" << nrmse << " maxe=" << emax << " psnr=" << std::fixed << std::setprecision(2) << psnr << std::endl;

  // clean up
  delete[] static_cast<unsigned char*>(f);
  delete[] static_cast<unsigned char*>(g);
  delete[] zip;

  return EXIT_SUCCESS;
}
