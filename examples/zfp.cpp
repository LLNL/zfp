#include <algorithm>
#include <climits>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <iomanip>
#include <string>
#include "zfpcompress.h"

static double
func(uint x, uint y, uint z, uint nx, uint ny, uint nz)
{
#if 1
  return cos(2 * M_PI * (x + 0.5) / nx) *
         cos(2 * M_PI * (y + 0.5) / ny) *
         cos(2 * M_PI * (z + 0.5) / nz);
#else
  return drand48();
#endif
}

int main(int argc, char* argv[])
{
  // default settings
  bool dp = false;
  uint nx = 0;
  uint ny = 0;
  uint nz = 0;
  uint minbits = 0;
  uint maxbits = 0;
  uint maxprec = 0;
  int minexp = INT_MIN;
  char* inpath = 0;
  char* outpath = 0;

  // parse arguments
  switch (argc) {
    case 11:
      outpath = argv[10];
      // FALLTHROUGH
    case 10:
      inpath = argv[9];
      // FALLTHROUGH
    case 9:
      if (sscanf(argv[8], "%d", &minexp) != 1)
        goto usage;
      // FALLTHROUGH
    case 8:
      if (sscanf(argv[7], "%u", &maxprec) != 1)
        goto usage;
      // FALLTHROUGH
    case 7:
      if (sscanf(argv[6], "%u", &maxbits) != 1)
        goto usage;
      // FALLTHROUGH
    case 6:
      if (sscanf(argv[5], "%u", &minbits) != 1)
        goto usage;
      // FALLTHROUGH
    case 5:
      if (sscanf(argv[4], "%u", &nz) != 1)
        goto usage;
      // FALLTHROUGH
    case 4:
      if (sscanf(argv[3], "%u", &ny) != 1)
        goto usage;
      // FALLTHROUGH
    case 3:
      if (sscanf(argv[2], "%u", &nx) != 1)
        goto usage;
      if (argv[1] == std::string("-f"))
        dp = false;
      else if (argv[1] == std::string("-d"))
        dp = true;
      else
        goto usage;
      break;
    default:
    usage:
      std::cerr << "Usage: zfp <-f|-d> <nx> [ny nz minbits maxbits maxprec minexp infile outfile]" << std::endl;
      std::cerr << "-f : single precision (float type)" << std::endl;
      std::cerr << "-d : double precision (double type)" << std::endl;
      std::cerr << "nx, ny, nz : grid dimensions (set nz = 0 for 2D, ny = nz = 0 for 1D)" << std::endl;
      std::cerr << "minbits : min # bits per 4^d values in d dimensions (= maxbits for fixed rate)" << std::endl;
      std::cerr << "maxbits : max # bits per 4^d values in d dimensions" << std::endl;
      std::cerr << "maxprec : max # bits of precision per value (0 for fixed rate)" << std::endl;
      std::cerr << "minexp : min bit plane coded (error tolerance = 2^minexp; -1024 for fixed rate)" << std:: endl;
      std::cerr << "infile : optional floating-point input file to compress" << std::endl;
      std::cerr << "outfile : optional output file for reconstructed data" << std::endl;
      std::cerr << "Examples:" << std::endl;
      std::cerr << "zfp -f 100 100 100 1024 1024 : 2x fixed-rate compression of 100x100x100 floats" << std::endl;
      std::cerr << "zfp -d 1000000 0 0 128 128 : 2x fixed-rate compression of stream of 1M doubles" << std::endl;
      std::cerr << "zfp -d 1000 1000 0 0 0 32 : 32-bit precision compression of 1000x1000 doubles" << std::endl;
      std::cerr << "zfp -d 1000000 0 0 0 0 0 -16 : compression of 1M doubles with < 2^-16 error" << std::endl;
      return EXIT_FAILURE;
  }

  // effective array dimensions
  uint mx = std::max(nx, 1u);
  uint my = std::max(ny, 1u);
  uint mz = std::max(nz, 1u);

  // array dimensionality
  uint dims = 3;
  if (nz == 0) {
    if (ny == 0) {
      if (nx == 0) {
        std::cerr << "cannot compress zero-size array" << std::endl;
        return EXIT_FAILURE;
      }
      else
        dims = 1;
    }
    else
      dims = 2;
  }
  else
    dims = 3;

  // number of floating-point values per block
  uint blocksize = 1u << (2 * dims);

  // number of blocks
  uint blocks = ((mx + 3) / 4) * ((my + 3) / 4) * ((mz + 3) / 4);

  // size of floating-point type in bytes
  size_t typesize = dp ? sizeof(double) : sizeof(float);

  // correct compression parameters if zero initialized
  if (maxbits == 0)
    maxbits = blocksize * CHAR_BIT * typesize;
  if (maxprec == 0)
    maxprec = CHAR_BIT * typesize;

  // allocate space for uncompressed and compressed fields
  size_t insize = mx * my * mz * typesize;
  void* f = new unsigned char[insize];
  float* ff = static_cast<float*>(f);
  double* fd = static_cast<double*>(f);
  void* g = new unsigned char[insize];
  float* gf = static_cast<float*>(g);
  double* gd = static_cast<double*>(g);
  size_t outsize = (blocks * std::min(maxbits, blocksize * maxprec) + CHAR_BIT - 1) / CHAR_BIT;
  outsize = std::max(outsize, 2 * insize);
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
  outsize = ZFP::compress(f, zip, dp, nx, ny, nz, minbits, maxbits, maxprec, minexp);
  if (outsize == 0) {
    std::cerr << "compression failed" << std::endl;
    return EXIT_FAILURE;
  }

#if IT_SEEMS_TOO_GOOD_TO_BE_TRUE
  // for skeptics: relocate compressed data
  unsigned char* copy = new unsigned char[outsize];
  std::copy(zip, zip + outsize, copy);
  delete[] zip;
  zip = copy;
#endif

  // decompress data
  if (!ZFP::decompress(zip, g, dp, nx, ny, nz, minbits, maxbits, maxprec, minexp)) {
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
  
  std::cerr << "in=" << insize << " out=" << outsize << " ratio=" << std::setprecision(3) << double(insize) / outsize << " rmse=" << std::setprecision(4) << e << " nrmse=" << nrmse << " maxe=" << emax << " psnr=" << psnr << std::endl;

  // clean up
  delete[] static_cast<unsigned char*>(f);
  delete[] static_cast<unsigned char*>(g);
  delete[] zip;

  return EXIT_SUCCESS;
}
