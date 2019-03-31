#include <float.h>
#include <limits.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "zfp.h"
#include "zfp/macros.h"

/*
File I/O is done using the following combinations of i, o, s, and z:
- i   : read uncompressed
- z   : read compressed
- i, s: read uncompressed, print stats
- i, o: read and write uncompressed
- i, z: read uncompressed, write compressed
- z, o: read compressed, write uncompressed

The 7 major tasks to be accomplished are:
- read uncompressed:  i
- read compressed:    !i
- compress:           i
- write compressed:   i && z
- decompress:         o || s || (!i && z)
- write uncompressed: o
- compute stats:      s
*/

/* compute and print reconstruction error */
static void
print_error(const void* fin, const void* fout, zfp_type type, size_t n)
{
  const int32* i32i = (const int32*)fin;
  const int64* i64i = (const int64*)fin;
  const float* f32i = (const float*)fin;
  const double* f64i = (const double*)fin;
  const int32* i32o = (const int32*)fout;
  const int64* i64o = (const int64*)fout;
  const float* f32o = (const float*)fout;
  const double* f64o = (const double*)fout;
  double fmin = +DBL_MAX;
  double fmax = -DBL_MAX;
  double erms = 0;
  double ermsn = 0;
  double emax = 0;
  double psnr = 0;
  size_t i;

  for (i = 0; i < n; i++) {
    double d, val;
    switch (type) {
      case zfp_type_int32:
        d = fabs((double)(i32i[i] - i32o[i]));
        val = (double)i32i[i];
        break;
      case zfp_type_int64:
        d = fabs((double)(i64i[i] - i64o[i]));
        val = (double)i64i[i];
        break;
      case zfp_type_float:
        d = fabs((double)(f32i[i] - f32o[i]));
        val = (double)f32i[i];
        break;
      case zfp_type_double:
        d = fabs(f64i[i] - f64o[i]);
        val = f64i[i];
        break;
      default:
        return;
    }
    emax = MAX(emax, d);
    erms += d * d;
    fmin = MIN(fmin, val);
    fmax = MAX(fmax, val);
  }
  erms = sqrt(erms / n);
  ermsn = erms / (fmax - fmin);
  psnr = 20 * log10((fmax - fmin) / (2 * erms));
  fprintf(stderr, " rmse=%.4g nrmse=%.4g maxe=%.4g psnr=%.2f", erms, ermsn, emax, psnr);
}

static void
usage()
{
  fprintf(stderr, "%s\n", zfp_version_string);
  fprintf(stderr, "Usage: zfp <options>\n");
  fprintf(stderr, "General options:\n");
  fprintf(stderr, "  -h : read/write array and compression parameters from/to compressed header\n");
  fprintf(stderr, "  -q : quiet mode; suppress output\n");
  fprintf(stderr, "  -s : print error statistics\n");
  fprintf(stderr, "Input and output:\n");
  fprintf(stderr, "  -i <path> : uncompressed binary input file (\"-\" for stdin)\n");
  fprintf(stderr, "  -o <path> : decompressed binary output file (\"-\" for stdout)\n");
  fprintf(stderr, "  -z <path> : compressed input (w/o -i) or output file (\"-\" for stdin/stdout)\n");
  fprintf(stderr, "Array type and dimensions (needed with -i):\n");
  fprintf(stderr, "  -f : single precision (float type)\n");
  fprintf(stderr, "  -d : double precision (double type)\n");
  fprintf(stderr, "  -t <i32|i64|f32|f64> : integer or floating scalar type\n");
  fprintf(stderr, "  -1 <nx> : dimensions for 1D array a[nx]\n");
  fprintf(stderr, "  -2 <nx> <ny> : dimensions for 2D array a[ny][nx]\n");
  fprintf(stderr, "  -3 <nx> <ny> <nz> : dimensions for 3D array a[nz][ny][nx]\n");
  fprintf(stderr, "  -4 <nx> <ny> <nz> <nw> : dimensions for 4D array a[nw][nz][ny][nx]\n");
  fprintf(stderr, "Compression parameters (needed with -i):\n");
  fprintf(stderr, "  -R : reversible (lossless) compression\n");
  fprintf(stderr, "  -r <rate> : fixed rate (# compressed bits per floating-point value)\n");
  fprintf(stderr, "  -p <precision> : fixed precision (# uncompressed bits per value)\n");
  fprintf(stderr, "  -a <tolerance> : fixed accuracy (absolute error tolerance)\n");
  fprintf(stderr, "  -c <minbits> <maxbits> <maxprec> <minexp> : advanced usage\n");
  fprintf(stderr, "      minbits : min # bits per 4^d values in d dimensions\n");
  fprintf(stderr, "      maxbits : max # bits per 4^d values in d dimensions (0 for unlimited)\n");
  fprintf(stderr, "      maxprec : max # bits of precision per value (0 for full)\n");
  fprintf(stderr, "      minexp : min bit plane # coded (-1074 for all bit planes)\n");
  fprintf(stderr, "Execution parameters:\n");
  fprintf(stderr, "  -x serial : serial compression (default)\n");
  fprintf(stderr, "  -x omp[=threads[,chunk_size]] : OpenMP parallel compression\n");
  fprintf(stderr, "  -x cuda : CUDA fixed rate parallel compression/decompression\n");
  fprintf(stderr, "Examples:\n");
  fprintf(stderr, "  -i file : read uncompressed file and compress to memory\n");
  fprintf(stderr, "  -z file : read compressed file and decompress to memory\n");
  fprintf(stderr, "  -i ifile -z zfile : read uncompressed ifile, write compressed zfile\n");
  fprintf(stderr, "  -z zfile -o ofile : read compressed zfile, write decompressed ofile\n");
  fprintf(stderr, "  -i ifile -o ofile : read ifile, compress, decompress, write ofile\n");
  fprintf(stderr, "  -i file -s : read uncompressed file, compress to memory, print stats\n");
  fprintf(stderr, "  -i - -o - -s : read stdin, compress, decompress, write stdout, print stats\n");
  fprintf(stderr, "  -f -3 100 100 100 -r 16 : 2x fixed-rate compression of 100x100x100 floats\n");
  fprintf(stderr, "  -d -1 1000000 -r 32 : 2x fixed-rate compression of 1M doubles\n");
  fprintf(stderr, "  -d -2 1000 1000 -p 32 : 32-bit precision compression of 1000x1000 doubles\n");
  fprintf(stderr, "  -d -1 1000000 -a 1e-9 : compression of 1M doubles with < 1e-9 max error\n");
  fprintf(stderr, "  -d -1 1000000 -c 64 64 0 -1074 : 4x fixed-rate compression of 1M doubles\n");
  fprintf(stderr, "  -x omp=16,256 : parallel compression with 16 threads, 256-block chunks\n");
  exit(EXIT_FAILURE);
}

int main(int argc, char* argv[])
{
  /* default settings */
  zfp_type type = zfp_type_none;
  size_t typesize = 0;
  uint dims = 0;
  uint nx = 0;
  uint ny = 0;
  uint nz = 0;
  uint nw = 0;
  size_t count = 0;
  double rate = 0;
  uint precision = 0;
  double tolerance = 0;
  uint minbits = ZFP_MIN_BITS;
  uint maxbits = ZFP_MAX_BITS;
  uint maxprec = ZFP_MAX_PREC;
  int minexp = ZFP_MIN_EXP;
  int header = 0;
  int quiet = 0;
  int stats = 0;
  char* inpath = 0;
  char* zfppath = 0;
  char* outpath = 0;
  char mode = 0;
  zfp_exec_policy exec = zfp_exec_serial;
  uint threads = 0;
  uint chunk_size = 0;

  /* local variables */
  int i;
  zfp_field* field = NULL;
  zfp_stream* zfp = NULL;
  bitstream* stream = NULL;
  void* fi = NULL;
  void* fo = NULL;
  void* buffer = NULL;
  size_t rawsize = 0;
  size_t zfpsize = 0;
  size_t bufsize = 0;

  if (argc == 1)
    usage();

  /* parse command-line arguments */
  for (i = 1; i < argc; i++) {
    if (argv[i][0] != '-' || argv[i][2])
      usage();
    switch (argv[i][1]) {
      case '1':
        if (++i == argc || sscanf(argv[i], "%u", &nx) != 1)
          usage();
        ny = nz = nw = 1;
        dims = 1;
        break;
      case '2':
        if (++i == argc || sscanf(argv[i], "%u", &nx) != 1 ||
            ++i == argc || sscanf(argv[i], "%u", &ny) != 1)
          usage();
        nz = nw = 1;
        dims = 2;
        break;
      case '3':
        if (++i == argc || sscanf(argv[i], "%u", &nx) != 1 ||
            ++i == argc || sscanf(argv[i], "%u", &ny) != 1 ||
            ++i == argc || sscanf(argv[i], "%u", &nz) != 1)
          usage();
        nw = 1;
        dims = 3;
        break;
      case '4':
        if (++i == argc || sscanf(argv[i], "%u", &nx) != 1 ||
            ++i == argc || sscanf(argv[i], "%u", &ny) != 1 ||
            ++i == argc || sscanf(argv[i], "%u", &nz) != 1 ||
            ++i == argc || sscanf(argv[i], "%u", &nw) != 1)
          usage();
        dims = 4;
        break;
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
        type = zfp_type_double;
        break;
      case 'f':
        type = zfp_type_float;
        break;
      case 'h':
        header = 1;
        break;
      case 'i':
        if (++i == argc)
          usage();
        inpath = argv[i];
        break;
      case 'o':
        if (++i == argc)
          usage();
        outpath = argv[i];
        break;
      case 'p':
        if (++i == argc || sscanf(argv[i], "%u", &precision) != 1)
          usage();
        mode = 'p';
        break;
      case 'q':
        quiet = 1;
        break;
      case 'r':
        if (++i == argc || sscanf(argv[i], "%lf", &rate) != 1)
          usage();
        mode = 'r';
        break;
      case 'R':
        mode = 'R';
        break;
      case 's':
        stats = 1;
        break;
      case 't':
        if (++i == argc)
          usage();
        if (!strcmp(argv[i], "i32"))
          type = zfp_type_int32;
        else if (!strcmp(argv[i], "i64"))
          type = zfp_type_int64;
        else if (!strcmp(argv[i], "f32"))
          type = zfp_type_float;
        else if (!strcmp(argv[i], "f64"))
          type = zfp_type_double;
        else
          usage();
        break;
      case 'x':
        if (++i == argc)
          usage();
        if (!strcmp(argv[i], "serial"))
          exec = zfp_exec_serial;
        else if (sscanf(argv[i], "omp=%u,%u", &threads, &chunk_size) == 2)
          exec = zfp_exec_omp;
        else if (sscanf(argv[i], "omp=%u", &threads) == 1) {
          exec = zfp_exec_omp;
          chunk_size = 0;
        }
        else if (!strcmp(argv[i], "omp")) {
          exec = zfp_exec_omp;
          threads = 0;
          chunk_size = 0;
        }
        else if (!strcmp(argv[i], "cuda"))
          exec = zfp_exec_cuda;
        else
          usage();
        break;
      case 'z':
        if (++i == argc)
          usage();
        zfppath = argv[i];
        break;
      default:
        usage();
        break;
    }
  }

  typesize = zfp_type_size(type);
  count = (size_t)nx * (size_t)ny * (size_t)nz * (size_t)nw;

  /* make sure one of the array dimensions is not zero */
  if (!count && dims) {
    fprintf(stderr, "array size must be nonzero\n");
    return EXIT_FAILURE;
  }

  /* make sure we have an input file */
  if (!inpath && !zfppath) {
    fprintf(stderr, "must specify uncompressed or compressed input file via -i or -z\n");
    return EXIT_FAILURE;
  }

  /* make sure we (will) know scalar type */
  if (!typesize) {
    if (inpath) {
      fprintf(stderr, "must specify scalar type via -f, -d, or -t to compress\n");
      return EXIT_FAILURE;
    }
    else if (!header) {
      fprintf(stderr, "must specify scalar type via -f, -d, or -t or header via -h to decompress\n");
      return EXIT_FAILURE;
    }
  }

  /* make sure we (will) know array dimensions */
  if (!dims) {
    if (inpath) {
      fprintf(stderr, "must specify array dimensions via -1, -2, -3, or -4 to compress\n");
      return EXIT_FAILURE;
    }
    else if (!header) {
      fprintf(stderr, "must specify array dimensions via -1, -2, -3, or -4 or header via -h to decompress\n");
      return EXIT_FAILURE;
    }
  }

  /* make sure we (will) know (de)compression mode and parameters */
  if (!mode) {
    if (inpath) {
      fprintf(stderr, "must specify compression parameters via -a, -c, -p, or -r to compress\n");
      return EXIT_FAILURE;
    }
    else if (!header) {
      fprintf(stderr, "must specify compression parameters via -a, -c, -p, or -r or header via -h to decompress\n");
      return EXIT_FAILURE;
    }
  }

  /* make sure we have input file for stats */
  if (stats && !inpath) {
    fprintf(stderr, "must specify input file via -i to compute stats\n");
    return EXIT_FAILURE;
  }

  /* make sure meta data comes from header or command line, not both */
  if (!inpath && zfppath && header && (typesize || dims)) {
    fprintf(stderr, "cannot specify both field type/size and header\n");
    return EXIT_FAILURE;
  }

  zfp = zfp_stream_open(NULL);
  field = zfp_field_alloc();

  /* read uncompressed or compressed file */
  if (inpath) {
    /* read uncompressed input file */
    FILE* file = !strcmp(inpath, "-") ? stdin : fopen(inpath, "rb");
    if (!file) {
      fprintf(stderr, "cannot open input file\n");
      return EXIT_FAILURE;
    }
    rawsize = typesize * count;
    fi = malloc(rawsize);
    if (!fi) {
      fprintf(stderr, "cannot allocate memory\n");
      return EXIT_FAILURE;
    }
    if (fread(fi, typesize, count, file) != count) {
      fprintf(stderr, "cannot read input file\n");
      return EXIT_FAILURE;
    }
    fclose(file);
    zfp_field_set_pointer(field, fi);
  }
  else {
    /* read compressed input file in increasingly large chunks */
    FILE* file = !strcmp(zfppath, "-") ? stdin : fopen(zfppath, "rb");
    if (!file) {
      fprintf(stderr, "cannot open compressed file\n");
      return EXIT_FAILURE;
    }
    bufsize = 0x100;
    do {
      bufsize *= 2;
      buffer = realloc(buffer, bufsize);
      if (!buffer) {
        fprintf(stderr, "cannot allocate memory\n");
        return EXIT_FAILURE;
      }
      zfpsize += fread((uchar*)buffer + zfpsize, 1, bufsize - zfpsize, file);
    } while (zfpsize == bufsize);
    if (ferror(file)) {
      fprintf(stderr, "cannot read compressed file\n");
      return EXIT_FAILURE;
    }
    fclose(file);

    /* associate bit stream with buffer */
    stream = stream_open(buffer, bufsize);
    if (!stream) {
      fprintf(stderr, "cannot open compressed stream\n");
      return EXIT_FAILURE;
    }
    zfp_stream_set_bit_stream(zfp, stream);
  }

  /* set field dimensions and (de)compression parameters */
  if (inpath || !header) {
    /* initialize uncompressed field */
    zfp_field_set_type(field, type);
    switch (dims) {
      case 1:
        zfp_field_set_size_1d(field, nx);
        break;
      case 2:
        zfp_field_set_size_2d(field, nx, ny);
        break;
      case 3:
        zfp_field_set_size_3d(field, nx, ny, nz);
        break;
      case 4:
        zfp_field_set_size_4d(field, nx, ny, nz, nw);
        break;
    }

    /* set (de)compression mode */
    switch (mode) {
      case 'R':
        zfp_stream_set_reversible(zfp);
        break;
      case 'a':
        zfp_stream_set_accuracy(zfp, tolerance);
        break;
      case 'p':
        zfp_stream_set_precision(zfp, precision);
        break;
      case 'r':
        zfp_stream_set_rate(zfp, rate, type, dims, 0);
        break;
      case 'c':
        if (!maxbits)
          maxbits = ZFP_MAX_BITS;
        if (!maxprec)
          maxprec = zfp_field_precision(field);
        if (!zfp_stream_set_params(zfp, minbits, maxbits, maxprec, minexp)) {
          fprintf(stderr, "invalid compression parameters\n");
          return EXIT_FAILURE;
        }
        break;
    }
  }

  /* specify execution policy */
  switch (exec) {
    case zfp_exec_cuda:
      if (!zfp_stream_set_execution(zfp, exec)) {
        fprintf(stderr, "cuda execution not available\n");
        return EXIT_FAILURE;
      }
      break;
    case zfp_exec_omp:
      if (!zfp_stream_set_execution(zfp, exec) ||
          !zfp_stream_set_omp_threads(zfp, threads) ||
          !zfp_stream_set_omp_chunk_size(zfp, chunk_size)) {
        fprintf(stderr, "OpenMP execution not available\n");
        return EXIT_FAILURE;
      }
      break;
    case zfp_exec_serial:
    default:
      if (!zfp_stream_set_execution(zfp, exec)) {
        fprintf(stderr, "serial execution not available\n");
        return EXIT_FAILURE;
      }
      break;
  }

  /* compress input file if provided */
  if (inpath) {
    /* allocate buffer for compressed data */
    bufsize = zfp_stream_maximum_size(zfp, field);
    if (!bufsize) {
      fprintf(stderr, "invalid compression parameters\n");
      return EXIT_FAILURE;
    }
    buffer = malloc(bufsize);
    if (!buffer) {
      fprintf(stderr, "cannot allocate memory\n");
      return EXIT_FAILURE;
    }

    /* associate compressed bit stream with memory buffer */
    stream = stream_open(buffer, bufsize);
    if (!stream) {
      fprintf(stderr, "cannot open compressed stream\n");
      return EXIT_FAILURE;
    }
    zfp_stream_set_bit_stream(zfp, stream);

    /* optionally write header */
    if (header && !zfp_write_header(zfp, field, ZFP_HEADER_FULL)) {
      fprintf(stderr, "cannot write header\n");
      return EXIT_FAILURE;
    }

    /* compress data */
    zfpsize = zfp_compress(zfp, field);
    if (zfpsize == 0) {
      fprintf(stderr, "compression failed\n");
      return EXIT_FAILURE;
    }

    /* optionally write compressed data */
    if (zfppath) {
      FILE* file = !strcmp(zfppath, "-") ? stdout : fopen(zfppath, "wb");
      if (!file) {
        fprintf(stderr, "cannot create compressed file\n");
        return EXIT_FAILURE;
      }
      if (fwrite(buffer, 1, zfpsize, file) != zfpsize) {
        fprintf(stderr, "cannot write compressed file\n");
        return EXIT_FAILURE;
      }
      fclose(file);
    }
  }

  /* decompress data if necessary */
  if ((!inpath && zfppath) || outpath || stats) {
    /* obtain metadata from header when present */
    zfp_stream_rewind(zfp);
    if (header) {
      if (!zfp_read_header(zfp, field, ZFP_HEADER_FULL)) {
        fprintf(stderr, "incorrect or missing header\n");
        return EXIT_FAILURE;
      }
      type = field->type;
      switch (type) {
        case zfp_type_float:
          typesize = sizeof(float);
          break;
        case zfp_type_double:
          typesize = sizeof(double);
          break;
        default:
          fprintf(stderr, "unsupported type\n");
          return EXIT_FAILURE;
      }
      nx = MAX(field->nx, 1u);
      ny = MAX(field->ny, 1u);
      nz = MAX(field->nz, 1u);
      nw = MAX(field->nw, 1u);
      count = (size_t)nx * (size_t)ny * (size_t)nz * (size_t)nw;
    }

    /* allocate memory for decompressed data */
    rawsize = typesize * count;
    fo = malloc(rawsize);
    if (!fo) {
      fprintf(stderr, "cannot allocate memory\n");
      return EXIT_FAILURE;
    }
    zfp_field_set_pointer(field, fo);

    /* decompress data */
    while (!zfp_decompress(zfp, field)) {
      /* fall back on serial decompression if execution policy not supported */
      if (inpath && zfp_stream_execution(zfp) != zfp_exec_serial) {
        if (!zfp_stream_set_execution(zfp, zfp_exec_serial)) {
          fprintf(stderr, "cannot change execution policy\n");
          return EXIT_FAILURE;
        }
      }
      else {
        fprintf(stderr, "decompression failed\n");
        return EXIT_FAILURE;
      }
    }

    /* optionally write reconstructed data */
    if (outpath) {
      FILE* file = !strcmp(outpath, "-") ? stdout : fopen(outpath, "wb");
      if (!file) {
        fprintf(stderr, "cannot create output file\n");
        return EXIT_FAILURE;
      }
      if (fwrite(fo, typesize, count, file) != count) {
        fprintf(stderr, "cannot write output file\n");
        return EXIT_FAILURE;
      }
      fclose(file);
    }
  }

  /* print compression and error statistics */
  if (!quiet) {
    const char* type_name[] = { "int32", "int64", "float", "double" };
    fprintf(stderr, "type=%s nx=%u ny=%u nz=%u nw=%u", type_name[type - zfp_type_int32], nx, ny, nz, nw);
    fprintf(stderr, " raw=%lu zfp=%lu ratio=%.3g rate=%.4g", (unsigned long)rawsize, (unsigned long)zfpsize, (double)rawsize / zfpsize, CHAR_BIT * (double)zfpsize / count);
    if (stats)
      print_error(fi, fo, type, count);
    fprintf(stderr, "\n");
  }

  /* free allocated storage */
  zfp_field_free(field);
  zfp_stream_close(zfp);
  stream_close(stream);
  free(buffer);
  free(fi);
  free(fo);

  return EXIT_SUCCESS;
}
