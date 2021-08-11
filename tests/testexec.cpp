#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include "zfp.h"

typedef long double real;

inline const char*
exec_string(zfp_exec_policy exec)
{
  static const char* string[] = {
    "serial", "OpenMP", "CUDA", "HIP",
  };
  return string[exec];
}

inline real
coord(uint i, uint n)
{
  return real(2) * (i + real(0.5)) / n - real(1);
}

template <typename type>
inline type
cube(type x)
{
  return x * x * x;
}

template <typename type>
inline type*
gen3d(uint nx, uint ny, uint nz)
{
  type* data = new type[nx * ny * nz];
  for (uint k = 0; k < nz; k++) {
    real z = coord(k, nz);
    for (uint j = 0; j < ny; j++) {
      real y = coord(j, ny);
      for (uint i = 0; i < nx; i++) {
        real x = coord(i, nx);
        real f = cube((-x + y + z) * (x - y + z) * (x + y - z));
        *data++ = static_cast<type>(f);
      }
    }
  }
  return data - nx * ny * nz;
}

inline uint32
checksum(const void* p, size_t n)
{
  uint32 h = 0;
  for (const uchar* q = static_cast<const uchar*>(p); n; q++, n--) {
    h += *q;
    h += h << 10;
    h ^= h >>  6;
  }
  h += h <<  3;
  h ^= h >> 11;
  h += h << 15;
  return h;
}

inline int
test(zfp_mode mode, int param, zfp_stream* zfp, const zfp_field* field, zfp_exec_policy exec, size_t& size, uint32& sum)
{
  double rate = param;
  uint prec = param;
  double tol = std::ldexp(1., -param);

  switch (mode) {
    case zfp_mode_fixed_rate:
      zfp_stream_set_rate(zfp, rate, zfp_field_type(field), zfp_field_dimensionality(field), zfp_false);
      if (exec == zfp_exec_serial)
        fprintf(stderr, "rate=%g ", rate);
      break;
    case zfp_mode_fixed_precision:
      zfp_stream_set_precision(zfp, prec);
      if (exec == zfp_exec_serial)
        fprintf(stderr, "prec=%d ", prec);
      break;
    case zfp_mode_fixed_accuracy:
      zfp_stream_set_accuracy(zfp, tol);
      if (exec == zfp_exec_serial)
        fprintf(stderr, "tol=%g ", tol);
      break;
    default:
      fprintf(stderr, "invalid execution mode\n");
      return 1;
  }

  zfp_stream_rewind(zfp);
  size_t zsize = zfp_compress(zfp, field);
  uint32 zsum = checksum(stream_data(zfp_stream_bit_stream(zfp)), zsize);

  if (zsize > stream_capacity(zfp_stream_bit_stream(zfp))) {
    fprintf(stderr, "ERROR: buffer overrun\n");
    return 1;
  }

  switch (exec) {
    case zfp_exec_serial:
      fprintf(stderr, "size=%lu checksum=%#x\n", (unsigned long)zsize, zsum);
      size = zsize;
      sum = zsum;
      break;
    default:
      if (zsize != size) {
        fprintf(stderr, "  ERROR: %s size (%lu) does not match serial\n", exec_string(exec), (unsigned long)zsize);
        return 1;
      }
      else {
        if (zsum != sum) {
          fprintf(stderr, "  ERROR: %s checksum (%#x) does not match serial\n", exec_string(exec), zsum);
          return 1;
        }
      }
      break;
  }

  return 0;
}

static int
usage()
{
  fprintf(stderr, "Usage: testexec [types [modes [nx [ny [nz]]]]]\n");
  fprintf(stderr, "Types (any combination of {f,d}):\n");
  fprintf(stderr, "  f : float\n");
  fprintf(stderr, "  d : double\n");
  fprintf(stderr, "Modes (any combination of {r,p,a}):\n");
  fprintf(stderr, "  r : fixed rate\n");
  fprintf(stderr, "  p : fixed precision\n");
  fprintf(stderr, "  a : fixed accuracy\n");
  fprintf(stderr, "Defaults: types=fd modes=rpa nx=ny=nz=32\n");
  return EXIT_FAILURE;
}

int main(int argc, char* argv[])
{
  bool test_float = true;
  bool test_double = true;
  bool test_rate = true;
  bool test_prec = true;
  bool test_acc = true;
  uint nx = 32;
  uint ny = 32;
  uint nz = 32;

  const zfp_exec_policy exec[] = {
    zfp_exec_serial,
#if ZFP_WITH_OPENMP
    zfp_exec_omp,
#endif
#if ZFP_WITH_CUDA
    zfp_exec_cuda,
#endif
#if ZFP_WITH_HIP
    zfp_exec_hip,
#endif
  };
  const size_t exec_modes = sizeof(exec) / sizeof(exec[0]);

  switch (argc) {
    case 6:
      if (sscanf(argv[5], "%u", &nz) != 1)
        return usage();
      // FALLTHROUGH
    case 5:
      if (sscanf(argv[4], "%u", &ny) != 1)
        return usage();
      // FALLTHROUGH
    case 4:
      if (sscanf(argv[3], "%u", &nx) != 1)
        return usage();
      // FALLTHROUGH
    case 3:
      test_rate = test_prec = test_acc = false;
      for (size_t i = 0; i < strlen(argv[2]); i++)
        switch (argv[2][i]) {
          case 'r':
            test_rate = true;
            break;
          case 'p':
            test_prec = true;
            break;
          case 'a':
            test_acc = true;
            break;
          default:
            return usage();
        }
      // FALLTHROUGH
    case 2:
      test_float = test_double = false;
      for (size_t i = 0; i < strlen(argv[1]); i++)
        switch (argv[1][i]) {
          case 'f':
            test_float = true;
            break;
          case 'd':
            test_double = true;
            break;
          default:
            return usage();
        }
      // FALLTHROUGH
    case 1:
      break;
  }

  size_t n = nx * ny * nz;

  fprintf(stderr, "testing execution policies {");
  for (size_t i = 0; i < exec_modes; i++)
    fprintf(stderr, "%s%s", exec_string(exec[i]), i == exec_modes - 1 ? "}" : ", ");
  fprintf(stderr, "\n\n");

  uint tests[exec_modes] = {};
  uint failures[exec_modes] = {};

  // loop over float, double
  for (int data_type = 0; data_type < 2; data_type++) {
    zfp_type type = zfp_type_none;
    void* array = 0;

    if (data_type == 0) {
      if (!test_float)
        continue;
      type = zfp_type_float;
      array = gen3d<float>(nx, ny, nz);
    }
    else {
      if (!test_double)
        continue;
      type = zfp_type_double;
      array = gen3d<double>(nx, ny, nz);
    }

    // loop over 1D, 2D, 3D
    for (uint dims = 1; dims <= 3; dims++) {
      fprintf(stderr, "type=%s dimensionality=%u\n", data_type == 0 ? "float" : "double", dims);

      // set up uncompressed field
      zfp_field* field = 0;
      switch (dims) {
        case 1:
          field = zfp_field_1d(array, type, nx * ny * nz);
          break;
        case 2:
          field = zfp_field_2d(array, type, nx, ny * nz);
          break;
        case 3:
          field = zfp_field_3d(array, type, nx, ny, nz);
          break;
      }

      // allocate compressed-data buffers and bit streams
      size_t bufsize = 2 * n * sizeof(uint64);
      void* buffer[exec_modes];
      bitstream* stream[exec_modes];
      zfp_stream* zfp[exec_modes];
      for (size_t i = 0; i < exec_modes; i++) {
        buffer[i] = new uint64[2 * n];
        stream[i] = stream_open(buffer[i], bufsize);
        zfp[i] = zfp_stream_open(stream[i]);
        if (!zfp_stream_set_execution(zfp[i], exec[i])) {
          fprintf(stderr, "%s execution not available\n", exec_string(exec[i]));
          return EXIT_FAILURE;
        }
      }

      if (test_rate) {
        // test all rates
        for (int rate = 1; rate <= (type == zfp_type_float ? 32 : 64); rate++) {
          size_t size;
          uint32 sum;
          for (size_t j = 0; j < exec_modes; j++) {
            failures[j] += test(zfp_mode_fixed_rate, rate, zfp[j], field, exec[j], size, sum);
            tests[j]++;
          }
        }
      }

      if (test_prec) {
        // test all precisions
        for (int prec = 1; prec <= (type == zfp_type_float ? 32 : 64); prec++) {
          size_t size;
          uint32 sum;
          for (size_t j = 0; j < exec_modes; j++) {
            failures[j] += test(zfp_mode_fixed_precision, prec, zfp[j], field, exec[j], size, sum);
            tests[j]++;
          }
        }
      }

      if (test_acc) {
        // test all tolerances
        for (int acc = 1; acc <= (type == zfp_type_float ? 32 : 64); acc++) {
          size_t size;
          uint32 sum;
          for (size_t j = 0; j < exec_modes; j++) {
            failures[j] += test(zfp_mode_fixed_accuracy, acc, zfp[j], field, exec[j], size, sum);
            tests[j]++;
          }
        }
      }

      fprintf(stderr, "\n");
    }
  }

  // count total tests and failures
  uint total_tests = 0;
  uint total_failures = 0;
  for (size_t j = 0; j < exec_modes; j++) {
    total_tests += tests[j];
    total_failures += failures[j];
  }

  // print results
  if (total_failures) {
    fprintf(stderr, "%u of %u tests failed\n", total_failures, total_tests);
    for (size_t j = 0; j < exec_modes; j++)
      if (failures[j])
        fprintf(stderr, "  %u of %u %s tests failed\n", failures[j], tests[j], exec_string(exec[j]));
      else
        fprintf(stderr, "  all %u %s tests passed\n", tests[j], exec_string(exec[j]));
    return EXIT_FAILURE;
  }
  else {
    fprintf(stderr, "all %u tests passed\n", total_tests);
    return EXIT_SUCCESS;
  }
}
