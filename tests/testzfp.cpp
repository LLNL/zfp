#include <algorithm>
#include <cmath>
#include <ctime>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <sstream>
#include <string>
#include "zfp.h"
#include "zfparray1.h"
#include "zfparray2.h"
#include "zfparray3.h"

enum ArraySize {
  Small  = 0, // 2^12 = 4096 scalars (2^12 = (2^6)^2 = (2^4)^3 = (2^3)^4)
  Large  = 1  // 2^24 = 16 M scalars (2^24 = (2^12)^2 = (2^8)^3 = (2^6)^4)
};

enum ScalarType {
  Float =  0, // 32-bit single precision
  Double = 1  // 64-bit double precision
};

static const int width = 72; // characters per line

inline uint
mask(uint i)
{
  return 1u << i;
}

inline uint
test_size(ArraySize size)
{
  return 2u << size;
}

// refine 1D array f[m] to g[2m]
inline void
refine1d(int* g, const int* f, size_t m)
{
  const int weight[4] = { -1, 9, 9, -1 };
  const size_t n = 2 * m;

  for (size_t x = 0; x < n; x++) {
    int s = 0;
    for (int i = 0; i < 4; i++) {
      size_t xx = x & 1u ? (x / 2 + i - 1 + m) % m : x / 2;
      s += weight[i] * f[xx];
    }
    g[x] = s / 16;
  }
}

// refine 2D array f[m][m] to g[2m][2m]
inline void
refine2d(int* g, const int* f, size_t m)
{
  const int weight[4] = { -1, 9, 9, -1 };
  const size_t n = 2 * m;

  for (size_t y = 0; y < n; y++)
    for (size_t x = 0; x < n; x++) {
      int s = 0;
      for (int j = 0; j < 4; j++) {
        size_t yy = y & 1u ? (y / 2 + j - 1 + m) % m : y / 2;
        for (int i = 0; i < 4; i++) {
          size_t xx = x & 1u ? (x / 2 + i - 1 + m) % m : x / 2;
          s += weight[i] * weight[j] * f[xx + m * yy];
        }
      }
      g[x + n * y] = s / (16 * 16);
    }
}

// refine 3D array f[m][m][m] to g[2m][2m][2m]
inline void
refine3d(int* g, const int* f, size_t m)
{
  const int weight[4] = { -1, 9, 9, -1 };
  const size_t n = 2 * m;

  for (size_t z = 0; z < n; z++)
    for (size_t y = 0; y < n; y++)
      for (size_t x = 0; x < n; x++) {
        int s = 0;
        for (int k = 0; k < 4; k++) {
          size_t zz = z & 1u ? (z / 2 + k - 1 + m) % m : z / 2;
          for (int j = 0; j < 4; j++) {
            size_t yy = y & 1u ? (y / 2 + j - 1 + m) % m : y / 2;
            for (int i = 0; i < 4; i++) {
              size_t xx = x & 1u ? (x / 2 + i - 1 + m) % m : x / 2;
              s += weight[i] * weight[j] * weight[k] * f[xx + m * (yy + m * zz)];
            }
          }
        }
        g[x + n * (y + n * z)] = s / (16 * 16 * 16);
      }
}

// refine 4D array f[m][m][m][m] to g[2m][2m][2m][2m]
inline void
refine4d(int* g, const int* f, size_t m)
{
  const int weight[4] = { -1, 9, 9, -1 };
  const size_t n = 2 * m;

  for (size_t w = 0; w < n; w++)
    for (size_t z = 0; z < n; z++)
      for (size_t y = 0; y < n; y++)
        for (size_t x = 0; x < n; x++) {
          int s = 0;
          for (int l = 0; l < 4; l++) {
            size_t ww = w & 1u ? (w / 2 + l - 1 + m) % m : w / 2;
            for (int k = 0; k < 4; k++) {
              size_t zz = z & 1u ? (z / 2 + k - 1 + m) % m : z / 2;
              for (int j = 0; j < 4; j++) {
                size_t yy = y & 1u ? (y / 2 + j - 1 + m) % m : y / 2;
                for (int i = 0; i < 4; i++) {
                  size_t xx = x & 1u ? (x / 2 + i - 1 + m) % m : x / 2;
                  s += weight[i] * weight[j] * weight[k] * weight[l] * f[xx + m * (yy + m * (zz + m * ww))];
                }
              }
            }
          }
          g[x + n * (y + n * (z + n * w))] = s / (16 * 16 * 16 * 16);
        }
}

template <typename real>
inline void
convert_ints_to_reals(real* data, const int* f, size_t n)
{
  for (size_t i = 0; i < n; i++)
    data[i] = std::ldexp(real(f[i]), -12);
}

// generate 1D test array of size n
template <typename real>
inline bool
gen_array_1d(real* data, size_t n)
{
  // ensure n >= 4 is a power of two
  if (n < 4 || n & (n - 1))
    return false;

  // initialize 4-element integer array
  int* f = new int[n];
  std::fill(f, f + 4, 0);
  for (uint x = 1; x < 3; x++)
    f[x] = 0x10000 * (1 - 2 * int(x & 1u));

  // refine to n-element array
  int* g = new int[n];
  for (size_t m = 4; m < n; m *= 2) {
    refine1d(g, f, m);
    std::swap(f, g);
  }
  delete[] g;

  // convert ints to real type
  convert_ints_to_reals(data, f, n);
  delete[] f;

  return true;
}

// generate 2D test array of size n^2
template <typename real>
inline bool
gen_array_2d(real* data, size_t n)
{
  // ensure n >= 4 is a power of two
  if (n < 4 || n & (n - 1))
    return false;

  // initialize 4x4 integer array
  int* f = new int[n * n];
  std::fill(f, f + 4 * 4, 0);
  for (uint y = 1; y < 3; y++)
    for (uint x = 1; x < 3; x++)
      f[x + 4 * y] = 0x10000 * (1 - 2 * int((x ^ y) & 1u));

  // refine to n^2 array
  int* g = new int[n * n];
  for (size_t m = 4; m < n; m *= 2) {
    refine2d(g, f, m);
    std::swap(f, g);
  }
  delete[] g;

  // convert ints to real type
  convert_ints_to_reals(data, f, n * n);
  delete[] f;

  return true;
}

// generate 3D test array of size n^3
template <typename real>
inline bool
gen_array_3d(real* data, size_t n)
{
  // ensure n >= 4 is a power of two
  if (n < 4 || n & (n - 1))
    return false;

  // initialize 4x4x4 integer array
  int* f = new int[n * n * n];
  std::fill(f, f + 4 * 4 * 4, 0);
  for (uint z = 1; z <= 2u; z++)
    for (uint y = 1; y <= 2u; y++)
      for (uint x = 1; x <= 2u; x++)
        f[x + 4 * (y + 4 * z)] = 0x10000 * (1 - 2 * int((x ^ y ^ z) & 1u));

  // refine to n^3 array
  int* g = new int[n * n * n];
  for (size_t m = 4; m < n; m *= 2) {
    refine3d(g, f, m);
    std::swap(f, g);
  }
  delete[] g;

  // convert ints to real type
  convert_ints_to_reals(data, f, n * n * n);
  delete[] f;

  return true;
}

// generate 4D test array of size n^4
template <typename real>
inline bool
gen_array_4d(real* data, size_t n)
{
  // ensure n >= 4 is a power of two
  if (n < 4 || n & (n - 1))
    return false;

  // initialize 4x4x4x4 integer array
  int* f = new int[n * n * n * n];
  std::fill(f, f + 4 * 4 * 4 * 4, 0);
  for (uint w = 1; w < 3; w++)
    for (uint z = 1; z < 3; z++)
      for (uint y = 1; y < 3; y++)
        for (uint x = 1; x < 3; x++)
          f[x + 4 * (y + 4 * (z + 4 * w))] = 0x10000 * (1 - 2 * int((x ^ y ^ z ^ w) & 1u));

  // refine to n^4 array
  int* g = new int[n * n * n * n];
  for (size_t m = 4; m < n; m *= 2) {
    refine4d(g, f, m);
    std::swap(f, g);
  }
  delete[] g;

  // convert ints to real type
  convert_ints_to_reals(data, f, n * n * n * n);
  delete[] f;

  return true;
}

// initialize array
template <typename Scalar>
inline void
initialize(Scalar* p, uint dims, ArraySize array_size)
{
  size_t size = 1ul << ((array_size == Small ? 12 : 24) / dims);

  switch (dims) {
    default:
    case 1:
      gen_array_1d<Scalar>(p, size);
      break;
    case 2:
      gen_array_2d<Scalar>(p, size);
      break;
    case 3:
      gen_array_3d<Scalar>(p, size);
      break;
    case 4:
      gen_array_4d<Scalar>(p, size);
      break;
  }
}

// compute checksum
inline uint32
hash(const void* p, size_t n)
{
  uint32 h = 0;
  for (const uchar* q = static_cast<const uchar*>(p); n; q++, n--) {
    // Jenkins one-at-a-time hash; see http://www.burtleburtle.net/bob/hash/doobs.html
    h += *q;
    h += h << 10;
    h ^= h >>  6;
  }
  h += h <<  3;
  h ^= h >> 11;
  h += h << 15;
  return h;
}

// test fixed-rate mode
template <typename Scalar>
inline uint
test_rate(zfp_stream* stream, const zfp_field* input, double rate, Scalar tolerance, bool timings = false)
{
  uint failures = 0;
  size_t n = zfp_field_size(input, NULL);
  uint dims = zfp_field_dimensionality(input);
  zfp_type type = zfp_field_type(input);

  // allocate memory for compressed data
  rate = zfp_stream_set_rate(stream, rate, type, dims, 0);
  size_t bufsize = zfp_stream_maximum_size(stream, input);
  uchar* buffer = new uchar[bufsize];
  bitstream* s = stream_open(buffer, bufsize);
  zfp_stream_set_bit_stream(stream, s);

  // perform compression test
  std::ostringstream status;
  status << "  compress:  ";
  status << " rate=" << std::fixed << std::setprecision(0) << std::setw(2) << rate;
  clock_t c = clock();
  zfp_stream_rewind(stream);
  size_t outsize = zfp_compress(stream, input);
  double time = double(clock() - c) / CLOCKS_PER_SEC;
  double throughput = (n * sizeof(Scalar)) / (0x100000 * time);
  if (timings)
    status << " throughput=" << std::setprecision(1) << std::setw(6) << throughput << " MB/s";
  bool pass = true;
  // make sure compressed size matches rate
  size_t bytes = (size_t)floor(rate * zfp_field_size(input, NULL) / CHAR_BIT + 0.5);
  if (outsize != bytes) {
    status << " [" << outsize << " != " << bytes << "]";
    pass = false;
  }
  std::cout << std::setw(width) << std::left << status.str() << (pass ? " OK " : "FAIL") << std::endl;
  if (!pass)
    failures++;

  // perform decompression test
  status.str("");
  status << "  decompress:";
  status << " rate=" << std::fixed << std::setprecision(0) << std::setw(2) << rate;
  Scalar* g = new Scalar[n];
  zfp_field* output = zfp_field_alloc();
  *output = *input;
  zfp_field_set_pointer(output, g);
  c = clock();
  zfp_stream_rewind(stream);
  pass = !!zfp_decompress(stream, output);
  if (!pass)
    status << " [decompression failed]";
  else {
    double time = double(clock() - c) / CLOCKS_PER_SEC;
    double throughput = (n * sizeof(Scalar)) / (0x100000 * time);
    if (timings)
      status << " throughput=" << std::setprecision(1) << std::setw(6) << throughput << " MB/s";
    // compute max error
    Scalar* f = static_cast<Scalar*>(zfp_field_pointer(input));
    Scalar emax = 0;
    for (uint i = 0; i < n; i++)
      emax = std::max(emax, std::abs(f[i] - g[i]));
    status << std::scientific;
    status.precision(3);
    // make sure max error is within tolerance
    if (emax <= tolerance)
      status << " " << emax << " <= " << tolerance;
    else {
      status << " [" << emax << " > " << tolerance << "]";
      pass = false;
    }
  }
  zfp_field_free(output);
  delete[] g;
  stream_close(s);
  delete[] buffer;
  std::cout << std::setw(width) << std::left << status.str() << (pass ? " OK " : "FAIL") << std::endl;
  if (!pass)
    failures++;

  return failures;
}

// test fixed-precision mode
template <typename Scalar>
inline uint
test_precision(zfp_stream* stream, const zfp_field* input, uint precision, size_t bytes)
{
  uint failures = 0;
  size_t n = zfp_field_size(input, NULL);

  // allocate memory for compressed data
  zfp_stream_set_precision(stream, precision);
  size_t bufsize = zfp_stream_maximum_size(stream, input);
  uchar* buffer = new uchar[bufsize];
  bitstream* s = stream_open(buffer, bufsize);
  zfp_stream_set_bit_stream(stream, s);

  // perform compression test
  std::ostringstream status;
  status << "  compress:  ";
  status << " precision=" << std::setw(2) << precision;
  zfp_stream_rewind(stream);
  size_t outsize = zfp_compress(stream, input);
  double ratio = double(n * sizeof(Scalar)) / outsize;
  status << " ratio=" << std::fixed << std::setprecision(3) << std::setw(7) << ratio;
  bool pass = true;
  // make sure compressed size agrees
  if (outsize != bytes) {
    status << " [" << outsize << " != " << bytes << "]";
    pass = false;
  }
  std::cout << std::setw(width) << std::left << status.str() << (pass ? " OK " : "FAIL") << std::endl;
  if (!pass)
    failures++;

  // perform decompression test
  status.str("");
  status << "  decompress:";
  status << " precision=" << std::setw(2) << precision;
  Scalar* g = new Scalar[n];
  zfp_field* output = zfp_field_alloc();
  *output = *input;
  zfp_field_set_pointer(output, g);
  zfp_stream_rewind(stream);
  pass = !!zfp_decompress(stream, output);
  if (!pass)
    status << " [decompression failed]";
  zfp_field_free(output);
  delete[] g;
  stream_close(s);
  delete[] buffer;
  std::cout << std::setw(width) << std::left << status.str() << (pass ? " OK " : "FAIL") << std::endl;
  if (!pass)
    failures++;

  return failures;
}

// test fixed-accuracy mode
template <typename Scalar>
inline uint
test_accuracy(zfp_stream* stream, const zfp_field* input, Scalar tolerance, size_t bytes)
{
  uint failures = 0;
  size_t n = zfp_field_size(input, NULL);

  // allocate memory for compressed data
  tolerance = static_cast<Scalar>(zfp_stream_set_accuracy(stream, tolerance));
  size_t bufsize = zfp_stream_maximum_size(stream, input);
  uchar* buffer = new uchar[bufsize];
  bitstream* s = stream_open(buffer, bufsize);
  zfp_stream_set_bit_stream(stream, s);

  // perform compression test
  std::ostringstream status;
  status << "  compress:  ";
  status << " tolerance=" << std::scientific << std::setprecision(3) << tolerance;
  zfp_stream_rewind(stream);
  size_t outsize = zfp_compress(stream, input);
  double ratio = double(n * sizeof(Scalar)) / outsize;
  status << " ratio=" << std::fixed << std::setprecision(3) << std::setw(7) << ratio;
  bool pass = true;
  // make sure compressed size agrees
  if (outsize != bytes) {
    status << " [" << outsize << " != " << bytes << "]";
    pass = false;
  }
  std::cout << std::setw(width) << std::left << status.str() << (pass ? " OK " : "FAIL") << std::endl;
  if (!pass)
    failures++;

  // perform decompression test
  status.str("");
  status << "  decompress:";
  status << " tolerance=" << std::scientific << std::setprecision(3) << tolerance;
  Scalar* g = new Scalar[n];
  zfp_field* output = zfp_field_alloc();
  *output = *input;
  zfp_field_set_pointer(output, g);
  zfp_stream_rewind(stream);
  pass = !!zfp_decompress(stream, output);
  if (!pass)
    status << " [decompression failed]";
  else {
    // compute max error
    Scalar* f = static_cast<Scalar*>(zfp_field_pointer(input));
    Scalar emax = 0;
    for (uint i = 0; i < n; i++)
      emax = std::max(emax, std::abs(f[i] - g[i]));
    status << std::scientific << std::setprecision(3) << " ";
    // make sure max error is within tolerance
    if (emax <= tolerance)
      status << emax << " <= " << tolerance;
    else if (tolerance == 0)
      status << "(" << emax << " > 0)";
    else {
      status << "[" << emax << " > " << tolerance << "]";
      pass = false;
    }
  }
  zfp_field_free(output);
  delete[] g;
  stream_close(s);
  delete[] buffer;
  std::cout << std::setw(width) << std::left << status.str() << (pass ? " OK " : "FAIL") << std::endl;
  if (!pass)
    failures++;

  return failures;
}

// test reversible mode
template <typename Scalar>
inline uint
test_reversible(zfp_stream* stream, const zfp_field* input, size_t bytes)
{
  uint failures = 0;
  size_t n = zfp_field_size(input, NULL);

  // allocate memory for compressed data
  zfp_stream_set_reversible(stream);
  size_t bufsize = zfp_stream_maximum_size(stream, input);
  uchar* buffer = new uchar[bufsize];
  bitstream* s = stream_open(buffer, bufsize);
  zfp_stream_set_bit_stream(stream, s);

  // perform compression test
  std::ostringstream status;
  status << "  compress:  ";
  status << " reversible";
  zfp_stream_rewind(stream);
  size_t outsize = zfp_compress(stream, input);
  double ratio = double(n * sizeof(Scalar)) / outsize;
  status << " ratio=" << std::fixed << std::setprecision(3) << std::setw(7) << ratio;
  bool pass = true;
  // make sure compressed size agrees
  if (outsize != bytes) {
    status << " [" << outsize << " != " << bytes << "]";
    pass = false;
  }
  std::cout << std::setw(width) << std::left << status.str() << (pass ? " OK " : "FAIL") << std::endl;
  if (!pass)
    failures++;

  // perform decompression test
  status.str("");
  status << "  decompress:";
  status << " reversible";
  Scalar* g = new Scalar[n];
  zfp_field* output = zfp_field_alloc();
  *output = *input;
  zfp_field_set_pointer(output, g);
  zfp_stream_rewind(stream);
  pass = !!zfp_decompress(stream, output);
  if (!pass)
    status << " [decompression failed]";
  else {
    // make sure reconstruction is bit-for-bit exact
    pass = !memcmp(zfp_field_pointer(input), zfp_field_pointer(output), n * sizeof(Scalar));
    if (!pass)
      status << " [reconstruction differs]";
  }
  zfp_field_free(output);
  delete[] g;
  stream_close(s);
  delete[] buffer;
  std::cout << std::setw(width) << std::left << status.str() << (pass ? " OK " : "FAIL") << std::endl;
  if (!pass)
    failures++;

  return failures;
}

// perform 1D differencing
template <typename Scalar>
inline void
update_array1(zfp::array1<Scalar>& a)
{
  for (uint i = 0; i < a.size() - 1; i++)
    a(i) -= a(i + 1);
  for (uint i = 0; i < a.size() - 1; i++)
    a(0) = std::max(a(0), a(i));
}

// perform 2D differencing
template <typename Scalar>
inline void
update_array2(zfp::array2<Scalar>& a)
{
  for (uint j = 0; j < a.size_y(); j++)
    for (uint i = 0; i < a.size_x() - 1; i++)
      a(i, j) -= a(i + 1, j);
  for (uint j = 0; j < a.size_y() - 1; j++)
    for (uint i = 0; i < a.size_x(); i++)
      a(i, j) -= a(i, j + 1);
  for (uint j = 0; j < a.size_y() - 1; j++)
    for (uint i = 0; i < a.size_x() - 1; i++)
      a(0, 0) = std::max(a(0, 0), a(i, j));
}

// perform 3D differencing
template <typename Scalar>
inline void
update_array3(zfp::array3<Scalar>& a)
{
  for (uint k = 0; k < a.size_z(); k++)
    for (uint j = 0; j < a.size_y(); j++)
      for (uint i = 0; i < a.size_x() - 1; i++)
        a(i, j, k) -= a(i + 1, j, k);
  for (uint k = 0; k < a.size_z(); k++)
    for (uint j = 0; j < a.size_y() - 1; j++)
      for (uint i = 0; i < a.size_x(); i++)
        a(i, j, k) -= a(i, j + 1, k);
  for (uint k = 0; k < a.size_z() - 1; k++)
    for (uint j = 0; j < a.size_y(); j++)
      for (uint i = 0; i < a.size_x(); i++)
        a(i, j, k) -= a(i, j, k + 1);
  for (uint k = 0; k < a.size_z() - 1; k++)
    for (uint j = 0; j < a.size_y() - 1; j++)
      for (uint i = 0; i < a.size_x() - 1; i++)
        a(0, 0, 0) = std::max(a(0, 0, 0), a(i, j, k));
}

template <class Array>
inline void update_array(Array& a);

template <>
inline void
update_array(zfp::array1<float>& a) { update_array1(a); }

template <>
inline void
update_array(zfp::array1<double>& a) { update_array1(a); }

template <>
inline void
update_array(zfp::array2<float>& a) { update_array2(a); }

template <>
inline void
update_array(zfp::array2<double>& a) { update_array2(a); }

template <>
inline void
update_array(zfp::array3<float>& a) { update_array3(a); }

template <>
inline void
update_array(zfp::array3<double>& a) { update_array3(a); }

// test random-accessible array primitive
template <class Array, typename Scalar>
inline uint
test_array(Array& a, const Scalar* f, uint n, double tolerance, double dfmax)
{
  uint failures = 0;

  // test construction
  std::ostringstream status;
  status << "  construct: ";
  Scalar emax = 0;
  for (uint i = 0; i < n; i++)
    emax = std::max(emax, std::abs(f[i] - a[i]));
  status << std::scientific;
  status.precision(3);
  // make sure max error is within tolerance
  bool pass = true;
  if (emax <= tolerance)
    status << " " << emax << " <= " << tolerance;
  else {
    status << " [" << emax << " > " << tolerance << "]";
    pass = false;
  }

  std::cout << std::setw(width) << std::left << status.str() << (pass ? " OK " : "FAIL") << std::endl;
  if (!pass)
    failures++;

  // test array updates
  status.str("");
  status << "  update:    ";
  update_array(a);
  Scalar amax = a[0];
  pass = true;
  if (std::abs(amax - dfmax) <= 1e-3 * dfmax)
    status << " " << amax << " ~ " << dfmax;
  else {
    status << " [" << amax << " != " << dfmax << "]";
    pass = false;
  }

  std::cout << std::setw(width) << std::left << status.str() << (pass ? " OK " : "FAIL") << std::endl;
  if (!pass)
    failures++;

  return failures;
}

// test small or large d-dimensional arrays of type Scalar
template <typename Scalar>
inline uint
test(uint dims, ArraySize array_size)
{
  uint failures = 0;
  uint m = test_size(array_size);
  uint n = m * m * m * m * m * m * m * m * m * m * m * m;
  Scalar* f = new Scalar[n];

  // determine array size
  uint nx, ny, nz ,nw;
  zfp_field* field = zfp_field_alloc();
  zfp_field_set_type(field, zfp::codec<Scalar>::type);
  zfp_field_set_pointer(field, f);
  switch (dims) {
    case 1:
      nx = n;
      ny = nz = nw = 0;
      zfp_field_set_size_1d(field, nx);
      break;
    case 2:
      nx = ny = m * m * m * m * m * m;
      nz = nw = 0;
      zfp_field_set_size_2d(field, nx, ny);
      break;
    case 3:
      nx = ny = nz = m * m * m * m;
      nw = 0;
      zfp_field_set_size_3d(field, nx, ny, nz);
      break;
    case 4:
      nx = ny = nz = nw = m * m * m;
      zfp_field_set_size_4d(field, nx, ny, nz, nw);
      break;
    default:
      std::cout << "invalid dimensions " << dims << std::endl;
      return 1;
  }
  initialize<Scalar>(f, dims, array_size);
  uint t = (zfp_field_type(field) == zfp_type_float ? 0 : 1);
  std::cout << "testing " << dims << "D array of " << (t == 0 ? "floats" : "doubles") << std::endl;

  // test data integrity
  uint32 checksum[2][2][4] = {
    // small
    {{ 0x54174c44u, 0x86609589u, 0xfc0a6a76u, 0xa3481e00u },
     { 0x7d257bb6u, 0x294bb210u, 0x68614d26u, 0xf6bd3a21u }},
    // large
    {{ 0xd1ce1aceu, 0x644274dau, 0xc0ad63fau, 0x700de480u },
     { 0xc3ed7116u, 0x644e2117u, 0xd7464b07u, 0x2516382eu }},
  };
  uint32 h = hash(f, n * sizeof(Scalar));
  if (h != checksum[array_size][t][dims - 1])
    std::cout << "warning: test data checksum " << std::hex << h << " != " << checksum[array_size][t][dims - 1] << "; tests below may fail" << std::endl;

  // open compressed stream
  zfp_stream* stream = zfp_stream_open(0);

  // test fixed rate
  for (uint rate = 2u >> t, i = 0; rate <= 32 * (t + 1); rate *= 4, i++) {
    // expected max errors
    double emax[2][2][4][4] = {
      // small
      {
        {
          {1.627e+01, 8.277e-02, 0.000e+00},
          {1.500e+00, 3.663e-03, 0.000e+00},
          {1.500e+00, 9.583e-03, 0.000e+00},
          {1.373e+01, 6.633e-01, 0.000e+00},
        },
        {
          {1.627e+01, 1.601e+01, 1.832e-04, 0.000e+00},
          {2.376e+01, 1.797e-01, 8.584e-06, 0.000e+00},
          {5.210e+00, 2.002e-01, 3.338e-05, 0.000e+00},
          {1.016e+01, 8.985e+00, 3.312e-03, 0.000e+00},
        },
      },
      // large
      {
        {
          {1.627e+01, 2.100e-02, 0.000e+00},
          {1.624e-01, 7.439e-05, 0.000e+00},
          {1.001e-02, 7.248e-05, 0.000e+00},
          {2.527e-02, 2.460e-04, 0.000e+00},
        },
        {
          {1.627e+01, 1.601e+01, 2.289e-05, 0.000e+00},
          {1.607e+01, 2.076e-03, 0.000e+00, 0.000e+00},
          {1.407e-01, 7.344e-04, 0.000e+00, 0.000e+00},
          {1.436e-01, 2.659e-03, 8.801e-08, 0.000e+00},
        }
      }
    };
    failures += test_rate<Scalar>(stream, field, rate, static_cast<Scalar>(emax[array_size][t][dims - 1][i]), array_size == Large);
  }

  if (stream_word_bits != 64)
    std::cout << "warning: stream word size is smaller than 64; tests below may fail" << std::endl;

  // test fixed precision
  for (uint prec = 4u << t, i = 0; i < 3; prec *= 2, i++) {
    // expected compressed sizes
    size_t bytes[2][2][4][3] = {
      // small
      {
        {
          {2192, 3280, 6328},
          { 592, 1328, 4384},
          { 152, 1040, 4600},
          {  64, 1760, 5856},
        },
        {
          {3664, 6712, 14104},
          {1424, 4480, 12616},
          {1064, 4624, 12808},
          {1768, 5864, 14056},
        },
      },
      // large
      {
        {
          {8965672, 13160560, 21835352},
          {2235560,  3512848, 10309240},
          { 568456,  1361056,  8759696},
          { 134344,   739632,  8896360},
        },
        {
          {14733112, 23407904, 44997832},
          { 3905240, 10701640, 40856544},
          { 1458368,  8857008, 41270184},
          {  763928,  8920656, 41574712},
        },
      }
    };
    failures += test_precision<Scalar>(stream, field, prec, bytes[array_size][t][dims - 1][i]);
  }

  // test fixed accuracy
  for (uint i = 0; i < 3; i++) {
    Scalar tol[] = { Scalar(1e-3), 2 * std::numeric_limits<Scalar>::epsilon(), 0 };
    // expected compressed sizes
    size_t bytes[2][2][4][3] = {
      // small
      {
        {
          {6328, 11944, 13720},
          {4936, 11064, 12520},
          {6104, 11752, 12784},
          {9440, 14048, 14048},
        },
        {
          {6712, 25888, 29064},
          {5032, 26016, 28984},
          {6128, 27120, 29192},
          {9448, 30440, 30440},
        },
      },
      // large
      {
        {
          {21815976, 38285256, 43425280},
          { 9187232, 32695984, 40464144},
          { 8914336, 33364208, 41172864},
          {12109200, 35921784, 41550416},
        },
        {
          {23388528, 79426016,  88659304},
          { 9579632, 89770896, 103388072},
          { 9011648, 94009072, 107606336},
          {12133496, 97126288, 107911568},
        },
      }
    };
    failures += test_accuracy<Scalar>(stream, field, tol[i], bytes[array_size][t][dims - 1][i]);
  }

  // test reversible
  {
    // expected compressed sizes
    size_t bytes[2][2][4] = {
      // small
      {
        {
          7272,
          5104,
          6096,
          6864,
        },
        {
          7784,
          5232,
          6128,
          6872,
        },
      },
      // large
      {
        {
          25037288,
          12792440,
          14187128,
          17135704,
        },
        {
          27134024,
          13315632,
          14316880,
          17168096,
        },
      }
    };
    failures += test_reversible<Scalar>(stream, field, bytes[array_size][t][dims - 1]);
  }

  // test compressed array support
  double emax[2][2][3] = {
    // small
    {
      {4.578e-05, 7.630e-06, 3.148e-05},
      {1.832e-04, 8.584e-06, 3.338e-05},
    },
    // large
    {
      {0.000e+00, 0.000e+00, 0.000e+00},
      {2.289e-05, 0.000e+00, 0.000e+00},
    }
  };
  double dfmax[2][2][3] = {
    // small
    {
      {2.155e-02, 3.755e-01, 1.846e+00},
      {2.155e-02, 3.755e-01, 1.846e+00},
    },
    // large
    {
      {2.441e-04, 4.883e-04, 1.221e-03},
      {2.670e-04, 4.883e-04, 1.221e-03},
    }
  };
  double rate = 16;
  switch (dims) {
    case 1: {
        zfp::array1<Scalar> a(nx, rate, f);
        failures += test_array(a, f, n, static_cast<Scalar>(emax[array_size][t][dims - 1]), static_cast<Scalar>(dfmax[array_size][t][dims - 1]));
      }
      break;
    case 2: {
        zfp::array2<Scalar> a(nx, ny, rate, f);
        failures += test_array(a, f, n, static_cast<Scalar>(emax[array_size][t][dims - 1]), static_cast<Scalar>(dfmax[array_size][t][dims - 1]));
      }
      break;
    case 3: {
        zfp::array3<Scalar> a(nx, ny, nz, rate, f);
        failures += test_array(a, f, n, static_cast<Scalar>(emax[array_size][t][dims - 1]), static_cast<Scalar>(dfmax[array_size][t][dims - 1]));
      }
      break;
    case 4: // 4D arrays not yet supported
      break;
  }

  std::cout << std::endl;
  zfp_stream_close(stream);
  zfp_field_free(field);

  delete[] f;
  return failures;
}

// various library and compiler sanity checks
inline uint
common_tests()
{
  uint failures = 0;
  // test library version
  if (zfp_codec_version != ZFP_CODEC || zfp_library_version != ZFP_VERSION) {
    std::cout << "library header and binary version mismatch" << std::endl;
    failures++;
  }
  // ensure integer type sizes are correct
  if (CHAR_BIT != 8) {
    std::cout << "byte type is not 8 bits wide" << std::endl;
    failures++;
  }
  if (sizeof(int8) != 1u || sizeof(uint8) != 1u) {
    std::cout << "8-bit integer type is not one byte wide" << std::endl;
    failures++;
  }
  if (sizeof(int16) != 2u || sizeof(uint16) != 2u) {
    std::cout << "16-bit integer type is not two bytes wide" << std::endl;
    failures++;
  }
  if (sizeof(int32) != 4u || sizeof(uint32) != 4u) {
    std::cout << "32-bit integer type is not four bytes wide" << std::endl;
    failures++;
  }
  if (sizeof(int64) != 8u || sizeof(uint64) != 8u) {
    std::cout << "64-bit integer type is not eight bytes wide" << std::endl;
    failures++;
  }
  // ensure signed right shifts are arithmetic
  int32 x32 = -2;
  if ((x32 >> 1) != -1 || (x32 >> 2) != -1) {
    std::cout << "32-bit arithmetic right shift not supported" << std::endl;
    failures++;
  }
  int64 x64 = -2;
  if ((x64 >> 1) != INT64C(-1) || (x64 >> 2) != INT64C(-1)) {
    std::cout << "64-bit arithmetic right shift not supported" << std::endl;
    failures++;
  }
  // testing requires default (64-bit) stream words
  if (stream_word_bits != 64) {
    std::cout << "regression testing requires BIT_STREAM_WORD_TYPE=uint64" << std::endl;
    failures++;
  }
  return failures;
}

int main(int argc, char* argv[])
{
  std::cout << zfp_version_string << std::endl;
  std::cout << "library version " << zfp_library_version << std::endl;
  std::cout << "CODEC version " << zfp_codec_version << std::endl;
  std::cout << "data model ";
  size_t model = ((sizeof(uint64) - 1) << 12) +
                 ((sizeof(void*) - 1) << 8) +
                 ((sizeof(unsigned long int) - 1) << 4) +
                 ((sizeof(unsigned int) - 1) << 0);
  switch (model) {
    case 0x7331u:
      std::cout << "LP32";
      break;
    case 0x7333u:
      std::cout << "ILP32";
      break;
    case 0x7733u:
      std::cout << "LLP64";
      break;
    case 0x7773u:
      std::cout << "LP64";
      break;
    case 0x7777u:
      std::cout << "ILP64";
      break;
    default:
      std::cout << "unknown (0x" << std::hex << model << ")";
      break;
  }
  std::cout << std::endl;
  std::cout << std::endl;

  uint sizes = 0;
  uint types = 0;
  uint dims = 0;

  for (int i = 1; i < argc; i++)
    if (std::string(argv[i]) == "small")
      sizes |= mask(Small);
    else if (std::string(argv[i]) == "large")
      sizes |= mask(Large);
    else if (std::string(argv[i]) == "float" || std::string(argv[i]) == "fp32")
      types |= mask(Float);
    else if (std::string(argv[i]) == "double" || std::string(argv[i]) == "fp64")
      types |= mask(Double);
    else if (std::string(argv[i]) == "1d")
      dims |= mask(1);
    else if (std::string(argv[i]) == "2d")
      dims |= mask(2);
    else if (std::string(argv[i]) == "3d")
      dims |= mask(3);
    else if (std::string(argv[i]) == "4d")
      dims |= mask(4);
    else if (std::string(argv[i]) == "all") {
      sizes |= mask(Small) | mask(Large);
      types |= mask(Float) | mask(Double);
      dims |= mask(1) | mask(2) | mask(3) | mask(4);
    }
    else {
      std::cerr << "Usage: testzfp [all] [small|large] [fp32|fp64|float|double] [1d|2d|3d|4d]" << std::endl;
      return EXIT_FAILURE;
    }

  // use defaults if not specified
  if (!sizes)
    sizes = mask(Small);
  if (!types)
    types = mask(Float) | mask(Double);
  if (!dims)
    dims = mask(1) | mask(2) | mask(3) | mask(4);

  // test library and compiler
  uint failures = common_tests();
  if (failures)
    return EXIT_FAILURE;

  // test arrays
  for (int size = Small; size <= Large; size++)
    if (sizes & mask(ArraySize(size))) {
      for (uint d = 1; d <= 4; d++)
        if (dims & mask(d)) {
          if (types & mask(Float))
            failures += test<float>(d, ArraySize(size));
          if (types & mask(Double))
            failures += test<double>(d, ArraySize(size));
       }
    }

  if (failures)
    std::cout << failures << " test(s) failed" << std::endl;
  else
    std::cout << "all tests passed" << std::endl;

  return failures ? EXIT_FAILURE : EXIT_SUCCESS;
}
