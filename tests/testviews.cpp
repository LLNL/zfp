#include <cmath>
#include <cstdlib>
#include <iostream>
#include <sstream>
#include "zfp/array2.hpp"
#include "zfp/array3.hpp"
#ifdef _OPENMP
#include <omp.h>
#endif

#define EPSILON 1e-3

// random integer in {begin, ..., end}
static size_t
rand(size_t begin, size_t end)
{
  return begin + size_t(rand()) % (end - begin + 1);
}

// ensure f and g are sufficiently close
static bool
verify(double f, double g)
{
  if (std::fabs(f - g) > EPSILON) {
#ifdef _OPENMP
    #pragma omp critical
#endif
    std::cout << " [FAIL]" << std::endl;
    std::cerr << "  error: " << f << " != " << g << std::endl;
    return false;
  }
  return true;
}

static void
test(const char* name, bool verbose)
{
  if (verbose)
    std::cout << std::endl;
  std::cout << name;
}

static void
tally(size_t& pass, size_t& fail, bool success)
{
  if (success) {
    std::cout << " [OK]" << std::endl;
    pass++;
  }
  else
    fail++;
}

static int
usage()
{
  std::cerr << "Usage: testviews [nx ny nz [x0 y0 z0 mx my mz]]" << std::endl;
  return EXIT_FAILURE;
}

int main(int argc, char* argv[])
{
  size_t nx = 8;
  size_t ny = 48;
  size_t nz = 32;
  size_t x0, y0, z0;
  size_t mx, my, mz;
  double rate = 16;
  bool verbose = false;
  size_t pass = 0;
  size_t fail = 0;

  // parse command-line arguments
  switch (argc) {
    case 10:
      if ((std::istringstream(argv[4]) >> x0).fail() ||
          (std::istringstream(argv[5]) >> y0).fail() ||
          (std::istringstream(argv[6]) >> z0).fail() ||
          (std::istringstream(argv[7]) >> mx).fail() || !mx ||
          (std::istringstream(argv[8]) >> my).fail() || !my ||
          (std::istringstream(argv[9]) >> mz).fail() || !mz)
        return usage();
      // FALLTHROUGH
    case 4:
      if ((std::istringstream(argv[1]) >> nx).fail() || !nx ||
          (std::istringstream(argv[2]) >> ny).fail() || !ny ||
          (std::istringstream(argv[3]) >> nz).fail() || !nz)
        return usage();
      // FALLTHROUGH
    case 1:
      break;
    default:
      return usage();
  }

  if (argc < 10) {
    // generate random view
    x0 = rand(0, nx - 1);
    y0 = rand(0, ny - 1);
    z0 = rand(0, nz - 1);
    mx = rand(1, nx - x0);
    my = rand(1, ny - y0);
    mz = rand(1, nz - z0);
  }

  // validate arguments
  if (x0 + mx > nx || y0 + my > ny || z0 + mz > nz) {
    std::cerr << "invalid view parameters" << std::endl;
    return EXIT_FAILURE;
  }

  std::cout << "a(" << nx << ", " << ny << ", " << nz << ")" << std::endl;
  std::cout << "v(" << mx << ", " << my << ", " << mz << ") + (" << x0 << ", " << y0 << ", " << z0 << ")" << std::endl;

  // initialize 3D array to linear function
  zfp::array3<double> a(nx, ny, nz, rate);
  for (size_t z = 0; z < nz; z++)
    for (size_t y = 0; y < ny; y++)
      for (size_t x = 0; x < nx; x++)
        a(x, y, z) = static_cast<double>(x + nx * (y + ny * z));

  bool success;

  // rectangular view into a
  test("3D view", verbose);
  zfp::array3<double>::view v(&a, x0, y0, z0, mx, my, mz);
  success = true;
  for (size_t z = 0; z < v.size_z() && success; z++)
    for (size_t y = 0; y < v.size_y() && success; y++)
      for (size_t x = 0; x < v.size_x() && success; x++) {
        if (verbose)
          std::cout << x << " " << y << " " << z << ": " << a(x0 + x, y0 + y, z0 + z) << " " << v(x, y, z) << std::endl;
        success = verify(a(x0 + x, y0 + y, z0 + z), v(x, y, z));
      }
  tally(pass, fail, success);

  // flat view of all of a
  test("3D flat view", verbose);
  zfp::array3<double>::flat_view fv(&a);
  success = true;
  for (size_t z = 0; z < fv.size_z() && success; z++)
    for (size_t y = 0; y < fv.size_y() && success; y++)
      for (size_t x = 0; x < fv.size_x() && success; x++) {
        if (verbose)
          std::cout << x << " " << y << " " << z << ": " << a(x, y, z) << " " << fv[fv.index(x, y, z)] << std::endl;
        success = verify(a(x, y, z), fv[fv.index(x, y, z)]);
      }
  tally(pass, fail, success);

  // nested view of all of a
  test("3D nested view", verbose);
  zfp::array3<double>::nested_view nv(&a);
  success = true;
  for (size_t z = 0; z < nv.size_z() && success; z++)
    for (size_t y = 0; y < nv.size_y() && success; y++)
      for (size_t x = 0; x < nv.size_x() && success; x++) {
        if (verbose)
          std::cout << x << " " << y << " " << z << ": " << a(x, y, z) << " " << nv[z][y][x] << std::endl;
        success = verify(a(x, y, z), nv[z][y][x]);
      }
  tally(pass, fail, success);

  // pointers and iterators into a via view v
  test("3D view pointers and iterators", verbose);
  zfp::array3<double>::view::const_reference vr = v(0, 0, 0);
  zfp::array3<double>::view::const_pointer p = &vr;
  p = &v(0, 0, 0);
  success = true;
  for (zfp::array3<double>::view::const_iterator it = v.begin(); it != v.end() && success; it++) {
    size_t x = it.i();
    size_t y = it.j();
    size_t z = it.k();
    if (verbose)
      std::cout << x << " " << y << " " << z << ": " << *it << " " << p[x + mx * (y + my * z)] << std::endl;
    success = verify(*it, p[x + mx * (y + my * z)]);
  }
  tally(pass, fail, success);

  // pointers and iterators into a via flat view fv
  test("3D flat view pointers and iterators", verbose);
  zfp::array3<double>::flat_view::const_reference fvr = fv[0];
  zfp::array3<double>::flat_view::const_pointer fp = &fvr;
  fp = &fv(0, 0, 0);
  success = true;
  for (zfp::array3<double>::flat_view::const_iterator it = fv.begin(); it != fv.end() && success; it++) {
    size_t x = it.i();
    size_t y = it.j();
    size_t z = it.k();
    if (verbose)
      std::cout << x << " " << y << " " << z << ": " << *it << " " << fp[x + nx * (y + ny * z)] << std::endl;
    success = verify(*it, fp[x + nx * (y + ny * z)]);
  }
  tally(pass, fail, success);

  // 2D slice of a
  test("2D slice", verbose);
  size_t z = rand(0, nv.size_z() - 1);
  zfp::array3<double>::nested_view2 slice2(nv[z]);
  success = true;
  for (size_t y = 0; y < slice2.size_y() && success; y++)
    for (size_t x = 0; x < slice2.size_x() && success; x++) {
      if (verbose)
        std::cout << x << " " << y << " " << z << ": " << a(x, y, z) << " " << slice2[y][x] << std::endl;
      success = verify(a(x, y, z), slice2[y][x]);
    }
  tally(pass, fail, success);

  // 2D array constructed from 2D slice (exercises deep copy via iterator)
  test("2D array from 2D slice", verbose);
  zfp::array2<double> b(slice2);
  success = true;
  for (size_t y = 0; y < b.size_y() && success; y++)
    for (size_t x = 0; x < b.size_x() && success; x++) {
      if (verbose)
        std::cout << x << " " << y << ": " << b(x, y) << " " << slice2[y][x] << std::endl;
      success = verify(b(x, y), slice2[y][x]);
    }
  tally(pass, fail, success);

  // 1D slice of a
  test("1D slice", verbose);
  size_t y = rand(0, slice2.size_y() - 1);
  zfp::array3<double>::nested_view1 slice1 = slice2[y];
  success = true;
  for (size_t x = 0; x < slice1.size_x() && success; x++) {
    if (verbose)
      std::cout << x << " " << y << " " << z << ": " << a(x, y, z) << " " << slice1[x] << std::endl;
    success = verify(a(x, y, z), slice1[x]);
  }
  tally(pass, fail, success);

  // 2D array constructed from 2D slice of 3D array (exercises deep copy via iterator)
  test("2D array from 2D slice of 3D array", verbose);
  zfp::array2<double> c(slice2);
  success = true;
  for (size_t y = 0; y < c.size_y() && success; y++)
    for (size_t x = 0; x < c.size_x() && success; x++) {
      if (verbose)
        std::cout << x << " " << y << ": " << c(x, y) << " " << slice2[y][x] << std::endl;
      success = verify(c(x, y), slice2[y][x]);
    }
  tally(pass, fail, success);

  // 2D thread-safe read-only view of c
  test("2D private read-only view", verbose);
  zfp::array2<double>::private_const_view d(&c);
  success = true;
  for (size_t y = 0; y < c.size_y() && success; y++)
    for (size_t x = 0; x < c.size_x() && success; x++) {
      if (verbose)
        std::cout << x << " " << y << ": " << c(x, y) << " " << d(x, y) << std::endl;
      success = verify(c(x, y), d(x, y));
    }
  tally(pass, fail, success);

#ifdef _OPENMP
  test("multithreaded 2D private read-only views", verbose);
  // copy c for verification; direct accesses to c are not thread-safe
  double* data = new double[c.size()];
  c.get(data);
  success = true;
  #pragma omp parallel reduction(&&:success)
  {
    // make a thread-local view into c
    zfp::array2<double>::private_const_view d(&c);
    for (size_t y = 0; y < d.size_y() && success; y++)
      for (size_t x = 0; x < d.size_x() && success; x++) {
        double val = data[x + nx * y];
        if (verbose && omp_get_thread_num() == 0)
          std::cout << x << " " << y << ": " << val << " " << d(x, y) << std::endl;
        success = verify(val, d(x, y));
      }
  }
  tally(pass, fail, success);

  test("multithreaded 2D private read-write views", verbose);
  success = true;
  #pragma omp parallel reduction(&&:success)
  {
    // partition c into disjoint views
    zfp::array2<double>::private_view d(&c);
    d.partition(omp_get_thread_num(), omp_get_num_threads());
    for (size_t j = 0; j < d.size_y(); j++)
      for (size_t i = 0; i < d.size_x(); i++) {
        d(i, j) += 1;
        size_t x = d.global_x(i);
        size_t y = d.global_y(j);
        double val = data[x + nx * y] + 1;
        if (verbose && omp_get_thread_num() == 0)
          std::cout << x << " " << y << ": " << val << " " << d(i, j) << std::endl;
        success = verify(val, d(i, j));
      }
  }
  tally(pass, fail, success);

  delete[] data;
#endif

  if (!fail) {
    std::cout << std::endl << "all " << pass << " tests passed" << std::endl;
    return EXIT_SUCCESS;
  }
  else {
    std::cout << std::endl << fail << " of " << pass + fail << " tests failed" << std::endl;
    return EXIT_FAILURE;
  }
}
