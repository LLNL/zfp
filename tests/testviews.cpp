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
static void
verify(double f, double g)
{
  if (std::fabs(f - g) > EPSILON) {
#ifdef _OPENMP
    #pragma omp critical
#endif
    std::cerr << "error: " << f << " != " << g << std::endl;
    exit(EXIT_FAILURE);
  }
}

// filter output; returns true for first head and last tail calls
static bool
filter_output(size_t head = 0, size_t tail = 0, size_t size = 0)
{
  static size_t i = 0;
  static size_t h = 0;
  static size_t t = 0;
  static size_t n = 0;

  if (size) {
    i = 0;
    h = head;
    t = tail;
    n = size;
    return false;
  }

  bool display = !(h <= i && i + t < n);
  if (!display && i == h)
    std::cout << "..." << std::endl;
  i++;

  return display;
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
      fallthrough_
    case 4:
      if ((std::istringstream(argv[1]) >> nx).fail() || !nx ||
          (std::istringstream(argv[2]) >> ny).fail() || !ny ||
          (std::istringstream(argv[3]) >> nz).fail() || !nz)
        return usage();
      fallthrough_
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

  // rectangular view into a
  std::cout << std::endl << "3D view" << std::endl;
  zfp::array3<double>::view v(&a, x0, y0, z0, mx, my, mz);
  filter_output(v.size_x() + 2, 3, v.size());
  for (size_t z = 0; z < v.size_z(); z++)
    for (size_t y = 0; y < v.size_y(); y++)
      for (size_t x = 0; x < v.size_x(); x++) {
        if (filter_output())
          std::cout << x << " " << y << " " << z << ": " << a(x0 + x, y0 + y, z0 + z) << " " << v(x, y, z) << std::endl;
        verify(a(x0 + x, y0 + y, z0 + z), v(x, y, z));
      }

  // flat view of all of a
  std::cout << std::endl << "3D flat view" << std::endl;
  zfp::array3<double>::flat_view fv(&a);
  filter_output(fv.size_x() + 2, 3, fv.size());
  for (size_t z = 0; z < fv.size_z(); z++)
    for (size_t y = 0; y < fv.size_y(); y++)
      for (size_t x = 0; x < fv.size_x(); x++) {
        if (filter_output())
          std::cout << x << " " << y << " " << z << ": " << a(x, y, z) << " " << fv[fv.index(x, y, z)] << std::endl;
        verify(a(x, y, z), fv[fv.index(x, y, z)]);
      }

  // nested view of all of a
  std::cout << std::endl << "3D nested view" << std::endl;
  zfp::array3<double>::nested_view nv(&a);
  filter_output(nv.size_x() + 2, 3, nv.size());
  for (size_t z = 0; z < nv.size_z(); z++)
    for (size_t y = 0; y < nv.size_y(); y++)
      for (size_t x = 0; x < nv.size_x(); x++) {
        if (filter_output())
          std::cout << x << " " << y << " " << z << ": " << a(x, y, z) << " " << nv[z][y][x] << std::endl;
        verify(a(x, y, z), nv[z][y][x]);
      }

  // pointers and iterators into a via view v
  std::cout << std::endl << "3D view pointers and iterators" << std::endl;
  zfp::array3<double>::view::const_reference vr = v(0, 0, 0);
  zfp::array3<double>::view::const_pointer p = &vr;
  p = &v(0, 0, 0);
  filter_output(v.size_x() + 2, 3, v.size());
  for (zfp::array3<double>::view::const_iterator it = v.begin(); it != v.end(); it++) {
    size_t x = it.i();
    size_t y = it.j();
    size_t z = it.k();
    if (filter_output())
      std::cout << x << " " << y << " " << z << ": " << *it << " " << p[x + mx * (y + my * z)] << std::endl;
    verify(*it, p[x + mx * (y + my * z)]);
  }

  // pointers and iterators into a via flat view fv
  std::cout << std::endl << "3D flat view pointers and iterators" << std::endl;
  zfp::array3<double>::flat_view::const_reference fvr = fv[0];
  zfp::array3<double>::flat_view::const_pointer fp = &fvr;
  fp = &fv(0, 0, 0);
  filter_output(fv.size_x() + 2, 3, fv.size());
  for (zfp::array3<double>::flat_view::const_iterator it = fv.begin(); it != fv.end(); it++) {
    size_t x = it.i();
    size_t y = it.j();
    size_t z = it.k();
    if (filter_output())
      std::cout << x << " " << y << " " << z << ": " << *it << " " << fp[x + nx * (y + ny * z)] << std::endl;
    verify(*it, fp[x + nx * (y + ny * z)]);
  }

  // 2D slice of a
  std::cout << std::endl << "2D slice" << std::endl;
  size_t z = rand(0, nv.size_z() - 1);
  zfp::array3<double>::nested_view2 slice2(nv[z]);
  filter_output(slice2.size_x() + 2, 3, slice2.size());
  for (size_t y = 0; y < slice2.size_y(); y++)
    for (size_t x = 0; x < slice2.size_x(); x++) {
      if (filter_output())
        std::cout << x << " " << y << " " << z << ": " << a(x, y, z) << " " << slice2[y][x] << std::endl;
      verify(a(x, y, z), slice2[y][x]);
    }

  // 2D array constructed from 2D slice (exercises deep copy via iterator)
  std::cout << std::endl << "2D array from 2D slice" << std::endl;
  zfp::array2<double> b(slice2);
  filter_output(b.size_x() + 2, 3, b.size());
  for (size_t y = 0; y < b.size_y(); y++)
    for (size_t x = 0; x < b.size_x(); x++) {
      if (filter_output())
        std::cout << x << " " << y << ": " << b(x, y) << " " << slice2[y][x] << std::endl;
      verify(b(x, y), slice2[y][x]);
    }

  // 1D slice of a
  std::cout << std::endl << "1D slice" << std::endl;
  size_t y = rand(0, slice2.size_y() - 1);
  zfp::array3<double>::nested_view1 slice1 = slice2[y];
  for (size_t x = 0; x < slice1.size_x(); x++) {
    std::cout << x << " " << y << " " << z << ": " << a(x, y, z) << " " << slice1[x] << std::endl;
    verify(a(x, y, z), slice1[x]);
  }

  // 2D array constructed from 2D slice of 3D array (exercises deep copy via iterator)
  std::cout << std::endl << "2D array from 2D slice of 3D array" << std::endl;
  zfp::array2<double> c(slice2);
  filter_output(c.size_x() + 2, 3, c.size());
  for (size_t y = 0; y < c.size_y(); y++)
    for (size_t x = 0; x < c.size_x(); x++) {
      if (filter_output())
        std::cout << x << " " << y << ": " << c(x, y) << " " << slice2[y][x] << std::endl;
      verify(c(x, y), slice2[y][x]);
    }

  // 2D thread-safe read-only view of c
  std::cout << std::endl << "2D private read-only view" << std::endl;
  zfp::array2<double>::private_const_view d(&c);
  filter_output(c.size_x() + 2, 3, c.size());
  for (size_t y = 0; y < c.size_y(); y++)
    for (size_t x = 0; x < c.size_x(); x++) {
      if (filter_output())
        std::cout << x << " " << y << ": " << c(x, y) << " " << d(x, y) << std::endl;
      verify(c(x, y), d(x, y));
    }

#ifdef _OPENMP
  std::cout << std::endl << "multithreaded 2D private read-only views" << std::endl;
  // copy c for verification; direct accesses to c are not thread-safe
  double* data = new double[c.size()];
  c.get(data);
  #pragma omp parallel
  {
    // make a thread-local view into c
    zfp::array2<double>::private_const_view d(&c);
    if (omp_get_thread_num() == 0)
      filter_output(d.size_x() + 2, 3, d.size());
    for (size_t y = 0; y < d.size_y(); y++)
      for (size_t x = 0; x < d.size_x(); x++) {
        double val = data[x + nx * y];
        if (omp_get_thread_num() == 0 && filter_output())
          std::cout << x << " " << y << ": " << val << " " << d(x, y) << std::endl;
        verify(val, d(x, y));
      }
  }

  std::cout << std::endl << "multithreaded 2D private read-write views" << std::endl;
  #pragma omp parallel
  {
    // partition c into disjoint views
    zfp::array2<double>::private_view d(&c);
    d.partition(omp_get_thread_num(), omp_get_num_threads());
    if (omp_get_thread_num() == 0)
      filter_output(d.size_x() + 2, 3, d.size());
    for (size_t j = 0; j < d.size_y(); j++)
      for (size_t i = 0; i < d.size_x(); i++) {
        d(i, j) += 1;
        size_t x = d.global_x(i);
        size_t y = d.global_y(j);
        double val = data[x + nx * y] + 1;
        if (omp_get_thread_num() == 0 && filter_output())
          std::cout << x << " " << y << ": " << val << " " << d(i, j) << std::endl;
        verify(val, d(i, j));
      }
  }
  delete[] data;
#endif

  std::cout << std::endl << "all tests passed" << std::endl;

  return 0;
}
