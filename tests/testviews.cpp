#include <cmath>
#include <cstdlib>
#include <iostream>
#include <sstream>
#include "zfparray2.h"
#include "zfparray3.h"

#define EPSILON 1e-3

// random integer in {begin, ..., end - 1}
static size_t
rand(size_t begin, size_t end)
{
  return begin + size_t(rand()) % (end - begin);
}

// ensure f and g are sufficiently close
static void
verify(double f, double g)
{
  if (std::fabs(f - g) > EPSILON) {
    std::cerr << "error: " << f << " != " << g << std::endl;
    exit(EXIT_FAILURE);
  }
}

static int
usage()
{
  std::cerr << "Usage: testviews [nx ny nz [x0 y0 z0 mx my mz]]" << std::endl;
  return EXIT_FAILURE;
}

int main(int argc, char* argv[])
{
  size_t nx = 16;
  size_t ny = 16;
  size_t nz = 16;
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
    x0 = rand(0, nx);
    y0 = rand(0, ny);
    z0 = rand(0, nz);
    mx = rand(0, nx - x0);
    my = rand(0, ny - y0);
    mz = rand(0, nz - z0);
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
  for (size_t z = 0; z < v.size_z(); z++)
    for (size_t y = 0; y < v.size_y(); y++)
      for (size_t x = 0; x < v.size_x(); x++) {
        std::cout << x << " " << y << " " << z << ": " << a(x0 + x, y0 + y, z0 + z) << " " << v(x, y, z) << std::endl;
        verify(a(x0 + x, y0 + y, z0 + z), v(x, y, z));
      }

  // flat view of all of a
  std::cout << std::endl << "3D flat view" << std::endl;
  zfp::array3<double>::flat_view fv(&a);
  for (size_t z = 0; z < fv.size_z(); z++)
    for (size_t y = 0; y < fv.size_y(); y++)
      for (size_t x = 0; x < fv.size_x(); x++) {
        std::cout << x << " " << y << " " << z << ": " << a(x, y, z) << " " << fv[fv.index(x, y, z)] << std::endl;
        verify(a(x, y, z), fv[fv.index(x, y, z)]);
      }

  // nested view of all of a
  std::cout << std::endl << "3D nested view" << std::endl;
  zfp::array3<double>::nested_view nv(&a);
  for (size_t z = 0; z < v.size_z(); z++)
    for (size_t y = 0; y < v.size_y(); y++)
      for (size_t x = 0; x < v.size_x(); x++) {
        std::cout << x << " " << y << " " << z << ": " << a(x, y, z) << " " << nv[z][y][x] << std::endl;
        verify(a(x, y, z), nv[z][y][x]);
      }

  // pointers and iterators into a via view v
  std::cout << std::endl << "3D view pointers and iterators" << std::endl;
  zfp::array3<double>::view::const_reference vr = v(0, 0, 0);
  zfp::array3<double>::view::const_pointer p = &vr;
  p = &v(0, 0, 0);
  for (zfp::array3<double>::view::const_iterator it = v.begin(); it != v.end(); it++) {
    size_t x = it.i();
    size_t y = it.j();
    size_t z = it.k();
    verify(*it, p[x + mx * (y + my * z)]);
  }

  // pointers and iterators into a via flat view fv
  std::cout << std::endl << "3D flat view pointers and iterators" << std::endl;
  zfp::array3<double>::flat_view::const_reference fvr = fv[0];
  zfp::array3<double>::flat_view::const_pointer fp = &fvr;
  fp = &fv(0, 0, 0);
  for (zfp::array3<double>::flat_view::const_iterator it = fv.begin(); it != fv.end(); it++) {
    size_t x = it.i();
    size_t y = it.j();
    size_t z = it.k();
    verify(*it, fp[x + nx * (y + ny * z)]);
  }

  // 2D slice of a
  std::cout << std::endl << "2D slice" << std::endl;
  size_t z = rand(0, nv.size_z());
  zfp::array3<double>::nested_view2 slice2(nv[z]);
  for (size_t y = 0; y < slice2.size_y(); y++)
    for (size_t x = 0; x < slice2.size_x(); x++) {
      std::cout << x << " " << y << " " << z << ": " << a(x, y, z) << " " << slice2[y][x] << std::endl;
      verify(a(x, y, z), slice2[y][x]);
    }

  // 2D array constructed from 2D slice (exercises deep copy via iterator)
  std::cout << std::endl << "2D array from 2D slice" << std::endl;
  zfp::array2<double> b(slice2);
  for (size_t y = 0; y < b.size_y(); y++)
    for (size_t x = 0; x < b.size_x(); x++) {
      std::cout << x << " " << y << ": " << b(x, y) << " " << slice2[y][x] << std::endl;
      verify(b(x, y), slice2[y][x]);
    }

  // 1D slice of a
  std::cout << std::endl << "1D slice" << std::endl;
  size_t y = rand(0, slice2.size_y());
  zfp::array3<double>::nested_view1 slice1 = slice2[y];
  for (size_t x = 0; x < slice1.size_x(); x++) {
    std::cout << x << " " << y << " " << z << ": " << a(x, y, z) << " " << slice1[x] << std::endl;
    verify(a(x, y, z), slice1[x]);
  }

  // 2D array constructed from 2D slice of 3D array (exercises deep copy via iterator)
  std::cout << std::endl << "2D array from 2D slice of 3D array" << std::endl;
  zfp::array2<double> c(slice2);
  for (size_t y = 0; y < c.size_y(); y++)
    for (size_t x = 0; x < c.size_x(); x++) {
      std::cout << x << " " << y << ": " << c(x, y) << " " << slice2[y][x] << std::endl;
      verify(c(x, y), slice2[y][x]);
    }

  // 2D thread-safe view of c
  std::cout << std::endl << "2D private view" << std::endl;
  zfp::array2<double>::private_const_view d(&c);
  for (size_t y = 0; y < c.size_y(); y++)
    for (size_t x = 0; x < c.size_x(); x++) {
      std::cout << x << " " << y << ": " << c(x, y) << " " << d(x, y) << std::endl;
      verify(c(x, y), d(x, y));
    }

  std::cout << std::endl << "all tests passed" << std::endl;

  return 0;
}
