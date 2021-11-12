#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstdlib>
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
    fprintf(stderr, "error: %g != %g\n", f, g);
    exit(EXIT_FAILURE);
  }
}

int main(int argc, char* argv[])
{
  size_t nx = 16;
  size_t ny = 16;
  size_t nz = 16;
  size_t x0 = rand(0, nx);
  size_t y0 = rand(0, ny);
  size_t z0 = rand(0, nz);
  size_t mx = rand(1, nx - x0);
  size_t my = rand(1, ny - y0);
  size_t mz = rand(1, nz - z0);
  double rate = 16;

  // Usage: test [nx ny nz [x0 y0 z0 mx my mz]]
  switch (argc) {
    case 10:
      if (sscanf(argv[4], "%zu", &x0) != 1 ||
          sscanf(argv[5], "%zu", &y0) != 1 ||
          sscanf(argv[6], "%zu", &z0) != 1 ||
          sscanf(argv[7], "%zu", &mx) != 1 ||
          sscanf(argv[8], "%zu", &my) != 1 ||
          sscanf(argv[9], "%zu", &mz) != 1)
        return EXIT_FAILURE;
      // FALLTHROUGH
    case 4:
      if (sscanf(argv[1], "%zu", &nx) != 1 ||
          sscanf(argv[2], "%zu", &ny) != 1 ||
          sscanf(argv[3], "%zu", &nz) != 1)
        return EXIT_FAILURE;
      // FALLTHROUGH
    case 1:
      break;
  }

  printf("a(%zu, %zu, %zu)\n", nx, ny, nz);
  printf("v(%zu, %zu, %zu) + (%zu, %zu, %zu)\n", mx, my, mz, x0, y0, z0);

  // initialize 3D array to linear function
  zfp::array3<double> a(nx, ny, nz, rate);
  for (size_t z = 0; z < nz; z++)
    for (size_t y = 0; y < ny; y++)
      for (size_t x = 0; x < nx; x++)
        a(x, y, z) = static_cast<double>(x + nx * (y + ny * z));

  // rectangular view into a
  printf("\n3D view\n");
  zfp::array3<double>::view v(&a, x0, y0, z0, mx, my, mz);
  for (size_t z = 0; z < v.size_z(); z++)
    for (size_t y = 0; y < v.size_y(); y++)
      for (size_t x = 0; x < v.size_x(); x++) {
        printf("%zu %zu %zu: %g %g\n", x, y, z, (double)a(x0 + x, y0 + y, z0 + z), (double)v(x, y, z));
        verify(a(x0 + x, y0 + y, z0 + z), v(x, y, z));
      }

  // flat view of all of a
  printf("\n3D flat view\n");
  zfp::array3<double>::flat_view fv(&a);
  for (size_t z = 0; z < fv.size_z(); z++)
    for (size_t y = 0; y < fv.size_y(); y++)
      for (size_t x = 0; x < fv.size_x(); x++) {
        printf("%zu %zu %zu: %g %g\n", x, y, z, (double)a(x, y, z), (double)fv[fv.index(x, y, z)]);
        verify(a(x, y, z), fv[fv.index(x, y, z)]);
      }

  // nested view of all of a
  printf("\n3D nested view\n");
  zfp::array3<double>::nested_view nv(&a);
  for (size_t z = 0; z < v.size_z(); z++)
    for (size_t y = 0; y < v.size_y(); y++)
      for (size_t x = 0; x < v.size_x(); x++) {
        printf("%zu %zu %zu: %g %g\n", x, y, z, (double)a(x, y, z), (double)nv[z][y][x]);
        verify(a(x, y, z), nv[z][y][x]);
      }

  // pointers and iterators into a via view v
  printf("\n3D view pointers and iterators\n");
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
  printf("\n3D flat view pointers and iterators\n");
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
  printf("\n2D slice\n");
  size_t z = rand(0, nv.size_z());
  zfp::array3<double>::nested_view2 slice2(nv[z]);
  for (size_t y = 0; y < slice2.size_y(); y++)
    for (size_t x = 0; x < slice2.size_x(); x++) {
      printf("%zu %zu %zu: %g %g\n", x, y, z, (double)a(x, y, z), (double)slice2[y][x]);
      verify(a(x, y, z), slice2[y][x]);
    }

  // 2D array constructed from 2D slice (exercises deep copy via iterator)
  printf("\n2D array from 2D slice\n");
  zfp::array2<double> b(slice2);
  for (size_t y = 0; y < b.size_y(); y++)
    for (size_t x = 0; x < b.size_x(); x++) {
      printf("%zu %zu: %g %g\n", x, y, (double)b(x, y), (double)slice2[y][x]);
      verify(b(x, y), slice2[y][x]);
    }

  // 1D slice of a
  printf("\n1D slice\n");
  size_t y = rand(0, slice2.size_y());
  zfp::array3<double>::nested_view1 slice1 = slice2[y];
  for (size_t x = 0; x < slice1.size_x(); x++) {
    printf("%zu %zu %zu: %g %g\n", x, y, z, (double)a(x, y, z), (double)slice1[x]);
    verify(a(x, y, z), slice1[x]);
  }

  // 2D array constructed from 2D slice of 3D array (exercises deep copy via iterator)
  printf("\n2D array from 2D slice of 3D array\n");
  zfp::array2<double> c(slice2);
  for (size_t y = 0; y < c.size_y(); y++)
    for (size_t x = 0; x < c.size_x(); x++) {
      printf("%zu %zu: %g %g\n", x, y, (double)c(x, y), (double)slice2[y][x]);
      verify(c(x, y), slice2[y][x]);
    }

  // 2D thread-safe view of c
  printf("\n2D private view\n");
  zfp::array2<double>::private_const_view d(&c);
  for (size_t y = 0; y < c.size_y(); y++)
    for (size_t x = 0; x < c.size_x(); x++) {
      printf("%zu %zu: %g %g\n", x, y, (double)c(x, y), (double)d(x, y));
      verify(c(x, y), d(x, y));
    }

  printf("\nall tests passed\n");

  return 0;
}
