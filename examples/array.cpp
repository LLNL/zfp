// simple example that shows how to work with zfp's compressed-array classes

#include <iostream>
#include <vector>
#include "zfp/array2.hpp"

int main()
{
  // array dimensions (can be arbitrary) and zfp memory footprint
  const size_t nx = 12;
  const size_t ny = 8;
  const double bits_per_value = 4.0;

  // declare 2D arrays using STL and zfp
  std::vector<double> vec(nx * ny);
  zfp::array2<double> arr(nx, ny, bits_per_value);

  // initialize arrays to linear ramp
  for (size_t y = 0; y < ny; y++)
    for (size_t x = 0; x < nx; x++)
      arr(x, y) = vec[x + nx * y] = x + nx * y;

  // alternative initialization of entire array, arr:
  // arr.set(&vec[0]);

  // optional: force compression of cached data
  arr.flush_cache();

  // print values
  for (size_t y = 0; y < ny; y++)
    for (size_t x = 0; x < nx; x++)
      std::cout << vec[x + nx * y] << " " << arr(x, y) << std::endl;

  // alternative using printf(); note the necessary cast:
  // printf("%g %g\n", vec[x + nx * y], (double)arr(x, y));

  // print storage size of payload data
  std::cout << "vec bytes = " << vec.capacity() * sizeof(vec[0]) << std::endl;
  std::cout << "zfp bytes = " << arr.size_bytes(ZFP_DATA_PAYLOAD) << std::endl;

  return 0;
}
