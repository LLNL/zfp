#include <algorithm>
#include <cstdlib>
#include <iostream>
#include "zfparray1.h"
#include "zfparray2.h"
#include "zfparray3.h"

void print1(zfp::array1<double>::pointer p, size_t n)
{
  for (size_t i = 0; i < n; i++)
    std::cout << p[i] << std::endl;
}

void print2(zfp::array2<double>::pointer p, size_t n)
{
  while (n--)
    std::cout << *p++ << std::endl;
}

void print3(zfp::array1<double>::iterator begin, zfp::array1<double>::iterator end)
{
  for (zfp::array1<double>::iterator p = begin; p != end; p++)
    std::cout << *p << std::endl;
}

int main()
{
  // some fun with 1D arrays
  zfp::array1<double> v(10, 64.0);
  // initialize and print array of random values
  for (zfp::array1<double>::iterator p = v.begin(); p != v.end(); p++)
    *p = rand();
  std::cout << "random array" << std::endl;
  print1(&v[0], v.size());
  std::cout << std::endl;
  // sorting is possible via random access iterators (1D arrays only)
  std::sort(v.begin(), v.end());
  // print array using iteration
  std::cout << "sorted array" << std::endl;
  print3(v.begin(), v.end());
  std::cout << std::endl;

  // some fun with 2D arrays
  zfp::array2<double> a(5, 7, 64.0);
  // print array indices visited in block-order traversal
  std::cout << "block order (x, y) indices" << std::endl;
  for (zfp::array2<double>::iterator p = a.begin(); p != a.end(); p++) {
    std::cout << "(" << p.i() << ", " << p.j() << ")" << std::endl;
    *p = p.i() + 10 * p.j();
  }
  std::cout << std::endl;
  // print array contents in row-major order
  std::cout << "row-major order yx indices" << std::endl;
  print2(&a[0], a.size());
  std::cout << std::endl;
  // pointer arithmetic
  std::cout << a.size_x() << " * " << a.size_y() << " = " << (&*a.end() - &*a.begin()) << std::endl;
  // min and max values
  std::cout << "min = " << *std::min_element(a.begin(), a.end()) << std::endl;
  std::cout << "max = " << *std::max_element(a.begin(), a.end()) << std::endl;
  std::cout << std::endl;

  // some fun with 3D arrays
  zfp::array3<double> b(7, 2, 5, 64.0);
  // print array indices visited in block-order traversal
  std::cout << "block order (x, y, z) indices" << std::endl;
  for (zfp::array3<double>::iterator p = b.begin(); p != b.end(); p++)
    std::cout << "(" << p.i() << ", " << p.j() << ", " << p.k() << ")" << std::endl;
  std::cout << std::endl;
  // pointer arithmetic
  std::cout << b.size_x() << " * " << b.size_y() << " * " << b.size_z() << " = " << (&*b.end() - &*b.begin()) << std::endl;

  return 0;
}
