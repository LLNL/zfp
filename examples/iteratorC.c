#include <stdio.h>
#include <stdlib.h>
#include "zfp/array.h"

void print1(cfp_ptr1d p, size_t n)
{
  size_t i;
  const cfp_array1d_api _ = cfp.array1d;

  for (i = 0; i < n; i++)
    printf("%g\n", _.reference.get(_.pointer.ref_at(p, i)));
}

void print2(cfp_ptr2d p, size_t n)
{
  const cfp_array2d_api _ = cfp.array2d;

  while (n--) {
    printf("%g\n", _.reference.get(_.pointer.ref(p)));
    p = _.pointer.inc(p);
  }
}

void print3(cfp_iter1d begin, cfp_iter1d end)
{
  const cfp_array1d_api _ = cfp.array1d;
  cfp_iter1d p;

  for (p = begin; !_.iterator.eq(p, end); p = _.iterator.inc(p))
    printf("%g\n", _.reference.get(_.iterator.ref(p)));
}

int main(void)
{
  const cfp_array1d_api _1d = cfp.array1d;
  const cfp_array2d_api _2d = cfp.array2d;
  const cfp_array3d_api _3d = cfp.array3d;
  cfp_array1d v;
  cfp_iter1d it1;
  cfp_array2d a;
  cfp_iter2d it2;
  cfp_ptr2d pb2;
  cfp_ptr2d pe2;
  cfp_array3d b;
  cfp_iter3d it3;
  cfp_ptr3d pb3;
  cfp_ptr3d pe3;
  size_t i, j, k;

  /* some fun with 1D arrays */
  v = _1d.ctor(10, 64.0, 0, 0);
  /* initialize and print array of random values */
  for (it1 = _1d.begin(v); !_1d.iterator.eq(it1, _1d.end(v)); it1 = _1d.iterator.inc(it1))
    _1d.reference.set(_1d.iterator.ref(it1), rand());
  printf("random array\n");
  print1(_1d.ptr(v, 0), _1d.size(v)); 
  printf("\n");

  /* some fun with 2D arrays */
  a = _2d.ctor(5, 7, 64.0, 0, 0);
  /* print array indices visited in block-order traversal*/
  printf("block order (x, y) indices\n");
  for (it2 = _2d.begin(a); !_2d.iterator.eq(it2, _2d.end(a)); it2 = _2d.iterator.inc(it2)) {
    i = _2d.iterator.i(it2);
    j = _2d.iterator.j(it2);
    printf("(%lu, %lu)\n", (unsigned long)i, (unsigned long)j);
    _2d.reference.set(_2d.iterator.ref(it2), i + 10 * j);
  }
  printf("\n");

  /* print array contents in row-major order */
  printf("row-major order yx indices\n");
  print2(_2d.ptr_flat(a, 0), _2d.size(a));
  printf("\n");
  /* pointer arithmetic */
  pb2 = _2d.reference.ptr(_2d.iterator.ref(_2d.begin(a)));
  pe2 = _2d.reference.ptr(_2d.iterator.ref(_2d.end(a)));
  printf("%lu * %lu = %ld\n", (unsigned long)_2d.size_x(a), (unsigned long)_2d.size_y(a), (long)_2d.pointer.distance(pb2, pe2));

  /* some fun with 3D arrays */
  b = _3d.ctor(7, 2, 5, 64.0, 0, 0);
  /* print array indices visited in block-order traversal */
  printf("block order (x, y, z) indices\n");
  for (it3 = _3d.begin(b); !_3d.iterator.eq(it3, _3d.end(b)); it3 = _3d.iterator.inc(it3)) {
    i = _3d.iterator.i(it3);
    j = _3d.iterator.j(it3);
    k = _3d.iterator.k(it3);
    printf("(%lu, %lu, %lu)\n", (unsigned long)i, (unsigned long)j, (unsigned long)k);
  }
  printf("\n");
  /* pointer arithmetic */
  pb3 = _3d.reference.ptr(_3d.iterator.ref(_3d.begin(b)));
  pe3 = _3d.reference.ptr(_3d.iterator.ref(_3d.end(b)));
  printf("%lu * %lu * %lu = %ld\n", (unsigned long)_3d.size_x(b), (unsigned long)_3d.size_y(b), (unsigned long)_3d.size_z(b), (long)_3d.pointer.distance(pb3, pe3));

  return 0;
}
