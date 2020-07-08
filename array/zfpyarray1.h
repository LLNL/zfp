#ifndef ZFPY_ARRAY1_H
#define ZFPY_ARRAY1_H

#include "zfparray1.h"

namespace zfp {

// zfpy interface (required because of current cython limitations)
template < typename Scalar, class Codec = zfp::codec<Scalar> >
class py_array1 : public array1<Scalar, Codec> {
public:
  py_array1(unsigned int nx, double rate, const Scalar* p = 0, size_t csize = 0) :
    array1<Scalar, Codec>(nx, rate, p, csize) {}

  // inspector
  Scalar get(uint i) const {
    return array1<Scalar, Codec>::get(i);
  }

  // mutator
  void set(uint i, Scalar val) {
    array1<Scalar, Codec>::set(i, val);
  }
};

}

#endif
