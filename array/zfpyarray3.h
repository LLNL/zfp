#ifndef ZFPY_ARRAY2_H
#define ZFPY_ARRAY2_H

#include "zfparray2.h"

namespace zfp {

// zfpy interface (required because of current cython limitations)
template < typename Scalar, class Codec = zfp::codec<Scalar> >
class py_array3 : public array3<Scalar, Codec> {
public:
  py_array3(uint nx, uint ny, uint nz, double rate, const Scalar* p = 0, size_t csize = 0) :
    array3<Scalar, Codec>(nx, ny, nz, rate, p, csize) {}

  // inspector
  Scalar get(uint i, uint j, uint k) const {
    return array3<Scalar, Codec>::get(i, j, k);
  }

  // mutator
  void set(uint i, uint j, uint k, Scalar val) {
    array3<Scalar, Codec>::set(i, j, k, val);
  }
};

}

#endif
