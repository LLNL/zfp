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

  py_array3(const zfp::array::header& h, const uchar* buffer = 0, size_t buffer_size_bytes = 0) :
    array3<Scalar>(h, buffer, buffer_size_bytes) {}

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
