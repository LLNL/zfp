#ifndef ZFP_ARRAY_HPP
#define ZFP_ARRAY_HPP

#include <algorithm>
#include <climits>
#include <string>
#include "zfp.h"
#include "zfp/internal/array/exception.hpp"

namespace zfp {

// abstract base class for compressed array of scalars
class array {
public:
  #include "zfp/internal/array/header.hpp"

  // factory function (see zfpfactory.h)
  static zfp::array* construct(const zfp::array::header& header, const void* buffer = 0, size_t buffer_size_bytes = 0);

  // public virtual destructor (can delete array through base class pointer)
  virtual ~array() {}

  // underlying scalar type
  zfp_type scalar_type() const { return type; }

  // dimensionality
  uint dimensionality() const { return dims; }

  // rate in bits per value
  virtual double rate() const = 0;

  // compressed data size and buffer
  virtual size_t compressed_size() const = 0;
  virtual void* compressed_data() const = 0;

protected:
  // default constructor
  array() :
    type(zfp_type_none),
    dims(0),
    nx(0), ny(0), nz(0), nw(0)
  {}

  // generic array with 'dims' dimensions and scalar type 'type'
  explicit array(uint dims, zfp_type type) :
    type(type),
    dims(dims),
    nx(0), ny(0), nz(0), nw(0)
  {}

  // constructor from previously-serialized compressed array
  explicit array(uint dims, zfp_type type, const zfp::array::header& header) :
    type(type),
    dims(dims),
    nx(header.size_x()), ny(header.size_y()), nz(header.size_z()), nw(header.size_w())
  {
    if (header.scalar_type() != type)
      throw zfp::exception("zfp array scalar type does not match header");
    if (header.dimensionality() != dims)
      throw zfp::exception("zfp array dimensionality does not match header");
  }

  // copy constructor--performs a deep copy
  array(const array& a)
  {
    deep_copy(a);
  }

  // assignment operator--performs a deep copy
  array& operator=(const array& a)
  {
    deep_copy(a);
    return *this;
  }

  // perform a deep copy
  void deep_copy(const array& a)
  {
    // copy metadata
    type = a.type;
    dims = a.dims;
    nx = a.nx;
    ny = a.ny;
    nz = a.nz;
    nw = a.nw;
  }

  zfp_type type;         // scalar type
  uint dims;             // array dimensionality (1, 2, 3, or 4)
  size_t nx, ny, nz, nw; // array dimensions
};

}

#endif
