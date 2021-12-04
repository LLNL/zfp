#ifndef ZFP_VARRAY_H
#define ZFP_VARRAY_H

#include <algorithm>
#include <climits>
#include <string>
#include "zfp.h"
#include "zfp/exception.h"

namespace zfp {

// abstract base class for variable-rate compressed array of scalars
class var_array {
public:
//  typedef Tile::storage storage;

  // public virtual destructor (can delete array through base class pointer)
  virtual ~var_array() {}

  // total number of elements in array
  virtual size_t size() const = 0;

  // underlying scalar type
  zfp_type scalar_type() const { return type; }

  // dimensionality
  uint dimensionality() const { return dims; }

//  virtual zfp_config configuration() const = 0;

protected:
  // default constructor
  var_array() :
    type(zfp_type_none),
    dims(0),
    nx(0), ny(0), nz(0), nw(0)
  {}

  // generic array with 'dims' dimensions and scalar type 'type'
  explicit var_array(uint dims, zfp_type type) :
    type(type),
    dims(dims),
    nx(0), ny(0), nz(0), nw(0)
  {}

  // copy constructor--performs a deep copy
  var_array(const var_array& a)
  {
    deep_copy(a);
  }

  // assignment operator--performs a deep copy
  var_array& operator=(const var_array& a)
  {
    if (this != &a)
      deep_copy(a);
    return *this;
  }

  // perform a deep copy
  void deep_copy(const var_array& a)
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
//  uint minbits;          // smallest non-empty block size in bits
};

}

#endif
