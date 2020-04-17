#ifndef ZFP_HEADER_H
#define ZFP_HEADER_H

namespace zfp {

// abstract base class for array header
class header {
public:
  virtual ~header() {}

  // array scalar type
  virtual zfp_type scalar_type() const = 0;

  // array dimensionality
  virtual uint dimensionality() const
  {
    return size_z() ? 3 : size_y() ? 2 : size_x() ? 1 : 0;
  }

  // array dimensions
  virtual size_t size_x() const = 0;
  virtual size_t size_y() const = 0;
  virtual size_t size_z() const = 0;

  // rate in bits per value
  virtual double rate() const = 0;

  // header payload: data pointer and byte size
  virtual const void* data() const = 0;
  virtual size_t size() const = 0;
};

}

#endif
