#ifndef ZFP_ARRAY_H
#define ZFP_ARRAY_H

#include <algorithm>
#include <climits>
//#include <cstring>
#include <stdexcept>
#include <string>
#include "zfp.h"

// all undefined at end
#define DIV_ROUND_UP(x, y) (((x) + (y) - 1) / (y))
#define BITS_TO_BYTES(x) DIV_ROUND_UP(x, CHAR_BIT)

#define ZFP_HEADER_SIZE_BITS (ZFP_MAGIC_BITS + ZFP_META_BITS + ZFP_MODE_SHORT_BITS)

namespace zfp {

// utility functions
template <typename Scalar>
inline zfp_type scalar_type();

template <>
inline zfp_type scalar_type<float>() { return zfp_type_float; }

template <>
inline zfp_type scalar_type<double>() { return zfp_type_double; }

// abstract base class for compressed array of scalars
class array {
public:
  #include "zfp/header.h"

  static zfp::array* construct(const zfp::array::header& header, const void* buffer = 0, size_t buffer_size_bytes = 0);

protected:
  // default constructor
  array() :
    dims(0), type(zfp_type_none),
    nx(0), ny(0), nz(0)
  {}

  // generic array with 'dims' dimensions and scalar type 'type'
  explicit array(uint dims, zfp_type type) :
    dims(dims), type(type),
    nx(0), ny(0), nz(0)
  {}

  // constructor, from previously-serialized compressed array
//  array(uint dims, zfp_type type, const zfp::array::header& h, size_t expected_buffer_size_bytes) :
  array(uint dims, zfp_type type, const zfp::array::header&, size_t) :
    dims(dims), type(type),
    nx(0), ny(0), nz(0)
  {
#if 0
    // read header to populate member variables associated with zfp_stream
    try {
      read_from_header(h);
    } catch (zfp::array::header::exception const &) {
      zfp_stream_close(zfp);
      throw;
    }

    if (expected_buffer_size_bytes && !is_valid_buffer_size(zfp, nx, ny, nz, expected_buffer_size_bytes)) {
      zfp_stream_close(zfp);
      throw zfp::array::header::exception("ZFP header expects a longer buffer than what was passed in.");
    }
#else
    throw std::runtime_error("(de)serialization not supported");
#endif
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

public:
  // public virtual destructor (can delete array through base class pointer)
  virtual ~array() {}

  // dimensionality
  uint dimensionality() const { return dims; }

  // underlying scalar type
  zfp_type scalar_type() const { return type; }

  // compressed data size and buffer
  virtual size_t compressed_size() const = 0;
  virtual void* compressed_data() const = 0;

  // write header with latest metadata
  zfp::array::header get_header() const
  {
#if 0
    // intermediate buffer needed (bitstream accesses multiples of wordsize)
    AlignedBufferHandle abh;
    DualBitstreamHandle dbh(zfp, abh);

    ZfpFieldHandle zfh(type, nx, ny, nz);

    // avoid long header (alignment issue)
    if (zfp_stream_mode(zfp) > ZFP_MODE_SHORT_MAX)
      throw zfp::array::header::exception("ZFP compressed arrays only support short headers at this time.");

    if (!zfp_write_header(zfp, zfh.field, ZFP_HEADER_FULL))
      throw zfp::array::header::exception("ZFP could not write a header to buffer.");
    stream_flush(zfp->stream);

    zfp::array::header h;
    abh.copy_to_header(&h);

    return h;
#else
    throw std::runtime_error("(de)serialization not supported");
#endif
  }

private:
  // private members used when reading/writing headers
  #include "zfp/headerHelpers.h"

protected:
  // perform a deep copy
  void deep_copy(const array& a)
  {
    // copy metadata
    dims = a.dims;
    type = a.type;
    nx = a.nx;
    ny = a.ny;
    nz = a.nz;
  }

  // attempt reading header from zfp::array::header
  // and verify header contents (throws exceptions upon failure)
//  void read_from_header(const zfp::array::header& h)
  void read_from_header(const zfp::array::header&)
  {
#if 0
    // copy header into aligned buffer
    AlignedBufferHandle abh(&h);
    DualBitstreamHandle dbh(zfp, abh);
    ZfpFieldHandle zfh;

    // read header to populate member variables associated with zfp_stream
    size_t readbits = zfp_read_header(zfp, zfh.field, ZFP_HEADER_FULL);
    if (!readbits)
      throw zfp::array::header::exception("Invalid ZFP header.");
    else if (readbits != ZFP_HEADER_SIZE_BITS)
      throw zfp::array::header::exception("ZFP compressed arrays only support short headers at this time.");

    // verify metadata on zfp_field match that for this object
    std::string err_msg = "";
    if (type != zfp_field_type(zfh.field))
      zfp::array::header::concat_sentence(err_msg, "ZFP header specified an underlying scalar type different than that for this object.");

    if (dims != zfp_field_dimensionality(zfh.field))
      zfp::array::header::concat_sentence(err_msg, "ZFP header specified a dimensionality different than that for this object.");

    verify_header_contents(zfp, zfh.field, err_msg);

    if (!err_msg.empty())
      throw zfp::array::header::exception(err_msg);

    // set class variables
    nx = zfh.field->nx;
    ny = zfh.field->ny;
    nz = zfh.field->nz;
    type = zfh.field->type;
    blkbits = zfp->maxbits;
#else
    throw std::runtime_error("(de)serialization not supported");
#endif
  }

  uint dims;       // array dimensionality (1, 2, or 3)
  zfp_type type;   // scalar type
  uint nx, ny, nz; // array dimensions
};

#undef DIV_ROUND_UP
#undef BITS_TO_BYTES

#undef ZFP_HEADER_SIZE_BITS

}

#endif
