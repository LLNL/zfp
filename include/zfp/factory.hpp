#ifndef ZFP_FACTORY_HPP
#define ZFP_FACTORY_HPP

// ensure zfp/array.hpp has already been included
#ifndef ZFP_ARRAY_HPP
  #error "zfp/array.hpp must be included before zfp/factory.hpp"
#endif

zfp::array* zfp::array::construct(const zfp::array::header& header, const void* buffer, size_t buffer_size_bytes)
{
  // extract metadata from header
  const zfp_type type = header.scalar_type();
  const double rate = header.rate();
  const uint dims = header.dimensionality();
  const size_t nx = header.size_x();
  const size_t ny = header.size_y();
  const size_t nz = header.size_z();
  const size_t nw = header.size_w();

  // construct once (passing zfp::array::header will read it again)
  zfp::array* arr = 0;
  std::string error;
  switch (dims) {
    case 4:
#ifdef ZFP_ARRAY4_HPP
      switch (type) {
        case zfp_type_float:
          arr = new zfp::array4f(nx, ny, nz, nw, rate);
          break;
        case zfp_type_double:
          arr = new zfp::array4d(nx, ny, nz, nw, rate);
          break;
        default:
          /* NOTREACHED */
          error = "zfp scalar type not supported";
          break;
      }
#else
      error = "array4 not supported; include zfp/array4.hpp before zfp/factory.hpp";
#endif
      break;

    case 3:
#ifdef ZFP_ARRAY3_HPP
      switch (type) {
        case zfp_type_float:
          arr = new zfp::array3f(nx, ny, nz, rate);
          break;
        case zfp_type_double:
          arr = new zfp::array3d(nx, ny, nz, rate);
          break;
        default:
          /* NOTREACHED */
          error = "zfp scalar type not supported";
          break;
      }
#else
      error = "array3 not supported; include zfp/array3.hpp before zfp/factory.hpp";
#endif
      break;

    case 2:
#ifdef ZFP_ARRAY2_HPP
      switch (type) {
        case zfp_type_float:
          arr = new zfp::array2f(nx, ny, rate);
          break;
        case zfp_type_double:
          arr = new zfp::array2d(nx, ny, rate);
          break;
        default:
          /* NOTREACHED */
          error = "zfp scalar type not supported";
          break;
      }
#else
      error = "array2 not supported; include zfp/array2.hpp before zfp/factory.hpp";
#endif
      break;

    case 1:
#ifdef ZFP_ARRAY1_HPP
      switch (type) {
        case zfp_type_float:
          arr = new zfp::array1f(nx, rate);
          break;
        case zfp_type_double:
          arr = new zfp::array1d(nx, rate);
          break;
        default:
          /* NOTREACHED */
          error = "zfp scalar type not supported";
          break;
      }
#else
      error = "array1 not supported; include zfp/array1.hpp before zfp/factory.hpp";
#endif
      break;

    default:
      error = "zfp array dimensionality other than {1, 2, 3, 4} not supported";
      break;
  }

  if (!error.empty())
    throw zfp::exception(error);

  if (buffer) {
    if (buffer_size_bytes && buffer_size_bytes < arr->compressed_size()) {
      delete arr;
      throw zfp::exception("zfp buffer size is smaller than required");
    }
    std::memcpy(arr->compressed_data(), buffer, arr->compressed_size());
  }

  return arr;
}

#endif
