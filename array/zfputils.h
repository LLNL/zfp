#ifndef ZFP_UTILS_H
#define ZFP_UTILS_H

// (assumes zfparray.h already included)

static void read_header_contents(const zfp::array::header& header, size_t bufferSizeBytes, uint& dims, zfp_type& type, double& rate, uint* n)
{
  // create zfp_stream and zfp_field structs to call C API zfp_read_header()
  uchar* buffer = new uchar[ZFP_HEADER_PADDED_TO_WORD_BYTES];
  memcpy(buffer, header.buffer, ZFP_HEADER_SIZE_BYTES);

  bitstream* bs = stream_open(buffer, ZFP_HEADER_PADDED_TO_WORD_BYTES);
  zfp_stream* stream = zfp_stream_open(bs);
  zfp_field* field = zfp_field_alloc();

  std::string errMsg = "";
  if (!zfp_read_header(stream, field, ZFP_HEADER_FULL)) {
    errMsg += "Invalid ZFP header.";
  } else {
    // gather metadata
    dims = zfp_field_dimensionality(field);
    type = zfp_field_type(field);

    uint numBlockEntries = 1u << (2 * dims);
    rate = (double)stream->maxbits / numBlockEntries;

    zfp_field_size(field, n);

    // validate metadata, accumulate exception msgs
    if (n[3]) {
      errMsg += "ZFP compressed arrays do not yet support dimensionalities beyond 1, 2, and 3.";
    }

    if (type < zfp_type_float) {
      if (!errMsg.empty())
        errMsg += " ";
      errMsg += "ZFP compressed arrays do not yet support scalar types beyond floats and doubles.";
    }

    if (bufferSizeBytes != 0) {
      // verify buffer is large enough, with what header describes
      uint mx = ((std::max(n[0], 1u)) + 3) / 4;
      uint my = ((std::max(n[1], 1u)) + 3) / 4;
      uint mz = ((std::max(n[2], 1u)) + 3) / 4;
      size_t blocks = (size_t)mx * (size_t)my * (size_t)mz;
      size_t describedSize = ((blocks * stream->maxbits + stream_word_bits - 1) & ~(stream_word_bits - 1)) / CHAR_BIT;
      if (bufferSizeBytes < describedSize) {
        if (!errMsg.empty())
          errMsg += " ";
        errMsg += "ZFP header expects a longer buffer than what was passed in.";
      }
    }
  }

  zfp_field_free(field);
  zfp_stream_close(stream);
  stream_close(bs);
  delete[] buffer;

  if (!errMsg.empty())
    throw zfp::array::header_exception(errMsg);
}

zfp::array* zfp::array::construct(const zfp::array::header& header, const uchar* buffer, size_t bufferSizeBytes)
{
  // gather array metadata via C API, then construct with metadata
  uint dims = 0;
  zfp_type type = zfp_type_none;
  double rate = 0;
  uint n[4] = {0};

  // read once (will throw if reads a noncompatible header)
  read_header_contents(header, bufferSizeBytes, dims, type, rate, n);

  // construct once (passing zfp::array::header will read it again)
  zfp::array* arr = 0;
  std::string errMsg = "";
  switch (dims) {
    case 3:
#ifdef ZFP_ARRAY3_H
      switch (type) {
        case zfp_type_double:
          arr = new zfp::array3d(n[0], n[1], n[2], rate);
          break;

        case zfp_type_float:
          arr = new zfp::array3f(n[0], n[1], n[2], rate);
          break;

        default:
          errMsg = "ZFP compressed arrays do not yet support scalar types beyond floats and doubles.";
          break;
      }
#else
      errMsg = "Header files for 3 dimensional ZFP compressed arrays were not included.";
#endif
      break;

    case 2:
#ifdef ZFP_ARRAY2_H
      switch (type) {
        case zfp_type_double:
          arr = new zfp::array2d(n[0], n[1], rate);
          break;

        case zfp_type_float:
          arr = new zfp::array2f(n[0], n[1], rate);
          break;

        default:
          errMsg = "ZFP compressed arrays do not yet support scalar types beyond floats and doubles.";
          break;
      }
#else
      errMsg = "Header files for 2 dimensional ZFP compressed arrays were not included.";
#endif
      break;

    case 1:
#ifdef ZFP_ARRAY1_H
      switch (type) {
        case zfp_type_double:
          arr = new zfp::array1d(n[0], rate);
          break;

        case zfp_type_float:
          arr = new zfp::array1f(n[0], rate);
          break;

        default:
          errMsg = "ZFP compressed arrays do not yet support scalar types beyond floats and doubles.";
          break;
      }
#else
      errMsg = "Header files for 1 dimensional ZFP compressed arrays were not included.";
#endif
      break;

    default:
      errMsg = "ZFP compressed arrays do not yet support dimensionalities beyond 1, 2, and 3.";
      break;
  }

  if (!errMsg.empty()) {
    throw zfp::array::header_exception(errMsg);
  }

  memcpy(arr->compressed_data(), buffer, bufferSizeBytes);

  return arr;
}

#endif
