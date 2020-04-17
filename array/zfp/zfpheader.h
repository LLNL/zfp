// zfp::codec_base::header
class header : public zfp::header {
public:
  // serialization: construct header from 1D array
  header(zfp_type type, size_t nx, double rate)
  {
    construct(type, nx, 0, 0, rate);
  }

  // serialization: construct header from 2D array
  header(zfp_type type, size_t nx, size_t ny, double rate)
  {
    construct(type, nx, ny, 0, rate);
  }

  // serialization: construct header from 3D array
  header(zfp_type type, size_t nx, size_t ny, size_t nz, double rate)
  {
    construct(type, nx, ny, nz, rate);
  }

  // deserialization: construct header from memory buffer of optional size
  header(const void* data, size_t bytes = 0) :
    bit_rate(0),
    type(zfp_type_none),
    nx(0), ny(0), nz(0)
  {
    std::string error;

    // ensure byte size matches
    if (bytes && bytes != byte_size)
      error = "zfp header length does not match expectations";
    else {
      // copy and parse header
      std::fill(buffer, buffer + word_size, 0);
      std::memcpy(buffer, data, byte_size);
      bitstream* stream = stream_open(buffer, sizeof(buffer));
      zfp_stream* zfp = zfp_stream_open(stream);
      zfp_field field;
      size_t bits = zfp_read_header(zfp, &field, ZFP_HEADER_FULL);
      if (!bits)
        error = "zfp header is corrupt";
      else if (bits != bit_size)
        error = "zfp deserialization supports only short headers";
      else if (zfp_stream_compression_mode(zfp) != zfp_mode_fixed_rate)
        error = "zfp deserialization supports only fixed-rate mode";
      else if (field.nw)
        error = "zfp deserialization supports only 1D, 2D, and 3D arrays";
      else {
        // success; initialize fields
        type = field.type;
        nx = field.nx;
        ny = field.ny;
        nz = field.nz;
        bit_rate = double(zfp->maxbits) / (1u << (2 * dimensionality()));
      }
      zfp_stream_close(zfp);
      stream_close(stream);
    }

    // throw exception upon error
    if (!error.empty())
      throw zfp::exception(error);
  }

  virtual ~header() {}

  // scalar type
  zfp_type scalar_type() const { return type; }

  // array dimensions
  size_t size_x() const { return nx; }
  size_t size_y() const { return ny; }
  size_t size_z() const { return nz; }

  // rate in bits per value
  double rate() const { return bit_rate; }

  // header data and byte size
  const void* data() const { return buffer; }
  size_t size() const { return byte_size; }

protected:
  // construct header from array metadata
  void construct(zfp_type type, size_t nx, size_t ny, size_t nz, double rate)
  {
    std::string error;

    // set scalar type
    switch (type) {
      case zfp_type_float:
      case zfp_type_double:
        this->type = type;
        break;
      default:
        error = "zfp serialization supports only float and double";
        break;
    }

    if (error.empty()) {
      // set dimensions
      this->nx = nx;
      this->ny = ny;
      this->nz = nz;

      // set up zfp stream and field for generating header
      bitstream* stream = stream_open(buffer, sizeof(buffer));
      zfp_stream* zfp = zfp_stream_open(stream);
      bit_rate = zfp_stream_set_rate(zfp, rate, type, dimensionality(), zfp_true);
      if (zfp_stream_mode(zfp) > ZFP_MODE_SHORT_MAX)
        error = "zfp serialization supports only short headers";
      else {
        // set up field
        zfp_field* field = 0;
        switch (dimensionality()) {
          case 1:
            field = zfp_field_1d(0, type, nx);
            break;
          case 2:
            field = zfp_field_2d(0, type, nx, ny);
            break;
          case 3:
            field = zfp_field_3d(0, type, nx, ny, nz);
            break;
          default:
            error = "zfp serialization supports only 1D, 2D, and 3D arrays";
            break;
        }

        if (field) {
          // write header to buffer
          size_t bits = zfp_write_header(zfp, field, ZFP_HEADER_FULL);
          if (bits != bit_size)
            error = "zfp header length does not match expected length";
          zfp_stream_flush(zfp);
          zfp_field_free(field);
        }
      }

      zfp_stream_close(zfp);
      stream_close(stream);
    }

    if (!error.empty())
      throw zfp::exception(error);
  }

  // header size measured in bits, bytes, and 64-bit words
  static const size_t bit_size = ZFP_MAGIC_BITS + ZFP_META_BITS + ZFP_MODE_SHORT_BITS;
  static const size_t byte_size = (bit_size + CHAR_BIT - 1) / CHAR_BIT;
  static const size_t word_size = (byte_size + sizeof(uint64) - 1) / sizeof(uint64);

  double bit_rate;          // array rate in bits per value
  zfp_type type;            // array scalar type
  size_t nx, ny, nz;        // array dimensions
  uint64 buffer[word_size]; // header data
};
