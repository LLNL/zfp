// zfp::codec::zfp_base::header
class header : public zfp::array::header {
public:
  // serialization: construct header from array
  header(const zfp::array& a) :
    zfp::array::header(a),
    bit_rate(a.rate())
  {
    std::string error;

    // set up zfp stream and field for generating header
    bitstream* stream = stream_open(buffer, sizeof(buffer));
    zfp_stream* zfp = zfp_stream_open(stream);
    bit_rate = zfp_stream_set_rate(zfp, bit_rate, type, dimensionality(), zfp_true);
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
        case 4:
          field = zfp_field_4d(0, type, nx, ny, nz, nw);
          break;
        default:
          error = "zfp serialization supports only 1D, 2D, 3D, and 4D arrays";
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

    if (!error.empty())
      throw zfp::exception(error);
  }

  // deserialization: construct header from memory buffer of optional size
  header(const void* data, size_t bytes = 0) :
    bit_rate(0)
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
      else {
        // success; initialize fields
        type = field.type;
        nx = field.nx;
        ny = field.ny;
        nz = field.nz;
        nw = field.nw;
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

  // rate in bits per value
  double rate() const { return bit_rate; }

  // header data
  const void* data() const { return buffer; }

  // header byte size
  size_t size_bytes(uint mask = ZFP_DATA_HEADER) const
  {
    size_t size = 0;
    if (mask & ZFP_DATA_META)
      size += sizeof(*this) - byte_size;
    if (mask & ZFP_DATA_HEADER)
      size += byte_size;
    return size;
  }

protected:
  // header size measured in bits, bytes, and 64-bit words
  static const size_t bit_size = ZFP_MAGIC_BITS + ZFP_META_BITS + ZFP_MODE_SHORT_BITS;
  static const size_t byte_size = (bit_size + CHAR_BIT - 1) / CHAR_BIT;
  static const size_t word_size = (byte_size + sizeof(uint64) - 1) / sizeof(uint64);

  using zfp::array::header::type;
  using zfp::array::header::nx;
  using zfp::array::header::ny;
  using zfp::array::header::nz;
  using zfp::array::header::nw;

  double bit_rate;          // array rate in bits per value
  uint64 buffer[word_size]; // header data
};
