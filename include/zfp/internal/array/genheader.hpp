// zfp::codec::generic_base::header
class header : public zfp::array::header {
public:
  // serialization: construct header from array
  header(const zfp::array& a) :
    zfp::array::header(a),
    bit_rate(static_cast<size_t>(a.rate()))
  {
    buffer[0] = magic;
    buffer[1] = 0; // TODO: codec identifier (dimensionality, internal type)
    buffer[2] = static_cast<uint64>(bit_rate);
    buffer[3] = static_cast<uint64>(type);
    buffer[4] = static_cast<uint64>(nx);
    buffer[5] = static_cast<uint64>(ny);
    buffer[6] = static_cast<uint64>(nz);
    buffer[7] = static_cast<uint64>(nw);
  }

  // deserialization: construct header from memory buffer of optional size
  header(const void* data, size_t bytes = 0) :
    bit_rate(0)
  {
    // ensure byte size matches
    if (bytes && bytes != byte_size)
      throw zfp::exception("zfp generic header length does not match expectations");
    else {
      // copy and parse header
      std::memcpy(buffer, data, byte_size);
      if (buffer[0] != magic)
        throw zfp::exception("zfp generic header is corrupt");
      bit_rate = static_cast<size_t>(buffer[2]);
      type = static_cast<zfp_type>(buffer[3]);
      nx = static_cast<size_t>(buffer[4]);
      ny = static_cast<size_t>(buffer[5]);
      nz = static_cast<size_t>(buffer[6]);
      nw = static_cast<size_t>(buffer[7]);
    }
  }

  virtual ~header() {}

  // rate in bits per value
  double rate() const { return static_cast<double>(bit_rate); }

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
  // magic word
  static const uint64 magic = UINT64C(0x000000008570667a);

  // header size measured in bits, bytes, and 64-bit words
  static const size_t word_size = 8;
  static const size_t byte_size = word_size * sizeof(uint64);
  static const size_t bit_size = byte_size * CHAR_BIT;

  using zfp::array::header::type;
  using zfp::array::header::nx;
  using zfp::array::header::ny;
  using zfp::array::header::nz;
  using zfp::array::header::nw;

  size_t bit_rate;          // array rate in bits per value
  uint64 buffer[word_size]; // header data
};
