// abstract base class for array header
class header {
public:
  // default constructor
  header() :
    type(zfp_type_none),
    nx(0), ny(0), nz(0), nw(0)
  {}

  // constructor
  header(const zfp::array& a) :
    type(a.type),
    nx(a.nx), ny(a.ny), nz(a.nz), nw(a.nw)
  {}

  // destructor
  virtual ~header() {}

  // array scalar type
  zfp_type scalar_type() const { return type; }

  // array dimensionality
  uint dimensionality() const { return nw ? 4 : nz ? 3 : ny ? 2 : nx ? 1 : 0; }

  // array dimensions
  size_t size_x() const { return nx; }
  size_t size_y() const { return ny; }
  size_t size_z() const { return nz; }
  size_t size_w() const { return nw; }

  // rate in bits per value
  virtual double rate() const = 0;

  // header payload: data pointer and byte size
  virtual const void* data() const = 0;
  virtual size_t size_bytes(uint mask = ZFP_DATA_HEADER) const = 0;

protected:
  zfp_type type;         // array scalar type
  size_t nx, ny, nz, nw; // array dimensions
};
