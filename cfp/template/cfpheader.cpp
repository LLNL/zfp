static CFP_HEADER_TYPE
_t1(CFP_HEADER_TYPE, ctor_buffer)(const void* data, size_t bytes)
{
  CFP_HEADER_TYPE h;
  h.object = 0;

  try {
    // construct generic header and query array type
    header hdr(data, bytes);
    uint dims = hdr.dimensionality();
    zfp_type scalar_type = hdr.scalar_type();
    // construct array-specific header
    switch (dims) {
      case 1:
        if (scalar_type == zfp_type_float)
          h.object = new zfp::array1f::header(data, bytes);
        else if (scalar_type == zfp_type_double)
          h.object = new zfp::array1d::header(data, bytes);
        break;
      case 2:
        if (scalar_type == zfp_type_float)
          h.object = new zfp::array2f::header(data, bytes);
        else if (scalar_type == zfp_type_double)
          h.object = new zfp::array2d::header(data, bytes);
        break;
      case 3:
        if (scalar_type == zfp_type_float)
          h.object = new zfp::array3f::header(data, bytes);
        else if (scalar_type == zfp_type_double)
          h.object = new zfp::array3d::header(data, bytes);
        break;
      case 4:
        if (scalar_type == zfp_type_float)
          h.object = new zfp::array4f::header(data, bytes);
        else if (scalar_type == zfp_type_double)
          h.object = new zfp::array4d::header(data, bytes);
        break;
    }
  }
  catch (...) {}
  return h;
}

static CFP_HEADER_TYPE
_t1(CFP_HEADER_TYPE, ctor_array1f)(cfp_array1f a)
{
  CFP_HEADER_TYPE h;
  h.object = new zfp::array1f::header(*static_cast<zfp::array1f*>(a.object));
  return h;
}

static CFP_HEADER_TYPE
_t1(CFP_HEADER_TYPE, ctor_array1d)(cfp_array1d a)
{
  CFP_HEADER_TYPE h;
  h.object = new zfp::array1d::header(*static_cast<zfp::array1d*>(a.object));
  return h;
}

static CFP_HEADER_TYPE
_t1(CFP_HEADER_TYPE, ctor_array2f)(cfp_array2f a)
{
  CFP_HEADER_TYPE h;
  h.object = new zfp::array2f::header(*static_cast<zfp::array2f*>(a.object));
  return h;
}

static CFP_HEADER_TYPE
_t1(CFP_HEADER_TYPE, ctor_array2d)(cfp_array2d a)
{
  CFP_HEADER_TYPE h;
  h.object = new zfp::array2d::header(*static_cast<zfp::array2d*>(a.object));
  return h;
}

static CFP_HEADER_TYPE
_t1(CFP_HEADER_TYPE, ctor_array3f)(cfp_array3f a)
{
  CFP_HEADER_TYPE h;
  h.object = new zfp::array3f::header(*static_cast<zfp::array3f*>(a.object));
  return h;
}

static CFP_HEADER_TYPE
_t1(CFP_HEADER_TYPE, ctor_array3d)(cfp_array3d a)
{
  CFP_HEADER_TYPE h;
  h.object = new zfp::array3d::header(*static_cast<zfp::array3d*>(a.object));
  return h;
}

static CFP_HEADER_TYPE
_t1(CFP_HEADER_TYPE, ctor_array4f)(cfp_array4f a)
{
  CFP_HEADER_TYPE h;
  h.object = new zfp::array4f::header(*static_cast<zfp::array4f*>(a.object));
  return h;
}

static CFP_HEADER_TYPE
_t1(CFP_HEADER_TYPE, ctor_array4d)(cfp_array4d a)
{
  CFP_HEADER_TYPE h;
  h.object = new zfp::array4d::header(*static_cast<zfp::array4d*>(a.object));
  return h;
}

static void
_t1(CFP_HEADER_TYPE, dtor)(CFP_HEADER_TYPE self)
{
  delete static_cast<ZFP_HEADER_TYPE*>(self.object);
}

static zfp_type
_t1(CFP_HEADER_TYPE, scalar_type)(CFP_HEADER_TYPE self)
{
  return static_cast<const ZFP_HEADER_TYPE*>(self.object)->scalar_type();
}

static uint
_t1(CFP_HEADER_TYPE, dimensionality)(CFP_HEADER_TYPE self)
{
  return static_cast<const ZFP_HEADER_TYPE*>(self.object)->dimensionality();
}

static size_t
_t1(CFP_HEADER_TYPE, size_x)(CFP_HEADER_TYPE self)
{
  return static_cast<const ZFP_HEADER_TYPE*>(self.object)->size_x();
}

static size_t
_t1(CFP_HEADER_TYPE, size_y)(CFP_HEADER_TYPE self)
{
  return static_cast<const ZFP_HEADER_TYPE*>(self.object)->size_y();
}

static size_t
_t1(CFP_HEADER_TYPE, size_z)(CFP_HEADER_TYPE self)
{
  return static_cast<const ZFP_HEADER_TYPE*>(self.object)->size_z();
}

static size_t
_t1(CFP_HEADER_TYPE, size_w)(CFP_HEADER_TYPE self)
{
  return static_cast<const ZFP_HEADER_TYPE*>(self.object)->size_w();
}

static double
_t1(CFP_HEADER_TYPE, rate)(CFP_HEADER_TYPE self)
{
  return static_cast<const ZFP_HEADER_TYPE*>(self.object)->rate();
}

static const void*
_t1(CFP_HEADER_TYPE, data)(CFP_HEADER_TYPE self)
{
  return static_cast<const ZFP_HEADER_TYPE*>(self.object)->data();
}

static size_t
_t1(CFP_HEADER_TYPE, size_bytes)(CFP_HEADER_TYPE self, uint mask)
{
  return static_cast<const ZFP_HEADER_TYPE*>(self.object)->size_bytes(mask);
}
