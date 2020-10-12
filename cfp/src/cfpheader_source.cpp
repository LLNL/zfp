static CFP_HEADER_TYPE
_t1(CFP_HEADER_TYPE, ctor_buffer)(uint dim, zfp_type scalar_type, const void* bytes, size_t n)
{
  CFP_HEADER_TYPE h;
  h.object = NULL;
  switch (dim) {
    case 1:
      if (scalar_type == zfp_type_float)
        h.object = static_cast<void*>(new zfp::array1f::header(bytes, n));
      else if (scalar_type == zfp_type_double) 
        h.object = static_cast<void*>(new zfp::array1d::header(bytes, n));
      break;
    case 2:
      if (scalar_type == zfp_type_float)
        h.object = static_cast<void*>(new zfp::array2f::header(bytes, n));
      else if (scalar_type == zfp_type_double) 
        h.object = static_cast<void*>(new zfp::array2d::header(bytes, n));
      break;
    case 3:
      if (scalar_type == zfp_type_float)
        h.object = static_cast<void*>(new zfp::array3f::header(bytes, n));
      else if (scalar_type == zfp_type_double) 
        h.object = static_cast<void*>(new zfp::array3d::header(bytes, n));
      break;
    case 4:
      if (scalar_type == zfp_type_float)
        h.object = static_cast<void*>(new zfp::array4f::header(bytes, n));
      else if (scalar_type == zfp_type_double) 
        h.object = static_cast<void*>(new zfp::array4d::header(bytes, n));
      break;
  }
  return h;
}

static CFP_HEADER_TYPE
_t1(CFP_HEADER_TYPE, ctor_array1f)(cfp_array1f a)
{
  CFP_HEADER_TYPE h;
  h.object = static_cast<void*>(
    new zfp::array1<float>::header(*static_cast<zfp::array1<float>*>(a.object))
  );
  return h;
}

static CFP_HEADER_TYPE
_t1(CFP_HEADER_TYPE, ctor_array1d)(cfp_array1d a)
{
  CFP_HEADER_TYPE h;
  h.object = static_cast<void*>(
    new zfp::array1<double>::header(*static_cast<zfp::array1<double>*>(a.object))
  );
  return h;
}

static CFP_HEADER_TYPE
_t1(CFP_HEADER_TYPE, ctor_array2f)(cfp_array2f a)
{
  CFP_HEADER_TYPE h;
  h.object = static_cast<void*>(
    new zfp::array2<float>::header(*static_cast<zfp::array2<float>*>(a.object))
  );
  return h;
}

static CFP_HEADER_TYPE
_t1(CFP_HEADER_TYPE, ctor_array2d)(cfp_array2d a)
{
  CFP_HEADER_TYPE h;
  h.object = static_cast<void*>(
    new zfp::array2<double>::header(*static_cast<zfp::array2<double>*>(a.object))
  );
  return h;
}

static CFP_HEADER_TYPE
_t1(CFP_HEADER_TYPE, ctor_array3f)(cfp_array3f a)
{
  CFP_HEADER_TYPE h;
  h.object = static_cast<void*>(
    new zfp::array3<float>::header(*static_cast<zfp::array3<float>*>(a.object))
  );
  return h;
}

static CFP_HEADER_TYPE
_t1(CFP_HEADER_TYPE, ctor_array3d)(cfp_array3d a)
{
  CFP_HEADER_TYPE h;
  h.object = static_cast<void*>(
    new zfp::array3<double>::header(*static_cast<zfp::array3<double>*>(a.object))
  );
  return h;
}

static CFP_HEADER_TYPE
_t1(CFP_HEADER_TYPE, ctor_array4f)(cfp_array4f a)
{
  CFP_HEADER_TYPE h;
  h.object = static_cast<void*>(
    new zfp::array4<float>::header(*static_cast<zfp::array4<float>*>(a.object))
  );
  return h;
}

static CFP_HEADER_TYPE
_t1(CFP_HEADER_TYPE, ctor_array4d)(cfp_array4d a)
{
  CFP_HEADER_TYPE h;
  h.object = static_cast<void*>(
    new zfp::array4<double>::header(*static_cast<zfp::array4<double>*>(a.object))
  );
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
  return static_cast<ZFP_HEADER_TYPE*>(self.object)->scalar_type();
}

static uint
_t1(CFP_HEADER_TYPE, dimensionality)(CFP_HEADER_TYPE self)
{
  return static_cast<ZFP_HEADER_TYPE*>(self.object)->dimensionality();
}

static size_t
_t1(CFP_HEADER_TYPE, size)(CFP_HEADER_TYPE self)
{
  return static_cast<ZFP_HEADER_TYPE*>(self.object)->size();
}

static size_t
_t1(CFP_HEADER_TYPE, size_x)(CFP_HEADER_TYPE self)
{
  return static_cast<ZFP_HEADER_TYPE*>(self.object)->size_x();
}

static size_t
_t1(CFP_HEADER_TYPE, size_y)(CFP_HEADER_TYPE self)
{
  return static_cast<ZFP_HEADER_TYPE*>(self.object)->size_y();
}

static size_t
_t1(CFP_HEADER_TYPE, size_z)(CFP_HEADER_TYPE self)
{
  return static_cast<ZFP_HEADER_TYPE*>(self.object)->size_z();
}

static size_t
_t1(CFP_HEADER_TYPE, size_w)(CFP_HEADER_TYPE self)
{
  return static_cast<ZFP_HEADER_TYPE*>(self.object)->size_w();
}

static double
_t1(CFP_HEADER_TYPE, rate)(CFP_HEADER_TYPE self)
{
  return static_cast<ZFP_HEADER_TYPE*>(self.object)->rate();
}

static const void*
_t1(CFP_HEADER_TYPE, data)(CFP_HEADER_TYPE self)
{
  return static_cast<ZFP_HEADER_TYPE*>(self.object)->data();
}
