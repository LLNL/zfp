// common constructor, destructor
static CFP_ARRAY_TYPE
_t1(CFP_ARRAY_TYPE, ctor_default)()
{
  CFP_ARRAY_TYPE a;
  a.object = new ZFP_ARRAY_TYPE();
  return a;
}

static CFP_ARRAY_TYPE
_t1(CFP_ARRAY_TYPE, ctor_copy)(CFP_ARRAY_TYPE src)
{
  CFP_ARRAY_TYPE a;
  a.object = new ZFP_ARRAY_TYPE(*static_cast<const ZFP_ARRAY_TYPE *>(src.object));
  return a;
}

static CFP_ARRAY_TYPE
_t1(CFP_ARRAY_TYPE, ctor_header)(CFP_HEADER_TYPE h, const void* buffer, size_t buffer_size_bytes)
{
  CFP_ARRAY_TYPE a;
  a.object = new ZFP_ARRAY_TYPE(*static_cast<zfp::array::header*>(h.object), buffer, buffer_size_bytes);
  return a;
}

static void
_t1(CFP_ARRAY_TYPE, dtor)(CFP_ARRAY_TYPE self)
{
  delete static_cast<ZFP_ARRAY_TYPE*>(self.object);
}

// functions defined in zfparray.h (base class)
static double
_t1(CFP_ARRAY_TYPE, rate)(CFP_ARRAY_TYPE self)
{
  return static_cast<const ZFP_ARRAY_TYPE*>(self.object)->rate();
}

static double
_t1(CFP_ARRAY_TYPE, set_rate)(CFP_ARRAY_TYPE self, double rate)
{
  return static_cast<ZFP_ARRAY_TYPE*>(self.object)->set_rate(rate);
}

static size_t
_t1(CFP_ARRAY_TYPE, size_bytes)(CFP_ARRAY_TYPE self, uint mask)
{
  return static_cast<const ZFP_ARRAY_TYPE*>(self.object)->size_bytes(mask);
}

static size_t
_t1(CFP_ARRAY_TYPE, compressed_size)(CFP_ARRAY_TYPE self)
{
  return static_cast<const ZFP_ARRAY_TYPE*>(self.object)->compressed_size();
}

static void*
_t1(CFP_ARRAY_TYPE, compressed_data)(CFP_ARRAY_TYPE self)
{
  return static_cast<const ZFP_ARRAY_TYPE*>(self.object)->compressed_data();
}

static void
_t1(CFP_ARRAY_TYPE, deep_copy)(CFP_ARRAY_TYPE self, const CFP_ARRAY_TYPE src)
{
  *static_cast<ZFP_ARRAY_TYPE*>(self.object) = *static_cast<const ZFP_ARRAY_TYPE*>(src.object);
}

// functions defined in subclasses
static size_t
_t1(CFP_ARRAY_TYPE, size)(CFP_ARRAY_TYPE self)
{
  return static_cast<const ZFP_ARRAY_TYPE*>(self.object)->size();
}

static size_t
_t1(CFP_ARRAY_TYPE, cache_size)(CFP_ARRAY_TYPE self)
{
  return static_cast<const ZFP_ARRAY_TYPE*>(self.object)->cache_size();
}

static void
_t1(CFP_ARRAY_TYPE, set_cache_size)(CFP_ARRAY_TYPE self, size_t bytes)
{
  static_cast<ZFP_ARRAY_TYPE*>(self.object)->set_cache_size(bytes);
}

static void
_t1(CFP_ARRAY_TYPE, clear_cache)(CFP_ARRAY_TYPE self)
{
  static_cast<const ZFP_ARRAY_TYPE*>(self.object)->clear_cache();
}

static void
_t1(CFP_ARRAY_TYPE, flush_cache)(CFP_ARRAY_TYPE self)
{
  static_cast<const ZFP_ARRAY_TYPE*>(self.object)->flush_cache();
}

static void
_t1(CFP_ARRAY_TYPE, get_array)(CFP_ARRAY_TYPE self, ZFP_SCALAR_TYPE * p)
{
  static_cast<const ZFP_ARRAY_TYPE*>(self.object)->get(p);
}

static void
_t1(CFP_ARRAY_TYPE, set_array)(CFP_ARRAY_TYPE self, const ZFP_SCALAR_TYPE * p)
{
  static_cast<ZFP_ARRAY_TYPE*>(self.object)->set(p);
}

static ZFP_SCALAR_TYPE
_t1(CFP_ARRAY_TYPE, get_flat)(CFP_ARRAY_TYPE self, size_t i)
{
  return static_cast<const ZFP_ARRAY_TYPE*>(self.object)->operator[](i);
}

static void
_t1(CFP_ARRAY_TYPE, set_flat)(CFP_ARRAY_TYPE self, size_t i, ZFP_SCALAR_TYPE val)
{
  static_cast<ZFP_ARRAY_TYPE*>(self.object)->operator[](i) = val;
}

static CFP_PTR_TYPE
_t2(CFP_ARRAY_TYPE, CFP_REF_TYPE, ptr)(CFP_REF_TYPE self)
{
  CFP_PTR_TYPE p;
  p.reference = self;
  return p;
}

static CFP_REF_TYPE
_t2(CFP_ARRAY_TYPE, CFP_PTR_TYPE, ref)(CFP_PTR_TYPE self)
{
  return self.reference;
}
