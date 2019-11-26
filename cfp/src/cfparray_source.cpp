// common constructor, destructor
static CFP_ARRAY_TYPE
_t1(CFP_ARRAY_TYPE, ctor_default)()
{
  return (CFP_ARRAY_TYPE){reinterpret_cast<void*>(new ZFP_ARRAY_TYPE())};
}

static CFP_ARRAY_TYPE
_t1(CFP_ARRAY_TYPE, ctor_copy)(CFP_ARRAY_TYPE src)
{
  return (CFP_ARRAY_TYPE){
    reinterpret_cast<void*>(new ZFP_ARRAY_TYPE(*reinterpret_cast<const ZFP_ARRAY_TYPE *>(src.object)))
  };
}

static void
_t1(CFP_ARRAY_TYPE, dtor)(CFP_ARRAY_TYPE self)
{
  delete reinterpret_cast<ZFP_ARRAY_TYPE*>(self.object);
}

// functions defined in zfparray.h (base class)
static double
_t1(CFP_ARRAY_TYPE, rate)(CFP_ARRAY_TYPE self)
{
  return reinterpret_cast<const ZFP_ARRAY_TYPE*>(self.object)->rate();
}

static double
_t1(CFP_ARRAY_TYPE, set_rate)(CFP_ARRAY_TYPE self, double rate)
{
  return reinterpret_cast<ZFP_ARRAY_TYPE*>(self.object)->set_rate(rate);
}

static size_t
_t1(CFP_ARRAY_TYPE, compressed_size)(CFP_ARRAY_TYPE self)
{
  return reinterpret_cast<const ZFP_ARRAY_TYPE*>(self.object)->compressed_size();
}

static uchar*
_t1(CFP_ARRAY_TYPE, compressed_data)(CFP_ARRAY_TYPE self)
{
  return reinterpret_cast<const ZFP_ARRAY_TYPE*>(self.object)->compressed_data();
}

static void
_t1(CFP_ARRAY_TYPE, deep_copy)(CFP_ARRAY_TYPE self, const CFP_ARRAY_TYPE src)
{
  *reinterpret_cast<ZFP_ARRAY_TYPE*>(self.object) = *reinterpret_cast<const ZFP_ARRAY_TYPE*>(src.object);
}

// functions defined in subclasses
static size_t
_t1(CFP_ARRAY_TYPE, size)(CFP_ARRAY_TYPE self)
{
  return reinterpret_cast<const ZFP_ARRAY_TYPE*>(self.object)->size();
}

static size_t
_t1(CFP_ARRAY_TYPE, cache_size)(CFP_ARRAY_TYPE self)
{
  return reinterpret_cast<const ZFP_ARRAY_TYPE*>(self.object)->cache_size();
}

static void
_t1(CFP_ARRAY_TYPE, set_cache_size)(CFP_ARRAY_TYPE self, size_t csize)
{
  reinterpret_cast<ZFP_ARRAY_TYPE*>(self.object)->set_cache_size(csize);
}

static void
_t1(CFP_ARRAY_TYPE, clear_cache)(CFP_ARRAY_TYPE self)
{
  reinterpret_cast<const ZFP_ARRAY_TYPE*>(self.object)->clear_cache();
}

static void
_t1(CFP_ARRAY_TYPE, flush_cache)(CFP_ARRAY_TYPE self)
{
  reinterpret_cast<const ZFP_ARRAY_TYPE*>(self.object)->flush_cache();
}

static void
_t1(CFP_ARRAY_TYPE, get_array)(CFP_ARRAY_TYPE self, ZFP_SCALAR_TYPE * p)
{
  reinterpret_cast<const ZFP_ARRAY_TYPE*>(self.object)->get(p);
}

static void
_t1(CFP_ARRAY_TYPE, set_array)(CFP_ARRAY_TYPE self, const ZFP_SCALAR_TYPE * p)
{
  reinterpret_cast<ZFP_ARRAY_TYPE*>(self.object)->set(p);
}

static ZFP_SCALAR_TYPE
_t1(CFP_ARRAY_TYPE, get_flat)(CFP_ARRAY_TYPE self, uint i)
{
  return reinterpret_cast<const ZFP_ARRAY_TYPE*>(self.object)->operator[](i);
}

static void
_t1(CFP_ARRAY_TYPE, set_flat)(CFP_ARRAY_TYPE self, uint i, ZFP_SCALAR_TYPE val)
{
  reinterpret_cast<ZFP_ARRAY_TYPE*>(self.object)->operator[](i) = val;
}
