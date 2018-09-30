// common constructor, destructor
static CFP_ARRAY_TYPE *
_t1(CFP_ARRAY_TYPE, ctor_default)()
{
  return reinterpret_cast<CFP_ARRAY_TYPE *>(new ZFP_ARRAY_TYPE());
}

static CFP_ARRAY_TYPE *
_t1(CFP_ARRAY_TYPE, ctor_copy)(const CFP_ARRAY_TYPE * src)
{
  return reinterpret_cast<CFP_ARRAY_TYPE *>(
    new ZFP_ARRAY_TYPE(*reinterpret_cast<const ZFP_ARRAY_TYPE *>(src))
  );
}

static void
_t1(CFP_ARRAY_TYPE, dtor)(CFP_ARRAY_TYPE * self)
{
  delete reinterpret_cast<ZFP_ARRAY_TYPE *>(self);
}

// functions defined in zfparray.h (base class)
static double
_t1(CFP_ARRAY_TYPE, rate)(const CFP_ARRAY_TYPE * self)
{
  return reinterpret_cast<const ZFP_ARRAY_TYPE *>(self)->rate();
}

static double
_t1(CFP_ARRAY_TYPE, set_rate)(CFP_ARRAY_TYPE * self, double rate)
{
  return reinterpret_cast<ZFP_ARRAY_TYPE *>(self)->set_rate(rate);
}

static size_t
_t1(CFP_ARRAY_TYPE, compressed_size)(const CFP_ARRAY_TYPE * self)
{
  return reinterpret_cast<const ZFP_ARRAY_TYPE *>(self)->compressed_size();
}

static uchar*
_t1(CFP_ARRAY_TYPE, compressed_data)(const CFP_ARRAY_TYPE * self)
{
  return reinterpret_cast<const ZFP_ARRAY_TYPE *>(self)->compressed_data();
}

static void
_t1(CFP_ARRAY_TYPE, deep_copy)(CFP_ARRAY_TYPE * self, const CFP_ARRAY_TYPE * src)
{
  *reinterpret_cast<ZFP_ARRAY_TYPE *>(self) = *reinterpret_cast<const ZFP_ARRAY_TYPE *>(src);
}

// functions defined in subclasses
static size_t
_t1(CFP_ARRAY_TYPE, size)(const CFP_ARRAY_TYPE * self)
{
  return reinterpret_cast<const ZFP_ARRAY_TYPE *>(self)->size();
}

static size_t
_t1(CFP_ARRAY_TYPE, cache_size)(const CFP_ARRAY_TYPE * self)
{
  return reinterpret_cast<const ZFP_ARRAY_TYPE *>(self)->cache_size();
}

static void
_t1(CFP_ARRAY_TYPE, set_cache_size)(CFP_ARRAY_TYPE * self, size_t csize)
{
  reinterpret_cast<ZFP_ARRAY_TYPE *>(self)->set_cache_size(csize);
}

static void
_t1(CFP_ARRAY_TYPE, clear_cache)(const CFP_ARRAY_TYPE * self)
{
  reinterpret_cast<const ZFP_ARRAY_TYPE *>(self)->clear_cache();
}

static void
_t1(CFP_ARRAY_TYPE, flush_cache)(const CFP_ARRAY_TYPE * self)
{
  reinterpret_cast<const ZFP_ARRAY_TYPE *>(self)->flush_cache();
}

static void
_t1(CFP_ARRAY_TYPE, get_array)(const CFP_ARRAY_TYPE * self, ZFP_SCALAR_TYPE * p)
{
  reinterpret_cast<const ZFP_ARRAY_TYPE *>(self)->get(p);
}

static void
_t1(CFP_ARRAY_TYPE, set_array)(CFP_ARRAY_TYPE * self, const ZFP_SCALAR_TYPE * p)
{
  reinterpret_cast<ZFP_ARRAY_TYPE *>(self)->set(p);
}

static ZFP_SCALAR_TYPE
_t1(CFP_ARRAY_TYPE, get_flat)(const CFP_ARRAY_TYPE * self, uint i)
{
  return reinterpret_cast<const ZFP_ARRAY_TYPE *>(self)->operator[](i);
}

static void
_t1(CFP_ARRAY_TYPE, set_flat)(CFP_ARRAY_TYPE * self, uint i, ZFP_SCALAR_TYPE val)
{
  reinterpret_cast<ZFP_ARRAY_TYPE *>(self)->operator[](i) = val;
}
