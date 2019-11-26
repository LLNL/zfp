static CFP_ARRAY_TYPE
_t1(CFP_ARRAY_TYPE, ctor)(uint n, double rate, const ZFP_SCALAR_TYPE * p, size_t csize)
{
  CFP_ARRAY_TYPE a;
  a.object = reinterpret_cast<void*>(new ZFP_ARRAY_TYPE(n, rate, p, csize));
  return a;
}

static void
_t1(CFP_ARRAY_TYPE, resize)(CFP_ARRAY_TYPE self, uint n, int clear)
{
  reinterpret_cast<ZFP_ARRAY_TYPE*>(self.object)->resize(n, clear);
}

static ZFP_SCALAR_TYPE
_t1(CFP_ARRAY_TYPE, get)(CFP_ARRAY_TYPE self, uint i)
{
  return reinterpret_cast<const ZFP_ARRAY_TYPE*>(self.object)->operator()(i);
}

static void
_t1(CFP_ARRAY_TYPE, set)(CFP_ARRAY_TYPE self, uint i, ZFP_SCALAR_TYPE val)
{
  reinterpret_cast<ZFP_ARRAY_TYPE*>(self.object)->operator()(i) = val;
}

static CFP_REF_TYPE
_t1(CFP_ARRAY_TYPE, get_ref)(CFP_ARRAY_TYPE self, uint i)
{
  CFP_REF_TYPE r;
  r.i = i;
  r.array = self;
  return r;
}

/* References */
static ZFP_SCALAR_TYPE
_t2(CFP_ARRAY_TYPE, CFP_REF_TYPE, get)(CFP_REF_TYPE self)
{
  return reinterpret_cast<const ZFP_ARRAY_TYPE*>(self.array.object)->operator()(self.i);
}

static void
_t2(CFP_ARRAY_TYPE, CFP_REF_TYPE, set)(CFP_REF_TYPE self, ZFP_SCALAR_TYPE val)
{
  reinterpret_cast<ZFP_ARRAY_TYPE*>(self.array.object)->operator()(self.i) = val;
}

static void
_t2(CFP_ARRAY_TYPE, CFP_REF_TYPE, copy)(CFP_REF_TYPE self, CFP_REF_TYPE src)
{
  reinterpret_cast<ZFP_ARRAY_TYPE*>(self.array.object)->operator()(self.i) =
    reinterpret_cast<ZFP_ARRAY_TYPE*>(src.array.object)->operator()(src.i);
}
