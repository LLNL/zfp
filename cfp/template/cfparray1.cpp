static CFP_ARRAY_TYPE
_t1(CFP_ARRAY_TYPE, ctor)(size_t n, double rate, const ZFP_SCALAR_TYPE* p, size_t cache_size)
{
  CFP_ARRAY_TYPE a;
  a.object = new ZFP_ARRAY_TYPE(n, rate, p, cache_size);
  return a;
}

static void
_t1(CFP_ARRAY_TYPE, resize)(CFP_ARRAY_TYPE self, size_t n, zfp_bool clear)
{
  static_cast<ZFP_ARRAY_TYPE*>(self.object)->resize(n, !!clear);
}

static ZFP_SCALAR_TYPE
_t1(CFP_ARRAY_TYPE, get)(CFP_ARRAY_TYPE self, size_t i)
{
  return static_cast<const ZFP_ARRAY_TYPE*>(self.object)->operator()(i);
}

static void
_t1(CFP_ARRAY_TYPE, set)(CFP_ARRAY_TYPE self, size_t i, ZFP_SCALAR_TYPE val)
{
  static_cast<ZFP_ARRAY_TYPE*>(self.object)->operator()(i) = val;
}

static CFP_REF_TYPE
_t1(CFP_ARRAY_TYPE, ref)(CFP_ARRAY_TYPE self, size_t i)
{
  CFP_REF_TYPE r;
  r.array = self;
  r.x = i;
  return r;
}

static CFP_REF_TYPE
_t1(CFP_ARRAY_TYPE, ref_flat)(CFP_ARRAY_TYPE self, size_t i)
{
  CFP_REF_TYPE r;
  r.array = self;
  r.x = i;
  return r;
}

static CFP_PTR_TYPE
_t1(CFP_ARRAY_TYPE, ptr)(CFP_ARRAY_TYPE self, size_t i)
{
  CFP_PTR_TYPE p;
  p.reference = _t1(CFP_ARRAY_TYPE, ref)(self, i);
  return p;
}

static CFP_PTR_TYPE
_t1(CFP_ARRAY_TYPE, ptr_flat)(CFP_ARRAY_TYPE self, size_t i)
{
  CFP_PTR_TYPE p;
  p.reference = _t1(CFP_ARRAY_TYPE, ref_flat)(self, i);
  return p;
}

static CFP_ITER_TYPE
_t1(CFP_ARRAY_TYPE, begin)(CFP_ARRAY_TYPE self)
{
  CFP_ITER_TYPE it;
  it.array = self;
  it.x = 0;
  return it;
}

static CFP_ITER_TYPE
_t1(CFP_ARRAY_TYPE, end)(CFP_ARRAY_TYPE self)
{
  CFP_ITER_TYPE it;
  it.array = self;
  it.x = static_cast<const ZFP_ARRAY_TYPE*>(self.object)->size_x();
  return it;
}

static ZFP_SCALAR_TYPE
_t2(CFP_ARRAY_TYPE, CFP_REF_TYPE, get)(CFP_REF_TYPE self)
{
  return static_cast<const ZFP_ARRAY_TYPE*>(self.array.object)->operator()(self.x);
}

static void
_t2(CFP_ARRAY_TYPE, CFP_REF_TYPE, set)(CFP_REF_TYPE self, ZFP_SCALAR_TYPE val)
{
  static_cast<ZFP_ARRAY_TYPE*>(self.array.object)->operator()(self.x) = val;
}

static void
_t2(CFP_ARRAY_TYPE, CFP_REF_TYPE, copy)(CFP_REF_TYPE self, CFP_REF_TYPE src)
{
  static_cast<ZFP_ARRAY_TYPE*>(self.array.object)->operator()(self.x) =
    static_cast<const ZFP_ARRAY_TYPE*>(src.array.object)->operator()(src.x);
}

static zfp_bool
_t2(CFP_ARRAY_TYPE, CFP_PTR_TYPE, lt)(CFP_PTR_TYPE lhs, CFP_PTR_TYPE rhs)
{
  return lhs.reference.array.object == rhs.reference.array.object &&
         lhs.reference.x < rhs.reference.x;
}

static zfp_bool
_t2(CFP_ARRAY_TYPE, CFP_PTR_TYPE, gt)(CFP_PTR_TYPE lhs, CFP_PTR_TYPE rhs)
{
  return lhs.reference.array.object == rhs.reference.array.object &&
         lhs.reference.x > rhs.reference.x;
}

static zfp_bool
_t2(CFP_ARRAY_TYPE, CFP_PTR_TYPE, leq)(CFP_PTR_TYPE lhs, CFP_PTR_TYPE rhs)
{
  return lhs.reference.array.object == rhs.reference.array.object &&
         lhs.reference.x <= rhs.reference.x;
}

static zfp_bool
_t2(CFP_ARRAY_TYPE, CFP_PTR_TYPE, geq)(CFP_PTR_TYPE lhs, CFP_PTR_TYPE rhs)
{
  return lhs.reference.array.object == rhs.reference.array.object &&
         lhs.reference.x >= rhs.reference.x;
}

static zfp_bool
_t2(CFP_ARRAY_TYPE, CFP_PTR_TYPE, eq)(CFP_PTR_TYPE lhs, CFP_PTR_TYPE rhs)
{
  return lhs.reference.array.object == rhs.reference.array.object &&
         lhs.reference.x == rhs.reference.x;
}

static zfp_bool
_t2(CFP_ARRAY_TYPE, CFP_PTR_TYPE, neq)(CFP_PTR_TYPE lhs, CFP_PTR_TYPE rhs)
{
  return !_t2(CFP_ARRAY_TYPE, CFP_PTR_TYPE, eq)(lhs, rhs);
}

static ptrdiff_t
_t2(CFP_ARRAY_TYPE, CFP_PTR_TYPE, distance)(CFP_PTR_TYPE first, CFP_PTR_TYPE last)
{
  return last.reference.x - first.reference.x;
}

static CFP_PTR_TYPE
_t2(CFP_ARRAY_TYPE, CFP_PTR_TYPE, next)(CFP_PTR_TYPE p, ptrdiff_t d)
{
  p.reference.x += d;
  return p;
}

static CFP_PTR_TYPE
_t2(CFP_ARRAY_TYPE, CFP_PTR_TYPE, prev)(CFP_PTR_TYPE p, ptrdiff_t d)
{
  p.reference.x -= d;
  return p;
}

static CFP_PTR_TYPE
_t2(CFP_ARRAY_TYPE, CFP_PTR_TYPE, inc)(CFP_PTR_TYPE p)
{
  p.reference.x++;
  return p;
}

static CFP_PTR_TYPE
_t2(CFP_ARRAY_TYPE, CFP_PTR_TYPE, dec)(CFP_PTR_TYPE p)
{
  p.reference.x--;
  return p;
}

static ZFP_SCALAR_TYPE
_t2(CFP_ARRAY_TYPE, CFP_PTR_TYPE, get)(CFP_PTR_TYPE self)
{
  return static_cast<const ZFP_ARRAY_TYPE*>(self.reference.array.object)->operator()(self.reference.x);
}

static ZFP_SCALAR_TYPE
_t2(CFP_ARRAY_TYPE, CFP_PTR_TYPE, get_at)(CFP_PTR_TYPE self, ptrdiff_t d)
{
  self = _t2(CFP_ARRAY_TYPE, CFP_PTR_TYPE, next)(self, d);
  return static_cast<const ZFP_ARRAY_TYPE*>(self.reference.array.object)->operator()(self.reference.x);
}

static void
_t2(CFP_ARRAY_TYPE, CFP_PTR_TYPE, set)(CFP_PTR_TYPE self, ZFP_SCALAR_TYPE val)
{
  static_cast<ZFP_ARRAY_TYPE*>(self.reference.array.object)->operator()(self.reference.x) = val;
}

static void
_t2(CFP_ARRAY_TYPE, CFP_PTR_TYPE, set_at)(CFP_PTR_TYPE self, ptrdiff_t d, ZFP_SCALAR_TYPE val)
{
  self = _t2(CFP_ARRAY_TYPE, CFP_PTR_TYPE, next)(self, d);
  static_cast<ZFP_ARRAY_TYPE*>(self.reference.array.object)->operator()(self.reference.x) = val;
}

static CFP_REF_TYPE
_t2(CFP_ARRAY_TYPE, CFP_PTR_TYPE, ref_at)(CFP_PTR_TYPE self, ptrdiff_t d)
{
  self = _t2(CFP_ARRAY_TYPE, CFP_PTR_TYPE, next)(self, d);
  return self.reference;
}

static zfp_bool
_t2(CFP_ARRAY_TYPE, CFP_ITER_TYPE, lt)(CFP_ITER_TYPE lhs, CFP_ITER_TYPE rhs)
{
  return lhs.array.object == rhs.array.object && lhs.x < rhs.x;
}

static zfp_bool
_t2(CFP_ARRAY_TYPE, CFP_ITER_TYPE, gt)(CFP_ITER_TYPE lhs, CFP_ITER_TYPE rhs)
{
  return lhs.array.object == rhs.array.object && lhs.x > rhs.x;
}

static zfp_bool
_t2(CFP_ARRAY_TYPE, CFP_ITER_TYPE, leq)(CFP_ITER_TYPE lhs, CFP_ITER_TYPE rhs)
{
  return lhs.array.object == rhs.array.object && lhs.x <= rhs.x;
}

static zfp_bool
_t2(CFP_ARRAY_TYPE, CFP_ITER_TYPE, geq)(CFP_ITER_TYPE lhs, CFP_ITER_TYPE rhs)
{
  return lhs.array.object == rhs.array.object && lhs.x >= rhs.x;
}

static zfp_bool
_t2(CFP_ARRAY_TYPE, CFP_ITER_TYPE, eq)(CFP_ITER_TYPE lhs, CFP_ITER_TYPE rhs)
{
  return lhs.array.object == rhs.array.object && lhs.x == rhs.x;
}

static zfp_bool
_t2(CFP_ARRAY_TYPE, CFP_ITER_TYPE, neq)(CFP_ITER_TYPE lhs, CFP_ITER_TYPE rhs)
{
  return !_t2(CFP_ARRAY_TYPE, CFP_ITER_TYPE, eq)(lhs, rhs);
}

static ptrdiff_t
_t2(CFP_ARRAY_TYPE, CFP_ITER_TYPE, distance)(CFP_ITER_TYPE first, CFP_ITER_TYPE last)
{
   return last.x - first.x;
}

static CFP_ITER_TYPE
_t2(CFP_ARRAY_TYPE, CFP_ITER_TYPE, next)(CFP_ITER_TYPE it, ptrdiff_t d)
{
  it.x += d;
  return it;
}

static CFP_ITER_TYPE
_t2(CFP_ARRAY_TYPE, CFP_ITER_TYPE, prev)(CFP_ITER_TYPE it, ptrdiff_t d)
{
  it.x -= d;
  return it;
}

static CFP_ITER_TYPE
_t2(CFP_ARRAY_TYPE, CFP_ITER_TYPE, inc)(CFP_ITER_TYPE it)
{
  it.x++;
  return it;
}

static CFP_ITER_TYPE
_t2(CFP_ARRAY_TYPE, CFP_ITER_TYPE, dec)(CFP_ITER_TYPE it)
{
  it.x--;
  return it;
}

static ZFP_SCALAR_TYPE
_t2(CFP_ARRAY_TYPE, CFP_ITER_TYPE, get)(CFP_ITER_TYPE self)
{
  return static_cast<const ZFP_ARRAY_TYPE*>(self.array.object)->operator()(self.x);
}

static ZFP_SCALAR_TYPE
_t2(CFP_ARRAY_TYPE, CFP_ITER_TYPE, get_at)(CFP_ITER_TYPE self, ptrdiff_t d)
{
  self = _t2(CFP_ARRAY_TYPE, CFP_ITER_TYPE, next)(self, d);
  return static_cast<const ZFP_ARRAY_TYPE*>(self.array.object)->operator()(self.x);
}

static void
_t2(CFP_ARRAY_TYPE, CFP_ITER_TYPE, set)(CFP_ITER_TYPE self, ZFP_SCALAR_TYPE val)
{
  static_cast<ZFP_ARRAY_TYPE*>(self.array.object)->operator()(self.x) = val;
}

static void
_t2(CFP_ARRAY_TYPE, CFP_ITER_TYPE, set_at)(CFP_ITER_TYPE self, ptrdiff_t d, ZFP_SCALAR_TYPE val)
{
  self = _t2(CFP_ARRAY_TYPE, CFP_ITER_TYPE, next)(self, d);
  static_cast<ZFP_ARRAY_TYPE*>(self.array.object)->operator()(self.x) = val;
}

static CFP_REF_TYPE
_t2(CFP_ARRAY_TYPE, CFP_ITER_TYPE, ref)(CFP_ITER_TYPE self)
{
  return _t1(CFP_ARRAY_TYPE, ref)(self.array, self.x);
}

static CFP_REF_TYPE
_t2(CFP_ARRAY_TYPE, CFP_ITER_TYPE, ref_at)(CFP_ITER_TYPE self, ptrdiff_t d)
{
  self = _t2(CFP_ARRAY_TYPE, CFP_ITER_TYPE, next)(self, d);
  return _t1(CFP_ARRAY_TYPE, ref)(self.array, self.x);
}

static CFP_PTR_TYPE
_t2(CFP_ARRAY_TYPE, CFP_ITER_TYPE, ptr)(CFP_ITER_TYPE self)
{
  return _t1(CFP_ARRAY_TYPE, ptr)(self.array, self.x);
}

static CFP_PTR_TYPE
_t2(CFP_ARRAY_TYPE, CFP_ITER_TYPE, ptr_at)(CFP_ITER_TYPE self, ptrdiff_t d)
{
  self = _t2(CFP_ARRAY_TYPE, CFP_ITER_TYPE, next)(self, d);
  return _t1(CFP_ARRAY_TYPE, ptr)(self.array, self.x);
}

static size_t
_t2(CFP_ARRAY_TYPE, CFP_ITER_TYPE, i)(CFP_ITER_TYPE self)
{
  return self.x;
}
