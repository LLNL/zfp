// utility function: compute onedimensional offset from multidimensional index
static ptrdiff_t
ref_offset(const CFP_REF_TYPE& self)
{
  size_t nx = static_cast<const ZFP_ARRAY_TYPE*>(self.array.object)->size_x();
  size_t ny = static_cast<const ZFP_ARRAY_TYPE*>(self.array.object)->size_y();
  return static_cast<ptrdiff_t>(self.x + nx * (self.y + ny * self.z));
}

// utility function: compute multidimensional index from onedimensional offset
static void
ref_set_offset(CFP_REF_TYPE& self, size_t offset)
{
  size_t nx = static_cast<const ZFP_ARRAY_TYPE*>(self.array.object)->size_x();
  size_t ny = static_cast<const ZFP_ARRAY_TYPE*>(self.array.object)->size_y();
  self.x = offset % nx; offset /= nx;
  self.y = offset % ny; offset /= ny;
  self.z = offset;
}

static CFP_ARRAY_TYPE
_t1(CFP_ARRAY_TYPE, ctor)(size_t nx, size_t ny, size_t nz, double rate, const ZFP_SCALAR_TYPE* p, size_t cache_size)
{
  CFP_ARRAY_TYPE a;
  a.object = new ZFP_ARRAY_TYPE(nx, ny, nz, rate, p, cache_size);
  return a;
}

static void
_t1(CFP_ARRAY_TYPE, resize)(CFP_ARRAY_TYPE self, size_t nx, size_t ny, size_t nz, zfp_bool clear)
{
  static_cast<ZFP_ARRAY_TYPE*>(self.object)->resize(nx, ny, nz, !!clear);
}

static CFP_REF_TYPE
_t1(CFP_ARRAY_TYPE, ref)(CFP_ARRAY_TYPE self, size_t i, size_t j, size_t k)
{
  CFP_REF_TYPE r;
  r.array = self;
  r.x = i;
  r.y = j;
  r.z = k;
  return r;
}

static CFP_REF_TYPE
_t1(CFP_ARRAY_TYPE, ref_flat)(CFP_ARRAY_TYPE self, size_t i)
{
  CFP_REF_TYPE r;
  r.array = self;
  ref_set_offset(r, i);
  return r;
}

static CFP_PTR_TYPE
_t1(CFP_ARRAY_TYPE, ptr)(CFP_ARRAY_TYPE self, size_t i, size_t j, size_t k)
{
  CFP_PTR_TYPE p;
  p.reference = _t1(CFP_ARRAY_TYPE, ref)(self, i, j, k);
  return p;
}

static CFP_PTR_TYPE
_t1(CFP_ARRAY_TYPE, ptr_flat)(CFP_ARRAY_TYPE self, size_t i)
{
  CFP_PTR_TYPE p;
  p.reference = _t1(CFP_ARRAY_TYPE, ref_flat)(self, i);
  return p;
}

static ZFP_SCALAR_TYPE
_t2(CFP_ARRAY_TYPE, CFP_REF_TYPE, get)(CFP_REF_TYPE self)
{
  return static_cast<const ZFP_ARRAY_TYPE*>(self.array.object)->operator()(self.x, self.y, self.z);
}

static void
_t2(CFP_ARRAY_TYPE, CFP_REF_TYPE, set)(CFP_REF_TYPE self, ZFP_SCALAR_TYPE val)
{
  static_cast<ZFP_ARRAY_TYPE*>(self.array.object)->operator()(self.x, self.y, self.z) = val;
}

static void
_t2(CFP_ARRAY_TYPE, CFP_REF_TYPE, copy)(CFP_REF_TYPE self, CFP_REF_TYPE src)
{
  static_cast<ZFP_ARRAY_TYPE*>(self.array.object)->operator()(self.x, self.y, self.z) =
    static_cast<const ZFP_ARRAY_TYPE*>(src.array.object)->operator()(src.x, src.y, src.z);
}

static zfp_bool
_t2(CFP_ARRAY_TYPE, CFP_PTR_TYPE, lt)(CFP_PTR_TYPE lhs, CFP_PTR_TYPE rhs)
{
  return lhs.reference.array.object == rhs.reference.array.object && ref_offset(lhs.reference) < ref_offset(rhs.reference);
}

static zfp_bool
_t2(CFP_ARRAY_TYPE, CFP_PTR_TYPE, gt)(CFP_PTR_TYPE lhs, CFP_PTR_TYPE rhs)
{
  return lhs.reference.array.object == rhs.reference.array.object && ref_offset(lhs.reference) > ref_offset(rhs.reference);
}

static zfp_bool
_t2(CFP_ARRAY_TYPE, CFP_PTR_TYPE, leq)(CFP_PTR_TYPE lhs, CFP_PTR_TYPE rhs)
{
  return lhs.reference.array.object == rhs.reference.array.object && ref_offset(lhs.reference) <= ref_offset(rhs.reference);
}

static zfp_bool
_t2(CFP_ARRAY_TYPE, CFP_PTR_TYPE, geq)(CFP_PTR_TYPE lhs, CFP_PTR_TYPE rhs)
{
  return lhs.reference.array.object == rhs.reference.array.object && ref_offset(lhs.reference) >= ref_offset(rhs.reference);
}

static zfp_bool
_t2(CFP_ARRAY_TYPE, CFP_PTR_TYPE, eq)(CFP_PTR_TYPE lhs, CFP_PTR_TYPE rhs)
{
  return lhs.reference.array.object == rhs.reference.array.object &&
         lhs.reference.x == rhs.reference.x &&
         lhs.reference.y == rhs.reference.y &&
         lhs.reference.z == rhs.reference.z;
}

static zfp_bool
_t2(CFP_ARRAY_TYPE, CFP_PTR_TYPE, neq)(CFP_PTR_TYPE lhs, CFP_PTR_TYPE rhs)
{
  return !_t2(CFP_ARRAY_TYPE, CFP_PTR_TYPE, eq)(lhs, rhs);
}

static ptrdiff_t
_t2(CFP_ARRAY_TYPE, CFP_PTR_TYPE, distance)(CFP_PTR_TYPE first, CFP_PTR_TYPE last)
{
  return ref_offset(last.reference) - ref_offset(first.reference);
}

static CFP_PTR_TYPE
_t2(CFP_ARRAY_TYPE, CFP_PTR_TYPE, next)(CFP_PTR_TYPE p, ptrdiff_t d)
{
  ref_set_offset(p.reference, ref_offset(p.reference) + d);
  return p;
}

static CFP_PTR_TYPE
_t2(CFP_ARRAY_TYPE, CFP_PTR_TYPE, prev)(CFP_PTR_TYPE p, ptrdiff_t d)
{
  return _t2(CFP_ARRAY_TYPE, CFP_PTR_TYPE, next)(p, -d);
}

static CFP_PTR_TYPE
_t2(CFP_ARRAY_TYPE, CFP_PTR_TYPE, inc)(CFP_PTR_TYPE p)
{
  return _t2(CFP_ARRAY_TYPE, CFP_PTR_TYPE, next)(p, +1);
}

static CFP_PTR_TYPE
_t2(CFP_ARRAY_TYPE, CFP_PTR_TYPE, dec)(CFP_PTR_TYPE p)
{
  return _t2(CFP_ARRAY_TYPE, CFP_PTR_TYPE, next)(p, -1);
}

static ZFP_SCALAR_TYPE
_t2(CFP_ARRAY_TYPE, CFP_PTR_TYPE, get)(CFP_PTR_TYPE self)
{
  return static_cast<const ZFP_ARRAY_TYPE*>(self.reference.array.object)->operator()(self.reference.x, self.reference.y, self.reference.z);
}

static ZFP_SCALAR_TYPE
_t2(CFP_ARRAY_TYPE, CFP_PTR_TYPE, get_at)(CFP_PTR_TYPE self, ptrdiff_t d)
{
  self = _t2(CFP_ARRAY_TYPE, CFP_PTR_TYPE, next)(self, d);
  return static_cast<const ZFP_ARRAY_TYPE*>(self.reference.array.object)->operator()(self.reference.x, self.reference.y, self.reference.z);
}

static void
_t2(CFP_ARRAY_TYPE, CFP_PTR_TYPE, set)(CFP_PTR_TYPE self, ZFP_SCALAR_TYPE val)
{
  static_cast<ZFP_ARRAY_TYPE*>(self.reference.array.object)->operator()(self.reference.x, self.reference.y, self.reference.z) = val;
}

static void
_t2(CFP_ARRAY_TYPE, CFP_PTR_TYPE, set_at)(CFP_PTR_TYPE self, ptrdiff_t d, ZFP_SCALAR_TYPE val)
{
  self = _t2(CFP_ARRAY_TYPE, CFP_PTR_TYPE, next)(self, d);
  static_cast<ZFP_ARRAY_TYPE*>(self.reference.array.object)->operator()(self.reference.x, self.reference.y, self.reference.z) = val;
}

static CFP_REF_TYPE
_t2(CFP_ARRAY_TYPE, CFP_PTR_TYPE, ref_at)(CFP_PTR_TYPE self, ptrdiff_t d)
{
  self = _t2(CFP_ARRAY_TYPE, CFP_PTR_TYPE, next)(self, d);
  return self.reference;
}
