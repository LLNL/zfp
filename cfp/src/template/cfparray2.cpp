// utility function: compute onedimensional offset from multidimensional index
static ptrdiff_t
ref_offset(const CFP_REF_TYPE& self)
{
  size_t nx = static_cast<const ZFP_ARRAY_TYPE*>(self.array.object)->size_x();
  return static_cast<ptrdiff_t>(self.x + nx * self.y);
}

// utility function: compute multidimensional index from onedimensional offset
static void
ref_set_offset(CFP_REF_TYPE& self, size_t offset)
{
  size_t nx = static_cast<const ZFP_ARRAY_TYPE*>(self.array.object)->size_x();
  self.x = offset % nx; offset /= nx;
  self.y = offset;
}

// utility function: compute onedimensional offset from multidimensional index
static ptrdiff_t
iter_offset(const CFP_ITER_TYPE& self)
{
  const ZFP_ARRAY_TYPE* container = static_cast<const ZFP_ARRAY_TYPE*>(self.array.object);
  size_t xmin = 0;
  size_t xmax = container->size_x();
  size_t ymin = 0;
  size_t ymax = container->size_y();
  size_t nx = xmax - xmin;
  size_t ny = ymax - ymin;
  size_t x = self.x;
  size_t y = self.y;
  size_t p = 0;
  if (y == ymax)
    p += nx * ny;
  else {
    size_t m = ~size_t(3);
    size_t by = std::max(y & m, ymin); size_t sy = std::min((by + 4) & m, ymax) - by; p += (by - ymin) * nx;
    size_t bx = std::max(x & m, xmin); size_t sx = std::min((bx + 4) & m, xmax) - bx; p += (bx - xmin) * sy;
    p += (y - by) * sx;
    p += (x - bx);
  }
  return static_cast<ptrdiff_t>(p);
}

// utility function: compute multidimensional index from onedimensional offset
static void
iter_set_offset(CFP_ITER_TYPE& self, size_t offset)
{
  const ZFP_ARRAY_TYPE* container = static_cast<const ZFP_ARRAY_TYPE*>(self.array.object);
  size_t xmin = 0;
  size_t xmax = container->size_x();
  size_t ymin = 0;
  size_t ymax = container->size_y();
  size_t nx = xmax - xmin;
  size_t ny = ymax - ymin;
  size_t p = offset;
  size_t x, y;
  if (p == nx * ny) {
    x = xmin;
    y = ymax;
  }
  else {
    size_t m = ~size_t(3);
    size_t by = std::max((ymin + p / nx) & m, ymin); size_t sy = std::min((by + 4) & m, ymax) - by; p -= (by - ymin) * nx;
    size_t bx = std::max((xmin + p / sy) & m, xmin); size_t sx = std::min((bx + 4) & m, xmax) - bx; p -= (bx - xmin) * sy;
    y = by + p / sx; p -= (y - by) * sx;
    x = bx + p;      p -= (x - bx);
  }
  self.x = x;
  self.y = y;
}

static CFP_ARRAY_TYPE
_t1(CFP_ARRAY_TYPE, ctor)(size_t nx, size_t ny, double rate, const ZFP_SCALAR_TYPE* p, size_t cache_size)
{
  CFP_ARRAY_TYPE a;
  a.object = new ZFP_ARRAY_TYPE(nx, ny, rate, p, cache_size);
  return a;
}

static size_t
_t1(CFP_ARRAY_TYPE, size_x)(CFP_ARRAY_TYPE self)
{
  return static_cast<const ZFP_ARRAY_TYPE*>(self.object)->size_x();
}

static size_t
_t1(CFP_ARRAY_TYPE, size_y)(CFP_ARRAY_TYPE self)
{
  return static_cast<const ZFP_ARRAY_TYPE*>(self.object)->size_y();
}

static void
_t1(CFP_ARRAY_TYPE, resize)(CFP_ARRAY_TYPE self, size_t nx, size_t ny, zfp_bool clear)
{
  static_cast<ZFP_ARRAY_TYPE*>(self.object)->resize(nx, ny, !!clear);
}

static ZFP_SCALAR_TYPE
_t1(CFP_ARRAY_TYPE, get)(CFP_ARRAY_TYPE self, size_t i, size_t j)
{
  return static_cast<const ZFP_ARRAY_TYPE*>(self.object)->operator()(i, j);
}

static void
_t1(CFP_ARRAY_TYPE, set)(CFP_ARRAY_TYPE self, size_t i, size_t j, ZFP_SCALAR_TYPE val)
{
  static_cast<ZFP_ARRAY_TYPE*>(self.object)->operator()(i, j) = val;
}

static CFP_REF_TYPE
_t1(CFP_ARRAY_TYPE, ref)(CFP_ARRAY_TYPE self, size_t i, size_t j)
{
  CFP_REF_TYPE r;
  r.array = self;
  r.x = i;
  r.y = j;
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
_t1(CFP_ARRAY_TYPE, ptr)(CFP_ARRAY_TYPE self, size_t i, size_t j)
{
  CFP_PTR_TYPE p;
  p.reference = _t1(CFP_ARRAY_TYPE, ref)(self, i, j);
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
  it.y = 0;
  return it;
}

static CFP_ITER_TYPE
_t1(CFP_ARRAY_TYPE, end)(CFP_ARRAY_TYPE self)
{
  CFP_ITER_TYPE it;
  it.array = self;
  it.x = 0;
  it.y = static_cast<const ZFP_ARRAY_TYPE*>(self.object)->size_y();
  return it;
}

static ZFP_SCALAR_TYPE
_t2(CFP_ARRAY_TYPE, CFP_REF_TYPE, get)(CFP_REF_TYPE self)
{
  return static_cast<const ZFP_ARRAY_TYPE*>(self.array.object)->operator()(self.x, self.y);
}

static void
_t2(CFP_ARRAY_TYPE, CFP_REF_TYPE, set)(CFP_REF_TYPE self, ZFP_SCALAR_TYPE val)
{
  static_cast<ZFP_ARRAY_TYPE*>(self.array.object)->operator()(self.x, self.y) = val;
}

static void
_t2(CFP_ARRAY_TYPE, CFP_REF_TYPE, copy)(CFP_REF_TYPE self, CFP_REF_TYPE src)
{
  static_cast<ZFP_ARRAY_TYPE*>(self.array.object)->operator()(self.x, self.y) =
    static_cast<const ZFP_ARRAY_TYPE*>(src.array.object)->operator()(src.x, src.y);
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
         lhs.reference.y == rhs.reference.y;
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
  return static_cast<const ZFP_ARRAY_TYPE*>(self.reference.array.object)->operator()(self.reference.x, self.reference.y);
}

static ZFP_SCALAR_TYPE
_t2(CFP_ARRAY_TYPE, CFP_PTR_TYPE, get_at)(CFP_PTR_TYPE self, ptrdiff_t d)
{
  self = _t2(CFP_ARRAY_TYPE, CFP_PTR_TYPE, next)(self, d);
  return static_cast<const ZFP_ARRAY_TYPE*>(self.reference.array.object)->operator()(self.reference.x, self.reference.y);
}

static void
_t2(CFP_ARRAY_TYPE, CFP_PTR_TYPE, set)(CFP_PTR_TYPE self, ZFP_SCALAR_TYPE val)
{
  static_cast<ZFP_ARRAY_TYPE*>(self.reference.array.object)->operator()(self.reference.x, self.reference.y) = val;
}

static void
_t2(CFP_ARRAY_TYPE, CFP_PTR_TYPE, set_at)(CFP_PTR_TYPE self, ptrdiff_t d, ZFP_SCALAR_TYPE val)
{
  self = _t2(CFP_ARRAY_TYPE, CFP_PTR_TYPE, next)(self, d);
  static_cast<ZFP_ARRAY_TYPE*>(self.reference.array.object)->operator()(self.reference.x, self.reference.y) = val;
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
  return lhs.array.object == rhs.array.object && iter_offset(lhs) < iter_offset(rhs);
}

static zfp_bool
_t2(CFP_ARRAY_TYPE, CFP_ITER_TYPE, gt)(CFP_ITER_TYPE lhs, CFP_ITER_TYPE rhs)
{
  return lhs.array.object == rhs.array.object && iter_offset(lhs) > iter_offset(rhs);
}

static zfp_bool
_t2(CFP_ARRAY_TYPE, CFP_ITER_TYPE, leq)(CFP_ITER_TYPE lhs, CFP_ITER_TYPE rhs)
{
  return lhs.array.object == rhs.array.object && iter_offset(lhs) <= iter_offset(rhs);
}

static zfp_bool
_t2(CFP_ARRAY_TYPE, CFP_ITER_TYPE, geq)(CFP_ITER_TYPE lhs, CFP_ITER_TYPE rhs)
{
  return lhs.array.object == rhs.array.object && iter_offset(lhs) >= iter_offset(rhs);
}

static zfp_bool
_t2(CFP_ARRAY_TYPE, CFP_ITER_TYPE, eq)(CFP_ITER_TYPE lhs, CFP_ITER_TYPE rhs)
{
  return lhs.array.object == rhs.array.object &&
         lhs.x == rhs.x &&
         lhs.y == rhs.y;
}

static zfp_bool
_t2(CFP_ARRAY_TYPE, CFP_ITER_TYPE, neq)(CFP_ITER_TYPE lhs, CFP_ITER_TYPE rhs)
{
  return !_t2(CFP_ARRAY_TYPE, CFP_ITER_TYPE, eq)(lhs, rhs);
}

static ptrdiff_t
_t2(CFP_ARRAY_TYPE, CFP_ITER_TYPE, distance)(CFP_ITER_TYPE first, CFP_ITER_TYPE last)
{
  return iter_offset(last) - iter_offset(first);
}

static CFP_ITER_TYPE
_t2(CFP_ARRAY_TYPE, CFP_ITER_TYPE, next)(CFP_ITER_TYPE it, ptrdiff_t d)
{
  iter_set_offset(it, iter_offset(it) + d);
  return it;
}

static CFP_ITER_TYPE
_t2(CFP_ARRAY_TYPE, CFP_ITER_TYPE, prev)(CFP_ITER_TYPE it, ptrdiff_t d)
{
  iter_set_offset(it, iter_offset(it) - d);
  return it;
}

static CFP_ITER_TYPE
_t2(CFP_ARRAY_TYPE, CFP_ITER_TYPE, inc)(CFP_ITER_TYPE it)
{
  const ZFP_ARRAY_TYPE* container = static_cast<const ZFP_ARRAY_TYPE*>(it.array.object);
  size_t xmin = 0;
  size_t xmax = container->size_x();
  size_t ymin = 0;
  size_t ymax = container->size_y();
  size_t m = ~size_t(3);
  ++it.x;
  if (!(it.x & 3u) || it.x == xmax) {
    it.x = std::max((it.x - 1) & m, xmin);
    ++it.y;
    if (!(it.y & 3u) || it.y == ymax) {
      it.y = std::max((it.y - 1) & m, ymin);
      // done with block; advance to next
      it.x = (it.x + 4) & m;
      if (it.x >= xmax) {
        it.x = xmin;
        it.y = (it.y + 4) & m;
        if (it.y >= ymax)
          it.y = ymax;
      }
    }
  }
  return it;
}

static CFP_ITER_TYPE
_t2(CFP_ARRAY_TYPE, CFP_ITER_TYPE, dec)(CFP_ITER_TYPE it)
{
  const ZFP_ARRAY_TYPE* container = static_cast<const ZFP_ARRAY_TYPE*>(it.array.object);
  size_t xmin = 0;
  size_t xmax = container->size_x();
  size_t ymin = 0;
  size_t ymax = container->size_y();
  size_t m = ~size_t(3);
  if (it.y == ymax) {
    it.x = xmax - 1;
    it.y = ymax - 1;
  }
  else {
    if (!(it.x & 3u) || it.x == xmin) {
      it.x = std::min((it.x + 4) & m, xmax);
      if (!(it.y & 3u) || it.y == ymin) {
        it.y = std::min((it.y + 4) & m, ymax);
        // done with block; advance to next
        it.x = (it.x - 1) & m;
        if (it.x <= xmin) {
          it.x = xmax;
          it.y = (it.y - 1) & m;
          if (it.y <= ymin)
            it.y = ymin;
        }
      }
      --it.y;
    }
    --it.x;
  }
  return it;
}

static ZFP_SCALAR_TYPE
_t2(CFP_ARRAY_TYPE, CFP_ITER_TYPE, get)(CFP_ITER_TYPE self)
{
  return static_cast<const ZFP_ARRAY_TYPE*>(self.array.object)->operator()(self.x, self.y);
}

static ZFP_SCALAR_TYPE
_t2(CFP_ARRAY_TYPE, CFP_ITER_TYPE, get_at)(CFP_ITER_TYPE self, ptrdiff_t d)
{
  self = _t2(CFP_ARRAY_TYPE, CFP_ITER_TYPE, next)(self, d);
  return static_cast<const ZFP_ARRAY_TYPE*>(self.array.object)->operator()(self.x, self.y);
}

static void
_t2(CFP_ARRAY_TYPE, CFP_ITER_TYPE, set)(CFP_ITER_TYPE self, ZFP_SCALAR_TYPE val)
{
  static_cast<ZFP_ARRAY_TYPE*>(self.array.object)->operator()(self.x, self.y) = val;
}

static void
_t2(CFP_ARRAY_TYPE, CFP_ITER_TYPE, set_at)(CFP_ITER_TYPE self, ptrdiff_t d, ZFP_SCALAR_TYPE val)
{
  self = _t2(CFP_ARRAY_TYPE, CFP_ITER_TYPE, next)(self, d);
  static_cast<ZFP_ARRAY_TYPE*>(self.array.object)->operator()(self.x, self.y) = val;
}

static CFP_REF_TYPE
_t2(CFP_ARRAY_TYPE, CFP_ITER_TYPE, ref)(CFP_ITER_TYPE self)
{
  return _t1(CFP_ARRAY_TYPE, ref)(self.array, self.x, self.y);
}

static CFP_REF_TYPE
_t2(CFP_ARRAY_TYPE, CFP_ITER_TYPE, ref_at)(CFP_ITER_TYPE self, ptrdiff_t d)
{
  self = _t2(CFP_ARRAY_TYPE, CFP_ITER_TYPE, next)(self, d);
  return _t1(CFP_ARRAY_TYPE, ref)(self.array, self.x, self.y);
}

static CFP_PTR_TYPE
_t2(CFP_ARRAY_TYPE, CFP_ITER_TYPE, ptr)(CFP_ITER_TYPE self)
{
  return _t1(CFP_ARRAY_TYPE, ptr)(self.array, self.x, self.y);
}

static CFP_PTR_TYPE
_t2(CFP_ARRAY_TYPE, CFP_ITER_TYPE, ptr_at)(CFP_ITER_TYPE self, ptrdiff_t d)
{
  self = _t2(CFP_ARRAY_TYPE, CFP_ITER_TYPE, next)(self, d);
  return _t1(CFP_ARRAY_TYPE, ptr)(self.array, self.x, self.y);
}

static size_t
_t2(CFP_ARRAY_TYPE, CFP_ITER_TYPE, i)(CFP_ITER_TYPE self)
{
  return self.x;
}

static size_t
_t2(CFP_ARRAY_TYPE, CFP_ITER_TYPE, j)(CFP_ITER_TYPE self)
{
  return self.y;
}
