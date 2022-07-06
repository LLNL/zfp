// utility function: compute onedimensional offset from multidimensional index
static ptrdiff_t
ref_offset(const CFP_REF_TYPE& self)
{
  size_t nx = static_cast<const ZFP_ARRAY_TYPE*>(self.array.object)->size_x();
  size_t ny = static_cast<const ZFP_ARRAY_TYPE*>(self.array.object)->size_y();
  size_t nz = static_cast<const ZFP_ARRAY_TYPE*>(self.array.object)->size_z();
  return static_cast<ptrdiff_t>(self.x + nx * (self.y + ny * (self.z + nz * self.w)));
}

// utility function: compute multidimensional index from onedimensional offset
static void
ref_set_offset(CFP_REF_TYPE& self, size_t offset)
{
  size_t nx = static_cast<const ZFP_ARRAY_TYPE*>(self.array.object)->size_x();
  size_t ny = static_cast<const ZFP_ARRAY_TYPE*>(self.array.object)->size_y();
  size_t nz = static_cast<const ZFP_ARRAY_TYPE*>(self.array.object)->size_z();
  self.x = offset % nx; offset /= nx;
  self.y = offset % ny; offset /= ny;
  self.z = offset % nz; offset /= nz;
  self.w = offset;
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
  size_t zmin = 0;
  size_t zmax = container->size_z();
  size_t wmin = 0;
  size_t wmax = container->size_w();
  size_t nx = xmax - xmin;
  size_t ny = ymax - ymin;
  size_t nz = zmax - zmin;
  size_t nw = wmax - wmin;
  size_t x = self.x;
  size_t y = self.y;
  size_t z = self.z;
  size_t w = self.w;
  size_t p = 0;
  if (w == wmax)
    p += nx * ny * nz * nw;
  else {
    size_t m = ~size_t(3);
    size_t bw = std::max(w & m, wmin); size_t sw = std::min((bw + 4) & m, wmax) - bw; p += (bw - wmin) * nx * ny * nz;
    size_t bz = std::max(z & m, zmin); size_t sz = std::min((bz + 4) & m, zmax) - bz; p += (bz - zmin) * nx * ny * sw;
    size_t by = std::max(y & m, ymin); size_t sy = std::min((by + 4) & m, ymax) - by; p += (by - ymin) * nx * sz * sw;
    size_t bx = std::max(x & m, xmin); size_t sx = std::min((bx + 4) & m, xmax) - bx; p += (bx - xmin) * sy * sz * sw;
    p += (w - bw) * sx * sy * sz;
    p += (z - bz) * sx * sy;
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
  size_t zmin = 0;
  size_t zmax = container->size_z();
  size_t wmin = 0;
  size_t wmax = container->size_w();
  size_t nx = xmax - xmin;
  size_t ny = ymax - ymin;
  size_t nz = zmax - zmin;
  size_t nw = wmax - wmin;
  size_t p = offset;
  size_t x, y, z, w;
  if (p == nx * ny * nz * nw) {
    x = xmin;
    y = ymin;
    z = zmin;
    w = wmax;
  }
  else {
    size_t m = ~size_t(3);
    size_t bw = std::max((wmin + p / (nx * ny * nz)) & m, wmin); size_t sw = std::min((bw + 4) & m, wmax) - bw; p -= (bw - wmin) * nx * ny * nz;
    size_t bz = std::max((zmin + p / (nx * ny * sw)) & m, zmin); size_t sz = std::min((bz + 4) & m, zmax) - bz; p -= (bz - zmin) * nx * ny * sw;
    size_t by = std::max((ymin + p / (nx * sz * sw)) & m, ymin); size_t sy = std::min((by + 4) & m, ymax) - by; p -= (by - ymin) * nx * sz * sw;
    size_t bx = std::max((xmin + p / (sy * sz * sw)) & m, xmin); size_t sx = std::min((bx + 4) & m, xmax) - bx; p -= (bx - xmin) * sy * sz * sw;
    w = bw + p / (sx * sy * sz); p -= (w - bw) * sx * sy * sz;
    z = bz + p / (sx * sy);      p -= (z - bz) * sx * sy;
    y = by + p / sx;             p -= (y - by) * sx;
    x = bx + p;                  p -= (x - bx);
  }
  self.x = x;
  self.y = y;
  self.z = z;
  self.w = w;
}

static CFP_ARRAY_TYPE
_t1(CFP_ARRAY_TYPE, ctor)(size_t nx, size_t ny, size_t nz, size_t nw, double rate, const ZFP_SCALAR_TYPE* p, size_t cache_size)
{
  CFP_ARRAY_TYPE a;
  a.object = new ZFP_ARRAY_TYPE(nx, ny, nz, nw, rate, p, cache_size);
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

static size_t
_t1(CFP_ARRAY_TYPE, size_z)(CFP_ARRAY_TYPE self)
{
  return static_cast<const ZFP_ARRAY_TYPE*>(self.object)->size_z();
}

static size_t
_t1(CFP_ARRAY_TYPE, size_w)(CFP_ARRAY_TYPE self)
{
  return static_cast<const ZFP_ARRAY_TYPE*>(self.object)->size_w();
}

static void
_t1(CFP_ARRAY_TYPE, resize)(CFP_ARRAY_TYPE self, size_t nx, size_t ny, size_t nz, size_t nw, zfp_bool clear)
{
  static_cast<ZFP_ARRAY_TYPE*>(self.object)->resize(nx, ny, nz, nw, !!clear);
}

static ZFP_SCALAR_TYPE
_t1(CFP_ARRAY_TYPE, get)(CFP_ARRAY_TYPE self, size_t i, size_t j, size_t k, size_t l)
{
  return static_cast<const ZFP_ARRAY_TYPE*>(self.object)->operator()(i, j, k, l);
}

static void
_t1(CFP_ARRAY_TYPE, set)(CFP_ARRAY_TYPE self, size_t i, size_t j, size_t k, size_t l, ZFP_SCALAR_TYPE val)
{
  static_cast<ZFP_ARRAY_TYPE*>(self.object)->operator()(i, j, k, l) = val;
}

static CFP_REF_TYPE
_t1(CFP_ARRAY_TYPE, ref)(CFP_ARRAY_TYPE self, size_t i, size_t j, size_t k, size_t l)
{
  CFP_REF_TYPE r;
  r.array = self;
  r.x = i;
  r.y = j;
  r.z = k;
  r.w = l;
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
_t1(CFP_ARRAY_TYPE, ptr)(CFP_ARRAY_TYPE self, size_t i, size_t j, size_t k, size_t l)
{
  CFP_PTR_TYPE p;
  p.reference = _t1(CFP_ARRAY_TYPE, ref)(self, i, j, k, l);
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
  it.z = 0;
  it.w = 0;
  return it;
}

static CFP_ITER_TYPE
_t1(CFP_ARRAY_TYPE, end)(CFP_ARRAY_TYPE self)
{
  CFP_ITER_TYPE it;
  it.array = self;
  it.x = 0;
  it.y = 0;
  it.z = 0;
  it.w = static_cast<const ZFP_ARRAY_TYPE*>(self.object)->size_w();
  return it;
}

static ZFP_SCALAR_TYPE
_t2(CFP_ARRAY_TYPE, CFP_REF_TYPE, get)(CFP_REF_TYPE self)
{
  return static_cast<const ZFP_ARRAY_TYPE*>(self.array.object)->operator()(self.x, self.y, self.z, self.w);
}

static void
_t2(CFP_ARRAY_TYPE, CFP_REF_TYPE, set)(CFP_REF_TYPE self, ZFP_SCALAR_TYPE val)
{
  static_cast<ZFP_ARRAY_TYPE*>(self.array.object)->operator()(self.x, self.y, self.z, self.w) = val;
}

static void
_t2(CFP_ARRAY_TYPE, CFP_REF_TYPE, copy)(CFP_REF_TYPE self, CFP_REF_TYPE src)
{
  static_cast<ZFP_ARRAY_TYPE*>(self.array.object)->operator()(self.x, self.y, self.z, self.w) =
    static_cast<const ZFP_ARRAY_TYPE*>(src.array.object)->operator()(src.x, src.y, src.z, src.w);
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
         lhs.reference.z == rhs.reference.z &&
         lhs.reference.w == rhs.reference.w;
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
  return static_cast<const ZFP_ARRAY_TYPE*>(self.reference.array.object)->operator()(self.reference.x, self.reference.y, self.reference.z, self.reference.w);
}

static ZFP_SCALAR_TYPE
_t2(CFP_ARRAY_TYPE, CFP_PTR_TYPE, get_at)(CFP_PTR_TYPE self, ptrdiff_t d)
{
  self = _t2(CFP_ARRAY_TYPE, CFP_PTR_TYPE, next)(self, d);
  return static_cast<const ZFP_ARRAY_TYPE*>(self.reference.array.object)->operator()(self.reference.x, self.reference.y, self.reference.z, self.reference.w);
}

static void
_t2(CFP_ARRAY_TYPE, CFP_PTR_TYPE, set)(CFP_PTR_TYPE self, ZFP_SCALAR_TYPE val)
{
  static_cast<ZFP_ARRAY_TYPE*>(self.reference.array.object)->operator()(self.reference.x, self.reference.y, self.reference.z, self.reference.w) = val;
}

static void
_t2(CFP_ARRAY_TYPE, CFP_PTR_TYPE, set_at)(CFP_PTR_TYPE self, ptrdiff_t d, ZFP_SCALAR_TYPE val)
{
  self = _t2(CFP_ARRAY_TYPE, CFP_PTR_TYPE, next)(self, d);
  static_cast<ZFP_ARRAY_TYPE*>(self.reference.array.object)->operator()(self.reference.x, self.reference.y, self.reference.z, self.reference.w) = val;
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
         lhs.y == rhs.y &&
         lhs.z == rhs.z &&
         lhs.w == rhs.w;
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
  size_t zmin = 0;
  size_t zmax = container->size_z();
  size_t wmin = 0;
  size_t wmax = container->size_w();
  size_t m = ~size_t(3);
  ++it.x;
  if (!(it.x & 3u) || it.x == xmax) {
    it.x = std::max((it.x - 1) & m, xmin);
    ++it.y;
    if (!(it.y & 3u) || it.y == ymax) {
      it.y = std::max((it.y - 1) & m, ymin);
      ++it.z;
      if (!(it.z & 3u) || it.z == zmax) {
        it.z = std::max((it.z - 1) & m, zmin);
        ++it.w;
        if (!(it.w & 3u) || it.w == wmax) {
          it.w = std::max((it.w - 1) & m, wmin);
          // done with block; advance to next
          it.x = (it.x + 4) & m;
          if (it.x >= xmax) {
            it.x = xmin;
            it.y = (it.y + 4) & m;
            if (it.y >= ymax) {
              it.y = ymin;
              it.z = (it.z + 4) & m;
              if (it.z >= zmax) {
                it.z = zmin;
                it.w = (it.w + 4) & m;
                if (it.w >= wmax)
                  it.w = wmax;
              }
            }
          }
        }
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
  size_t zmin = 0;
  size_t zmax = container->size_z();
  size_t wmin = 0;
  size_t wmax = container->size_w();
  size_t m = ~size_t(3);
  if (it.w == wmax) {
    it.x = xmax - 1;
    it.y = ymax - 1;
    it.z = zmax - 1;
    it.w = wmax - 1;
  }
  else {
    if (!(it.x & 3u) || it.x == xmin) {
      it.x = std::min((it.x + 4) & m, xmax);
      if (!(it.y & 3u) || it.y == ymin) {
        it.y = std::min((it.y + 4) & m, ymax);
        if (!(it.z & 3u) || it.z == zmin) {
          it.z = std::min((it.z + 4) & m, zmax);
          if (!(it.w & 3u) || it.w == wmin) {
            it.w = std::min((it.w + 4) & m, wmax);
            // done with block; advance to next
            it.x = (it.x - 1) & m;
            if (it.x <= xmin) {
              it.x = xmax;
              it.y = (it.y - 1) & m;
              if (it.y <= ymin) {
                it.y = ymax;
                it.z = (it.z - 1) & m;
                if (it.z <= zmin) {
                  it.z = zmax;
                  it.w = (it.w - 1) & m;
                  if (it.w <= wmin)
                    it.w = wmin;
                }
              }
            }
          }
          --it.w;
        }
        --it.z;
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
  return static_cast<ZFP_ARRAY_TYPE*>(self.array.object)->operator()(self.x, self.y, self.z, self.w);
}

static ZFP_SCALAR_TYPE
_t2(CFP_ARRAY_TYPE, CFP_ITER_TYPE, get_at)(CFP_ITER_TYPE self, ptrdiff_t d)
{
  self = _t2(CFP_ARRAY_TYPE, CFP_ITER_TYPE, next)(self, d);
  return static_cast<ZFP_ARRAY_TYPE*>(self.array.object)->operator()(self.x, self.y, self.z, self.w);
}

static void
_t2(CFP_ARRAY_TYPE, CFP_ITER_TYPE, set)(CFP_ITER_TYPE self, ZFP_SCALAR_TYPE val)
{
  static_cast<ZFP_ARRAY_TYPE*>(self.array.object)->operator()(self.x, self.y, self.z, self.w) = val;
}

static void
_t2(CFP_ARRAY_TYPE, CFP_ITER_TYPE, set_at)(CFP_ITER_TYPE self, ptrdiff_t d, ZFP_SCALAR_TYPE val)
{
  self = _t2(CFP_ARRAY_TYPE, CFP_ITER_TYPE, next)(self, d);
  static_cast<ZFP_ARRAY_TYPE*>(self.array.object)->operator()(self.x, self.y, self.z, self.w) = val;
}

static CFP_REF_TYPE
_t2(CFP_ARRAY_TYPE, CFP_ITER_TYPE, ref)(CFP_ITER_TYPE self)
{
  return _t1(CFP_ARRAY_TYPE, ref)(self.array, self.x, self.y, self.z, self.w);
}

static CFP_REF_TYPE
_t2(CFP_ARRAY_TYPE, CFP_ITER_TYPE, ref_at)(CFP_ITER_TYPE self, ptrdiff_t d)
{
  self = _t2(CFP_ARRAY_TYPE, CFP_ITER_TYPE, next)(self, d);
  return _t1(CFP_ARRAY_TYPE, ref)(self.array, self.x, self.y, self.z, self.w);
}

static CFP_PTR_TYPE
_t2(CFP_ARRAY_TYPE, CFP_ITER_TYPE, ptr)(CFP_ITER_TYPE self)
{
  return _t1(CFP_ARRAY_TYPE, ptr)(self.array, self.x, self.y, self.z, self.w);
}

static CFP_PTR_TYPE
_t2(CFP_ARRAY_TYPE, CFP_ITER_TYPE, ptr_at)(CFP_ITER_TYPE self, ptrdiff_t d)
{
  self = _t2(CFP_ARRAY_TYPE, CFP_ITER_TYPE, next)(self, d);
  return _t1(CFP_ARRAY_TYPE, ptr)(self.array, self.x, self.y, self.z, self.w);
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

static size_t
_t2(CFP_ARRAY_TYPE, CFP_ITER_TYPE, k)(CFP_ITER_TYPE self)
{
  return self.z;
}

static size_t
_t2(CFP_ARRAY_TYPE, CFP_ITER_TYPE, l)(CFP_ITER_TYPE self)
{
  return self.w;
}
