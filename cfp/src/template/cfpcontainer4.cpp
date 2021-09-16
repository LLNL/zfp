/* utility function: compute one-dimensional offset from multi-dimensional index */
static ptrdiff_t
_t1(CFP_CONTAINER_TYPE, ref_offset)(const CFP_REF_TYPE& self)
{
  size_t nx = static_cast<const ZFP_CONTAINER_TYPE*>(self.container)->size_x();
  size_t ny = static_cast<const ZFP_CONTAINER_TYPE*>(self.container)->size_y();
  size_t nz = static_cast<const ZFP_CONTAINER_TYPE*>(self.container)->size_z();
  return static_cast<ptrdiff_t>(self.x + nx * (self.y + ny * (self.z + nz * self.w)));
}

static ptrdiff_t
_t1(CFP_CONTAINER_TYPE, ptr_offset)(const CFP_PTR_TYPE& self)
{
  size_t nx = static_cast<const ZFP_CONTAINER_TYPE*>(self.container)->size_x();
  size_t ny = static_cast<const ZFP_CONTAINER_TYPE*>(self.container)->size_y();
  size_t nz = static_cast<const ZFP_CONTAINER_TYPE*>(self.container)->size_z();
  return static_cast<ptrdiff_t>(self.x + nx * (self.y + ny * (self.z + nz * self.w)));
}

static ptrdiff_t
_t1(CFP_CONTAINER_TYPE, iter_offset)(const CFP_ITER_TYPE& self)
{
  const ZFP_CONTAINER_TYPE* container = static_cast<const ZFP_CONTAINER_TYPE*>(self.container);
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

/* utility function: compute multi-dimensional index from one-dimensional offset */
static void
_t1(CFP_CONTAINER_TYPE, ref_set_offset)(CFP_REF_TYPE& self, size_t offset)
{
  size_t nx = static_cast<const ZFP_CONTAINER_TYPE*>(self.container)->size_x();
  size_t ny = static_cast<const ZFP_CONTAINER_TYPE*>(self.container)->size_y();
  size_t nz = static_cast<const ZFP_CONTAINER_TYPE*>(self.container)->size_z();
  self.x = offset % nx; offset /= nx;
  self.y = offset % ny; offset /= ny;
  self.z = offset % nz; offset /= nz;
  self.w = offset;
}

static void
_t1(CFP_CONTAINER_TYPE, ptr_set_offset)(CFP_PTR_TYPE& self, size_t offset)
{
  size_t nx = static_cast<const ZFP_CONTAINER_TYPE*>(self.container)->size_x();
  size_t ny = static_cast<const ZFP_CONTAINER_TYPE*>(self.container)->size_y();
  size_t nz = static_cast<const ZFP_CONTAINER_TYPE*>(self.container)->size_z();
  self.x = offset % nx; offset /= nx;
  self.y = offset % ny; offset /= ny;
  self.z = offset % nz; offset /= nz;
  self.w = offset;
}

static void
_t1(CFP_CONTAINER_TYPE, iter_set_offset)(CFP_ITER_TYPE& self, size_t offset)
{
  const ZFP_CONTAINER_TYPE* container = static_cast<const ZFP_CONTAINER_TYPE*>(self.container);
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

/* Containers */
static size_t
_t1(CFP_CONTAINER_TYPE, size_x)(const CFP_CONTAINER_TYPE self)
{
  return static_cast<const ZFP_CONTAINER_TYPE*>(self.object)->size_x();
}

static size_t
_t1(CFP_CONTAINER_TYPE, size_y)(const CFP_CONTAINER_TYPE self)
{
  return static_cast<const ZFP_CONTAINER_TYPE*>(self.object)->size_y();
}

static size_t
_t1(CFP_CONTAINER_TYPE, size_z)(const CFP_CONTAINER_TYPE self)
{
  return static_cast<const ZFP_CONTAINER_TYPE*>(self.object)->size_z();
}

static size_t
_t1(CFP_CONTAINER_TYPE, size_w)(const CFP_CONTAINER_TYPE self)
{
  return static_cast<const ZFP_CONTAINER_TYPE*>(self.object)->size_w();
}

static ZFP_SCALAR_TYPE
_t1(CFP_CONTAINER_TYPE, get)(const CFP_CONTAINER_TYPE self, size_t i, size_t j, size_t k, size_t l)
{
  return static_cast<const ZFP_CONTAINER_TYPE*>(self.object)->operator()(i, j, k, l);
}

static void
_t1(CFP_CONTAINER_TYPE, set)(CFP_CONTAINER_TYPE self, size_t i, size_t j, size_t k, size_t l, ZFP_SCALAR_TYPE val)
{
  static_cast<ZFP_CONTAINER_TYPE*>(self.object)->operator()(i, j, k, l) = val;
}

static CFP_REF_TYPE
_t1(CFP_CONTAINER_TYPE, ref)(CFP_CONTAINER_TYPE self, size_t i, size_t j, size_t k, size_t l)
{
  CFP_REF_TYPE r;
  r.container = self.object;
  r.x = i;
  r.y = j;
  r.z = k;
  r.w = l;
  return r;
}

static CFP_PTR_TYPE
_t1(CFP_CONTAINER_TYPE, ptr)(CFP_CONTAINER_TYPE self, size_t i, size_t j, size_t k, size_t l)
{
  CFP_PTR_TYPE p;
  p.container = self.object;
  p.x = i;
  p.y = j;
  p.z = k;
  p.w = l;
  return p;
}

static CFP_REF_TYPE
_t1(CFP_CONTAINER_TYPE, ref_flat)(CFP_CONTAINER_TYPE self, size_t i)
{
  CFP_REF_TYPE r;
  r.container = self.object;
  _t1(CFP_CONTAINER_TYPE, ref_set_offset)(r, i);
  return r;
}

static CFP_PTR_TYPE
_t1(CFP_CONTAINER_TYPE, ptr_flat)(CFP_CONTAINER_TYPE self, size_t i)
{
  CFP_PTR_TYPE p;
  p.container = self.object;
  _t1(CFP_CONTAINER_TYPE, ptr_set_offset)(p, i);
  return p;
}

/* References */
static CFP_PTR_TYPE
_t2(CFP_CONTAINER_TYPE, CFP_REF_TYPE, ptr)(CFP_REF_TYPE self)
{
  CFP_PTR_TYPE p;
  p.container = self.container;
  p.x = self.x;
  p.y = self.y;
  p.z = self.z;
  p.w = self.w;
  return p;
}

static ZFP_SCALAR_TYPE
_t2(CFP_CONTAINER_TYPE, CFP_REF_TYPE, get)(CFP_REF_TYPE self)
{
  return static_cast<const ZFP_CONTAINER_TYPE*>(self.container)->operator()(self.x, self.y, self.z, self.w);
}

static void
_t2(CFP_CONTAINER_TYPE, CFP_REF_TYPE, set)(CFP_REF_TYPE self, ZFP_SCALAR_TYPE val)
{
  static_cast<ZFP_CONTAINER_TYPE*>(self.container)->operator()(self.x, self.y, self.z, self.w) = val;
}

static void
_t2(CFP_CONTAINER_TYPE, CFP_REF_TYPE, copy)(CFP_REF_TYPE self, CFP_REF_TYPE src)
{
  static_cast<ZFP_CONTAINER_TYPE*>(self.container)->operator()(self.x, self.y, self.z, self.w) =
    static_cast<const ZFP_CONTAINER_TYPE*>(src.container)->operator()(src.x, src.y, src.z, src.w);
}

/* Pointers */
static CFP_REF_TYPE
_t2(CFP_CONTAINER_TYPE, CFP_PTR_TYPE, ref)(CFP_PTR_TYPE self)
{
  CFP_REF_TYPE r;
  r.container = self.container;
  r.x = self.x;
  r.y = self.y;
  r.z = self.z;
  r.w = self.w;
  return r;
}

static zfp_bool
_t2(CFP_CONTAINER_TYPE, CFP_PTR_TYPE, lt)(CFP_PTR_TYPE lhs, CFP_PTR_TYPE rhs)
{
  return lhs.container == rhs.container && _t1(CFP_CONTAINER_TYPE, ptr_offset)(lhs) < _t1(CFP_CONTAINER_TYPE, ptr_offset)(rhs);
}

static zfp_bool
_t2(CFP_CONTAINER_TYPE, CFP_PTR_TYPE, gt)(CFP_PTR_TYPE lhs, CFP_PTR_TYPE rhs)
{
  return lhs.container == rhs.container && _t1(CFP_CONTAINER_TYPE, ptr_offset)(lhs) > _t1(CFP_CONTAINER_TYPE, ptr_offset)(rhs);
}

static zfp_bool
_t2(CFP_CONTAINER_TYPE, CFP_PTR_TYPE, leq)(CFP_PTR_TYPE lhs, CFP_PTR_TYPE rhs)
{
  return lhs.container == rhs.container && _t1(CFP_CONTAINER_TYPE, ptr_offset)(lhs) <= _t1(CFP_CONTAINER_TYPE, ptr_offset)(rhs);
}

static zfp_bool
_t2(CFP_CONTAINER_TYPE, CFP_PTR_TYPE, geq)(CFP_PTR_TYPE lhs, CFP_PTR_TYPE rhs)
{
  return lhs.container == rhs.container && _t1(CFP_CONTAINER_TYPE, ptr_offset)(lhs) >= _t1(CFP_CONTAINER_TYPE, ptr_offset)(rhs);
}

static zfp_bool
_t2(CFP_CONTAINER_TYPE, CFP_PTR_TYPE, eq)(CFP_PTR_TYPE lhs, CFP_PTR_TYPE rhs)
{
  return lhs.container == rhs.container &&
         lhs.x == rhs.x &&
         lhs.y == rhs.y &&
         lhs.z == rhs.z &&
         lhs.w == rhs.w;
}

static zfp_bool
_t2(CFP_CONTAINER_TYPE, CFP_PTR_TYPE, neq)(CFP_PTR_TYPE lhs, CFP_PTR_TYPE rhs)
{
  return !_t2(CFP_CONTAINER_TYPE, CFP_PTR_TYPE, eq)(lhs, rhs);
}

static ptrdiff_t
_t2(CFP_CONTAINER_TYPE, CFP_PTR_TYPE, distance)(CFP_PTR_TYPE first, CFP_PTR_TYPE last)
{
  return _t1(CFP_CONTAINER_TYPE, ptr_offset)(last) -_t1(CFP_CONTAINER_TYPE,  ptr_offset)(first);
}

static CFP_PTR_TYPE
_t2(CFP_CONTAINER_TYPE, CFP_PTR_TYPE, next)(CFP_PTR_TYPE p, ptrdiff_t d)
{
  _t1(CFP_CONTAINER_TYPE, ptr_set_offset)(p, _t1(CFP_CONTAINER_TYPE, ptr_offset)(p) + d);
  return p;
}

static CFP_PTR_TYPE
_t2(CFP_CONTAINER_TYPE, CFP_PTR_TYPE, prev)(CFP_PTR_TYPE p, ptrdiff_t d)
{
  return _t2(CFP_CONTAINER_TYPE, CFP_PTR_TYPE, next)(p, -d);
}

static CFP_PTR_TYPE
_t2(CFP_CONTAINER_TYPE, CFP_PTR_TYPE, inc)(CFP_PTR_TYPE p)
{
  return _t2(CFP_CONTAINER_TYPE, CFP_PTR_TYPE, next)(p, +1);
}

static CFP_PTR_TYPE
_t2(CFP_CONTAINER_TYPE, CFP_PTR_TYPE, dec)(CFP_PTR_TYPE p)
{
  return _t2(CFP_CONTAINER_TYPE, CFP_PTR_TYPE, next)(p, -1);
}

static ZFP_SCALAR_TYPE
_t2(CFP_CONTAINER_TYPE, CFP_PTR_TYPE, get)(CFP_PTR_TYPE self)
{
  return static_cast<const ZFP_CONTAINER_TYPE*>(self.container)->operator()(self.x, self.y, self.z, self.w);
}

static ZFP_SCALAR_TYPE
_t2(CFP_CONTAINER_TYPE, CFP_PTR_TYPE, get_at)(CFP_PTR_TYPE self, ptrdiff_t d)
{
  self = _t2(CFP_CONTAINER_TYPE, CFP_PTR_TYPE, next)(self, d);
  return static_cast<const ZFP_CONTAINER_TYPE*>(self.container)->operator()(self.x, self.y, self.z, self.w);
}

static void
_t2(CFP_CONTAINER_TYPE, CFP_PTR_TYPE, set)(CFP_PTR_TYPE self, ZFP_SCALAR_TYPE val)
{
  static_cast<ZFP_CONTAINER_TYPE*>(self.container)->operator()(self.x, self.y, self.z, self.w) = val;
}

static void
_t2(CFP_CONTAINER_TYPE, CFP_PTR_TYPE, set_at)(CFP_PTR_TYPE self, ptrdiff_t d, ZFP_SCALAR_TYPE val)
{
  self = _t2(CFP_CONTAINER_TYPE, CFP_PTR_TYPE, next)(self, d);
  static_cast<ZFP_CONTAINER_TYPE*>(self.container)->operator()(self.x, self.y, self.z, self.w) = val;
}

static CFP_REF_TYPE
_t2(CFP_CONTAINER_TYPE, CFP_PTR_TYPE, ref_at)(CFP_PTR_TYPE self, ptrdiff_t d)
{
  self = _t2(CFP_CONTAINER_TYPE, CFP_PTR_TYPE, next)(self, d);
  CFP_REF_TYPE r;
  r.container = self.container;
  r.x = self.x;
  r.y = self.y;
  r.z = self.z;
  r.w = self.w;
  return r;
}

/* Iterators */
static CFP_ITER_TYPE
_t1(CFP_CONTAINER_TYPE, begin)(CFP_CONTAINER_TYPE self)
{
  CFP_ITER_TYPE it;
  it.container = self.object;
  it.x = 0;
  it.y = 0;
  it.z = 0;
  it.w = 0;
  return it;
}

static CFP_ITER_TYPE
_t1(CFP_CONTAINER_TYPE, end)(CFP_CONTAINER_TYPE self)
{
  CFP_ITER_TYPE it;
  it.container = self.object;
  it.x = 0;
  it.y = 0;
  it.z = 0;
  it.w = static_cast<const ZFP_CONTAINER_TYPE*>(self.object)->size_w();
  return it;
}

static zfp_bool
_t1(CFP_ITER_TYPE, lt)(CFP_ITER_TYPE lhs, CFP_ITER_TYPE rhs)
{
  return lhs.container == rhs.container && _t1(CFP_CONTAINER_TYPE, iter_offset)(lhs) < _t1(CFP_CONTAINER_TYPE, iter_offset)(rhs);
}

static zfp_bool
_t1(CFP_ITER_TYPE, gt)(CFP_ITER_TYPE lhs, CFP_ITER_TYPE rhs)
{
  return lhs.container == rhs.container && _t1(CFP_CONTAINER_TYPE, iter_offset)(lhs) > _t1(CFP_CONTAINER_TYPE, iter_offset)(rhs);
}

static zfp_bool
_t1(CFP_ITER_TYPE, leq)(CFP_ITER_TYPE lhs, CFP_ITER_TYPE rhs)
{
  return lhs.container == rhs.container && _t1(CFP_CONTAINER_TYPE, iter_offset)(lhs) <= _t1(CFP_CONTAINER_TYPE, iter_offset)(rhs);
}

static zfp_bool
_t1(CFP_ITER_TYPE, geq)(CFP_ITER_TYPE lhs, CFP_ITER_TYPE rhs)
{
  return lhs.container == rhs.container && _t1(CFP_CONTAINER_TYPE, iter_offset)(lhs) >= _t1(CFP_CONTAINER_TYPE, iter_offset)(rhs);
}

static zfp_bool
_t1(CFP_ITER_TYPE, eq)(CFP_ITER_TYPE lhs, CFP_ITER_TYPE rhs)
{
  return lhs.container == rhs.container &&
         lhs.x == rhs.x &&
         lhs.y == rhs.y &&
         lhs.z == rhs.z &&
         lhs.w == rhs.w;
}

static zfp_bool
_t1(CFP_ITER_TYPE, neq)(CFP_ITER_TYPE lhs, CFP_ITER_TYPE rhs)
{
  return !_t1(CFP_ITER_TYPE, eq)(lhs, rhs);
}

static ptrdiff_t
_t1(CFP_ITER_TYPE, distance)(CFP_ITER_TYPE first, CFP_ITER_TYPE last)
{
  return _t1(CFP_CONTAINER_TYPE, iter_offset)(last) - _t1(CFP_CONTAINER_TYPE, iter_offset)(first);
}

static CFP_ITER_TYPE
_t1(CFP_ITER_TYPE, next)(CFP_ITER_TYPE it, ptrdiff_t d)
{
  _t1(CFP_CONTAINER_TYPE, iter_set_offset)(it, _t1(CFP_CONTAINER_TYPE, iter_offset)(it) + d);
  return it;
}

static CFP_ITER_TYPE
_t1(CFP_ITER_TYPE, prev)(CFP_ITER_TYPE it, ptrdiff_t d)
{
  _t1(CFP_CONTAINER_TYPE, iter_set_offset)(it, _t1(CFP_CONTAINER_TYPE, iter_offset)(it) - d);
  return it;
}

static CFP_ITER_TYPE
_t1(CFP_ITER_TYPE, inc)(CFP_ITER_TYPE it)
{
  const ZFP_CONTAINER_TYPE* container = static_cast<const ZFP_CONTAINER_TYPE*>(it.container);
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
_t1(CFP_ITER_TYPE, dec)(CFP_ITER_TYPE it)
{
  const ZFP_CONTAINER_TYPE* container = static_cast<const ZFP_CONTAINER_TYPE*>(it.container);
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
_t1(CFP_ITER_TYPE, get)(CFP_ITER_TYPE self)
{
  return static_cast<ZFP_CONTAINER_TYPE*>(self.container)->operator()(self.x, self.y, self.z, self.w);
}

static ZFP_SCALAR_TYPE
_t1(CFP_ITER_TYPE, get_at)(CFP_ITER_TYPE self, ptrdiff_t d)
{
  self = _t1(CFP_ITER_TYPE, next)(self, d);
  return static_cast<ZFP_CONTAINER_TYPE*>(self.container)->operator()(self.x, self.y, self.z, self.w);
}

static void
_t1(CFP_ITER_TYPE, set)(CFP_ITER_TYPE self, ZFP_SCALAR_TYPE val)
{
  static_cast<ZFP_CONTAINER_TYPE*>(self.container)->operator()(self.x, self.y, self.z, self.w) = val;
}

static void
_t1(CFP_ITER_TYPE, set_at)(CFP_ITER_TYPE self, ptrdiff_t d, ZFP_SCALAR_TYPE val)
{
  self = _t1(CFP_ITER_TYPE, next)(self, d);
  static_cast<ZFP_CONTAINER_TYPE*>(self.container)->operator()(self.x, self.y, self.z, self.w) = val;
}

static CFP_REF_TYPE
_t1(CFP_ITER_TYPE, ref)(CFP_ITER_TYPE self)
{
  CFP_CONTAINER_TYPE a;
  a.object = self.container;
  return _t1(CFP_CONTAINER_TYPE, ref)(a, self.x, self.y, self.z, self.w);
}

static CFP_REF_TYPE
_t1(CFP_ITER_TYPE, ref_at)(CFP_ITER_TYPE self, ptrdiff_t d)
{
  self = _t1(CFP_ITER_TYPE, next)(self, d);
  CFP_CONTAINER_TYPE a;
  a.object = self.container;
  return _t1(CFP_CONTAINER_TYPE, ref)(a, self.x, self.y, self.z, self.w);
}

static CFP_PTR_TYPE
_t1(CFP_ITER_TYPE, ptr)(CFP_ITER_TYPE self)
{
  CFP_CONTAINER_TYPE a;
  a.object = self.container;
  return _t1(CFP_CONTAINER_TYPE, ptr)(a, self.x, self.y, self.z, self.w);
}

static CFP_PTR_TYPE
_t1(CFP_ITER_TYPE, ptr_at)(CFP_ITER_TYPE self, ptrdiff_t d)
{
  self = _t1(CFP_ITER_TYPE, next)(self, d);
  CFP_CONTAINER_TYPE a;
  a.object = self.container;
  return _t1(CFP_CONTAINER_TYPE, ptr)(a, self.x, self.y, self.z, self.w);
}

static size_t
_t1(CFP_ITER_TYPE, i)(CFP_ITER_TYPE self)
{
  return self.x;
}

static size_t
_t1(CFP_ITER_TYPE, j)(CFP_ITER_TYPE self)
{
  return self.y;
}

static size_t
_t1(CFP_ITER_TYPE, k)(CFP_ITER_TYPE self)
{
  return self.z;
}

static size_t
_t1(CFP_ITER_TYPE, l)(CFP_ITER_TYPE self)
{
  return self.w;
}
