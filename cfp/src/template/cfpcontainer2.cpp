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

static ZFP_SCALAR_TYPE
_t1(CFP_CONTAINER_TYPE, get)(const CFP_CONTAINER_TYPE self, size_t i, size_t j)
{
  return static_cast<const ZFP_CONTAINER_TYPE*>(self.object)->operator()(i, j);
}

static void
_t1(CFP_CONTAINER_TYPE, set)(CFP_CONTAINER_TYPE self, size_t i, size_t j, ZFP_SCALAR_TYPE val)
{
  static_cast<ZFP_CONTAINER_TYPE*>(self.object)->operator()(i, j) = val;
}

/* Iterators */
  /* utility function: compute one-dimensional offset from multi-dimensional index */
static ptrdiff_t
_t1(CFP_CONTAINER_TYPE, iter_offset)(const CFP_ITER_TYPE& self)
{
  const ZFP_CONTAINER_TYPE* container = static_cast<const ZFP_CONTAINER_TYPE*>(self.container);
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

  /* utility function: compute multi-dimensional index from one-dimensional offset */
static void
_t1(CFP_CONTAINER_TYPE, iter_set_offset)(CFP_ITER_TYPE& self, size_t offset)
{
  const ZFP_CONTAINER_TYPE* container = static_cast<const ZFP_CONTAINER_TYPE*>(self.container);
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

static CFP_ITER_TYPE
_t1(CFP_CONTAINER_TYPE, begin)(CFP_CONTAINER_TYPE self)
{
  CFP_ITER_TYPE it;
  it.container = self.object;
  it.x = 0;
  it.y = 0;
  return it;
}

static CFP_ITER_TYPE
_t1(CFP_CONTAINER_TYPE, end)(CFP_CONTAINER_TYPE self)
{
  CFP_ITER_TYPE it;
  it.container = self.object;
  it.x = 0;
  it.y = static_cast<const ZFP_CONTAINER_TYPE*>(self.object)->size_y();
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
         lhs.y == rhs.y;
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
_t1(CFP_ITER_TYPE, dec)(CFP_ITER_TYPE it)
{
  const ZFP_CONTAINER_TYPE* container = static_cast<const ZFP_CONTAINER_TYPE*>(it.container);
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
_t1(CFP_ITER_TYPE, get)(CFP_ITER_TYPE self)
{
  return static_cast<const ZFP_CONTAINER_TYPE*>(self.container)->operator()(self.x, self.y);
}

static ZFP_SCALAR_TYPE
_t1(CFP_ITER_TYPE, get_at)(CFP_ITER_TYPE self, ptrdiff_t d)
{
  self = _t1(CFP_ITER_TYPE, next)(self, d);
  return static_cast<const ZFP_CONTAINER_TYPE*>(self.container)->operator()(self.x, self.y);
}

static void
_t1(CFP_ITER_TYPE, set)(CFP_ITER_TYPE self, ZFP_SCALAR_TYPE val)
{
  static_cast<ZFP_CONTAINER_TYPE*>(self.container)->operator()(self.x, self.y) = val;
}

static void
_t1(CFP_ITER_TYPE, set_at)(CFP_ITER_TYPE self, ptrdiff_t d, ZFP_SCALAR_TYPE val)
{
  self = _t1(CFP_ITER_TYPE, next)(self, d);
  static_cast<ZFP_CONTAINER_TYPE*>(self.container)->operator()(self.x, self.y) = val;
}

static CFP_REF_TYPE
_t1(CFP_ITER_TYPE, ref)(CFP_ITER_TYPE self)
{
  CFP_CONTAINER_TYPE a;
  a.object = self.container;
  //return _t1(CFP_CONTAINER_TYPE, ref)(a, self.x, self.y);
}

static CFP_REF_TYPE
_t1(CFP_ITER_TYPE, ref_at)(CFP_ITER_TYPE self, ptrdiff_t d)
{
  self = _t1(CFP_ITER_TYPE, next)(self, d);
  CFP_CONTAINER_TYPE a;
  a.object = self.container;
  //return _t1(CFP_CONTAINER_TYPE, ref)(a, self.x, self.y);
}

static CFP_PTR_TYPE
_t1(CFP_ITER_TYPE, ptr)(CFP_ITER_TYPE self)
{
  CFP_CONTAINER_TYPE a;
  a.object = self.container;
  //return _t1(CFP_CONTAINER_TYPE, ptr)(a, self.x, self.y);
}

static CFP_PTR_TYPE
_t1(CFP_ITER_TYPE, ptr_at)(CFP_ITER_TYPE self, ptrdiff_t d)
{
  self = _t1(CFP_ITER_TYPE, next)(self, d);
  CFP_CONTAINER_TYPE a;
  a.object = self.container;
  //return _t1(CFP_CONTAINER_TYPE, ptr)(a, self.x, self.y);
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
