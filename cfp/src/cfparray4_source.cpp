// utility function: compute multidimensional index from onedimensional offset
static void
ref_index(CFP_REF_TYPE& self, size_t offset)
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
ref_offset(CFP_REF_TYPE self)
{
  size_t nx = static_cast<const ZFP_ARRAY_TYPE*>(self.array.object)->size_x();
  size_t ny = static_cast<const ZFP_ARRAY_TYPE*>(self.array.object)->size_y();
  size_t nz = static_cast<const ZFP_ARRAY_TYPE*>(self.array.object)->size_z();
  return static_cast<ptrdiff_t>(self.x + nx * (self.y + ny * (self.z + nz * self.w)));
}

static CFP_ARRAY_TYPE
_t1(CFP_ARRAY_TYPE, ctor)(size_t nx, size_t ny, size_t nz, size_t nw, double rate, const ZFP_SCALAR_TYPE * p, size_t csize)
{
  CFP_ARRAY_TYPE a;
  a.object = static_cast<void*>(new ZFP_ARRAY_TYPE(nx, ny, nz, nw, rate, p, csize));
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
  static_cast<ZFP_ARRAY_TYPE*>(self.object)->resize(nx, ny, nz, nw, clear);
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
  ref_index(r, i);
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
_t2(CFP_ARRAY_TYPE, CFP_PTR_TYPE, lt)(CFP_PTR_TYPE self, CFP_PTR_TYPE src)
{
  return self.reference.array.object == src.reference.array.object && ref_offset(self.reference) < ref_offset(src.reference);
}

static zfp_bool
_t2(CFP_ARRAY_TYPE, CFP_PTR_TYPE, gt)(CFP_PTR_TYPE self, CFP_PTR_TYPE src)
{
  return self.reference.array.object == src.reference.array.object && ref_offset(self.reference) > ref_offset(src.reference);
}

static zfp_bool
_t2(CFP_ARRAY_TYPE, CFP_PTR_TYPE, leq)(CFP_PTR_TYPE self, CFP_PTR_TYPE src)
{
  return self.reference.array.object == src.reference.array.object && ref_offset(self.reference) <= ref_offset(src.reference);
}

static zfp_bool
_t2(CFP_ARRAY_TYPE, CFP_PTR_TYPE, geq)(CFP_PTR_TYPE self, CFP_PTR_TYPE src)
{
  return self.reference.array.object == src.reference.array.object && ref_offset(self.reference) >= ref_offset(src.reference);
}

static zfp_bool
_t2(CFP_ARRAY_TYPE, CFP_PTR_TYPE, eq)(CFP_PTR_TYPE self, CFP_PTR_TYPE src)
{
  return self.reference.array.object == src.reference.array.object &&
         self.reference.x == src.reference.x &&
         self.reference.y == src.reference.y &&
         self.reference.z == src.reference.z &&
         self.reference.w == src.reference.w;
}

static zfp_bool
_t2(CFP_ARRAY_TYPE, CFP_PTR_TYPE, neq)(CFP_PTR_TYPE self, CFP_PTR_TYPE src)
{
  return !_t2(CFP_ARRAY_TYPE, CFP_PTR_TYPE, eq)(self, src);
}

static ptrdiff_t
_t2(CFP_ARRAY_TYPE, CFP_PTR_TYPE, distance)(CFP_PTR_TYPE self, CFP_PTR_TYPE src)
{
  return ref_offset(src.reference) - ref_offset(self.reference);
}

static CFP_PTR_TYPE
_t2(CFP_ARRAY_TYPE, CFP_PTR_TYPE, next)(CFP_PTR_TYPE self, ptrdiff_t d)
{
  ref_index(self.reference, ref_offset(self.reference) + d);
  return self;
}

static CFP_PTR_TYPE
_t2(CFP_ARRAY_TYPE, CFP_PTR_TYPE, prev)(CFP_PTR_TYPE self, ptrdiff_t d)
{
  return _t2(CFP_ARRAY_TYPE, CFP_PTR_TYPE, next)(self, -d);
}

static CFP_PTR_TYPE
_t2(CFP_ARRAY_TYPE, CFP_PTR_TYPE, inc)(CFP_PTR_TYPE self)
{
  return _t2(CFP_ARRAY_TYPE, CFP_PTR_TYPE, next)(self, +1);
}

static CFP_PTR_TYPE
_t2(CFP_ARRAY_TYPE, CFP_PTR_TYPE, dec)(CFP_PTR_TYPE self)
{
  return _t2(CFP_ARRAY_TYPE, CFP_PTR_TYPE, next)(self, -1);
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
_t2(CFP_ARRAY_TYPE, CFP_PTR_TYPE, set_at)(CFP_PTR_TYPE self, ZFP_SCALAR_TYPE val, ptrdiff_t d)
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

static CFP_ITER_TYPE
_t2(CFP_ARRAY_TYPE, CFP_ITER_TYPE, inc)(CFP_ITER_TYPE self)
{
  CFP_ITER_TYPE it = self;
  size_t nx = static_cast<const ZFP_ARRAY_TYPE*>(self.array.object)->size_x();
  size_t ny = static_cast<const ZFP_ARRAY_TYPE*>(self.array.object)->size_y();
  size_t nz = static_cast<const ZFP_ARRAY_TYPE*>(self.array.object)->size_z();
  size_t nw = static_cast<const ZFP_ARRAY_TYPE*>(self.array.object)->size_w();

  it.x++;
  if (!(it.x & 3u) || it.x == nx) {
    it.x = (it.x - 1) & ~3u;
    it.y++;
    if (!(it.y & 3u) || it.y == ny) {
      it.y = (it.y - 1) & ~3u;
      it.z++;
      if (!(it.z & 3u) || it.z == nz) {
        it.z = (it.z - 1) & ~3u;
        it.w++;
        if (!(it.w & 3u) || it.w == nw) {
          it.w = (it.w - 1) & ~3u;
          // done with block; advance to next
          if ((it.x += 4) >= nx) {
            it.x = 0;
            if ((it.y += 4) >= ny) {
              it.y = 0;
              if ((it.z += 4) >= nz) {
                it.z = 0;
                if ((it.w += 4) >= nw)
                  it.w = nw;
              }
            }
          }
        }
      }
    }
  }
  return it;
}

static zfp_bool
_t2(CFP_ARRAY_TYPE, CFP_ITER_TYPE, eq)(CFP_ITER_TYPE self, CFP_ITER_TYPE src)
{
  return self.array.object == src.array.object &&
         self.x == src.x &&
         self.y == src.y &&
         self.z == src.z &&
         self.w == src.w;
}

static zfp_bool
_t2(CFP_ARRAY_TYPE, CFP_ITER_TYPE, neq)(CFP_ITER_TYPE self, CFP_ITER_TYPE src)
{
  return !_t2(CFP_ARRAY_TYPE, CFP_ITER_TYPE, eq)(self, src);
}

static void
_t2(CFP_ARRAY_TYPE, CFP_ITER_TYPE, set)(CFP_ITER_TYPE self, ZFP_SCALAR_TYPE val)
{
  static_cast<ZFP_ARRAY_TYPE*>(self.array.object)->operator()(self.x, self.y, self.z, self.w) = val;
}

static ZFP_SCALAR_TYPE
_t2(CFP_ARRAY_TYPE, CFP_ITER_TYPE, get)(CFP_ITER_TYPE self)
{
  return static_cast<ZFP_ARRAY_TYPE*>(self.array.object)->operator()(self.x, self.y, self.z, self.w);
}

static CFP_REF_TYPE
_t2(CFP_ARRAY_TYPE, CFP_ITER_TYPE, ref)(CFP_ITER_TYPE self)
{
  return _t1(CFP_ARRAY_TYPE, ref)(self.array, self.x, self.y, self.z, self.w);
}

static CFP_PTR_TYPE
_t2(CFP_ARRAY_TYPE, CFP_ITER_TYPE, ptr)(CFP_ITER_TYPE self)
{
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
