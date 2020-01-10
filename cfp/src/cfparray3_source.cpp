static CFP_ARRAY_TYPE
_t1(CFP_ARRAY_TYPE, ctor)(uint nx, uint ny, uint nz, double rate, const ZFP_SCALAR_TYPE * p, size_t csize)
{
  CFP_ARRAY_TYPE a;
  a.object = reinterpret_cast<void*>(new ZFP_ARRAY_TYPE(nx, ny, nz, rate, p, csize));
  return a;
}

static uint
_t1(CFP_ARRAY_TYPE, size_x)(CFP_ARRAY_TYPE self)
{
  return reinterpret_cast<const ZFP_ARRAY_TYPE*>(self.object)->size_x();
}

static uint
_t1(CFP_ARRAY_TYPE, size_y)(CFP_ARRAY_TYPE self)
{
  return reinterpret_cast<const ZFP_ARRAY_TYPE*>(self.object)->size_y();
}

static uint
_t1(CFP_ARRAY_TYPE, size_z)(CFP_ARRAY_TYPE self)
{
  return reinterpret_cast<const ZFP_ARRAY_TYPE*>(self.object)->size_z();
}

static void
_t1(CFP_ARRAY_TYPE, resize)(CFP_ARRAY_TYPE self, uint nx, uint ny, uint nz, int clear)
{
  reinterpret_cast<ZFP_ARRAY_TYPE*>(self.object)->resize(nx, ny, nz, clear);
}

static ZFP_SCALAR_TYPE
_t1(CFP_ARRAY_TYPE, get)(CFP_ARRAY_TYPE self, uint i, uint j, uint k)
{
  return reinterpret_cast<const ZFP_ARRAY_TYPE*>(self.object)->operator()(i, j, k);
}

static void
_t1(CFP_ARRAY_TYPE, set)(CFP_ARRAY_TYPE self, uint i, uint j, uint k, ZFP_SCALAR_TYPE val)
{
  reinterpret_cast<ZFP_ARRAY_TYPE*>(self.object)->operator()(i, j, k) = val;
}

static CFP_REF_TYPE
_t1(CFP_ARRAY_TYPE, ref)(CFP_ARRAY_TYPE self, uint i, uint j, uint k)
{
  CFP_REF_TYPE r;
  r.idx = i + 
          j * reinterpret_cast<const ZFP_ARRAY_TYPE*>(self.object)->size_x() + 
          k * reinterpret_cast<const ZFP_ARRAY_TYPE*>(self.object)->size_x() * 
              reinterpret_cast<const ZFP_ARRAY_TYPE*>(self.object)->size_y();
  r.array = self;
  return r;
}

static CFP_PTR_TYPE
_t1(CFP_ARRAY_TYPE, ptr)(CFP_ARRAY_TYPE self, uint i, uint j, uint k)
{
  CFP_PTR_TYPE p;
  p.reference = _t1(CFP_ARRAY_TYPE, ref)(self, i, j, k);
  return p;
}

static CFP_ITER_TYPE
_t1(CFP_ARRAY_TYPE, begin)(CFP_ARRAY_TYPE self)
{
  CFP_ITER_TYPE it;
  it.i = 0;
  it.j = 0;
  it.k = 0;
  it.array = self;
  return it;
}

static CFP_ITER_TYPE
_t1(CFP_ARRAY_TYPE, end)(CFP_ARRAY_TYPE self)
{
  CFP_ITER_TYPE it;
  it.i = 0;
  it.j = 0;
  it.k = reinterpret_cast<ZFP_ARRAY_TYPE*>(self.object)->size_z();
  it.array = self;
  return it;
}

static CFP_REF_TYPE
_t2(CFP_ARRAY_TYPE, CFP_ITER_TYPE, ref)(CFP_ITER_TYPE self)
{
  CFP_REF_TYPE r;
  r.idx = self.i + 
          self.j * reinterpret_cast<const ZFP_ARRAY_TYPE*>(self.array.object)->size_x() + 
          self.k * reinterpret_cast<const ZFP_ARRAY_TYPE*>(self.array.object)->size_x() * 
                   reinterpret_cast<const ZFP_ARRAY_TYPE*>(self.array.object)->size_y();
  r.array = self.array;
  return r;
}

static CFP_ITER_TYPE
_t2(CFP_ARRAY_TYPE, CFP_ITER_TYPE, inc)(CFP_ITER_TYPE self)
{
  uint nx = reinterpret_cast<const ZFP_ARRAY_TYPE*>(self.array.object)->size_x();
  uint ny = reinterpret_cast<const ZFP_ARRAY_TYPE*>(self.array.object)->size_y();
  uint nz = reinterpret_cast<const ZFP_ARRAY_TYPE*>(self.array.object)->size_z();

  CFP_ITER_TYPE it = self;

  it.i++;
  if (!(it.i & 3u) || it.i == nx) {
    it.i = (it.i - 1) & ~3u;
    it.j++;
    if (!(it.j & 3u) || it.j == ny) {
      it.j = (it.j - 1) & ~3u;
      it.k++;
      if (!(it.k & 3u) || it.k == nz) {
        it.k = (it.k - 1) & ~3u;
        // done with block; advance to next
        if ((it.i += 4) >= nx) {
          it.i = 0;
          if ((it.j += 4) >= ny) {
            it.j = 0;
            if ((it.k += 4) >= nz)
              it.k = nz;
          }
        }
      }
    }
  }
  return it;
}

static int
_t2(CFP_ARRAY_TYPE, CFP_ITER_TYPE, eq)(CFP_ITER_TYPE self, CFP_ITER_TYPE src)
{
  return (self.i == src.i &&
          self.j == src.j &&
          self.k == src.k &&
          self.array.object == src.array.object);
}

static uint
_t2(CFP_ARRAY_TYPE, CFP_ITER_TYPE, i)(CFP_ITER_TYPE self)
{
  return self.i;
}

static uint
_t2(CFP_ARRAY_TYPE, CFP_ITER_TYPE, j)(CFP_ITER_TYPE self)
{
  return self.j;
}

static uint
_t2(CFP_ARRAY_TYPE, CFP_ITER_TYPE, k)(CFP_ITER_TYPE self)
{
  return self.k;
}
