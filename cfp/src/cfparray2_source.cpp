static CFP_ARRAY_TYPE
_t1(CFP_ARRAY_TYPE, ctor)(uint nx, uint ny, double rate, const ZFP_SCALAR_TYPE * p, size_t csize)
{
  CFP_ARRAY_TYPE a;
  a.object = reinterpret_cast<void*>(new ZFP_ARRAY_TYPE(nx, ny, rate, p, csize));
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

static void
_t1(CFP_ARRAY_TYPE, resize)(CFP_ARRAY_TYPE self, uint nx, uint ny, int clear)
{
  reinterpret_cast<ZFP_ARRAY_TYPE*>(self.object)->resize(nx, ny, clear);
}

static ZFP_SCALAR_TYPE
_t1(CFP_ARRAY_TYPE, get)(CFP_ARRAY_TYPE self, uint i, uint j)
{
  return reinterpret_cast<const ZFP_ARRAY_TYPE*>(self.object)->operator()(i, j);
}

static void
_t1(CFP_ARRAY_TYPE, set)(CFP_ARRAY_TYPE self, uint i, uint j, ZFP_SCALAR_TYPE val)
{
  reinterpret_cast<ZFP_ARRAY_TYPE*>(self.object)->operator()(i, j) = val;
}

static CFP_REF_TYPE
_t1(CFP_ARRAY_TYPE, ref)(CFP_ARRAY_TYPE self, uint i, uint j)
{
  CFP_REF_TYPE r;
  r.i = i;
  r.j = j;
  r.array = self;
  return r;
}

static CFP_REF_TYPE
_t1(CFP_ARRAY_TYPE, flat_ref)(CFP_ARRAY_TYPE self, uint i)
{
  CFP_REF_TYPE r;
  r.i = i % reinterpret_cast<const ZFP_ARRAY_TYPE*>(self.object)->size_x();
  r.j = i / reinterpret_cast<const ZFP_ARRAY_TYPE*>(self.object)->size_x();
  r.array = self;
  return r;
}

static CFP_PTR_TYPE
_t1(CFP_ARRAY_TYPE, ptr)(CFP_ARRAY_TYPE self, uint i, uint j)
{
  CFP_PTR_TYPE p;
  p.reference = _t1(CFP_ARRAY_TYPE, ref)(self, i, j);
  return p;
}

static CFP_PTR_TYPE
_t1(CFP_ARRAY_TYPE, flat_ptr)(CFP_ARRAY_TYPE self, uint i)
{
  CFP_PTR_TYPE p;
  p.reference = _t1(CFP_ARRAY_TYPE, flat_ref)(self, i);
  return p;
}

static CFP_ITER_TYPE
_t1(CFP_ARRAY_TYPE, begin)(CFP_ARRAY_TYPE self)
{
  CFP_ITER_TYPE it;
  it.i = 0;
  it.j = 0;
  it.array = self;
  return it;
}

static CFP_ITER_TYPE
_t1(CFP_ARRAY_TYPE, end)(CFP_ARRAY_TYPE self)
{
  CFP_ITER_TYPE it;
  it.i = 0;
  it.j = reinterpret_cast<const ZFP_ARRAY_TYPE*>(self.object)->size_y();
  it.array = self;
  return it;
}

static ZFP_SCALAR_TYPE
_t2(CFP_ARRAY_TYPE, CFP_REF_TYPE, get)(CFP_REF_TYPE self)
{
  return reinterpret_cast<const ZFP_ARRAY_TYPE*>(self.array.object)->operator()(self.i, self.j);
}

static void
_t2(CFP_ARRAY_TYPE, CFP_REF_TYPE, set)(CFP_REF_TYPE self, ZFP_SCALAR_TYPE val)
{
  reinterpret_cast<ZFP_ARRAY_TYPE*>(self.array.object)->operator()(self.i, self.j) = val;
}

static void
_t2(CFP_ARRAY_TYPE, CFP_REF_TYPE, copy)(CFP_REF_TYPE self, CFP_REF_TYPE src)
{
  reinterpret_cast<ZFP_ARRAY_TYPE*>(self.array.object)->operator()(self.i, self.j) =
    reinterpret_cast<const ZFP_ARRAY_TYPE*>(src.array.object)->operator()(src.i, src.j);
}

static int
_t2(CFP_ARRAY_TYPE, CFP_PTR_TYPE, eq)(CFP_PTR_TYPE self, CFP_PTR_TYPE src)
{
  return self.reference.i == src.reference.i && 
         self.reference.j == src.reference.j &&
         self.reference.array.object == src.reference.array.object;
}

static int
_t2(CFP_ARRAY_TYPE, CFP_PTR_TYPE, diff)(CFP_PTR_TYPE self, CFP_PTR_TYPE src)
{
  return (self.reference.i + self.reference.j * reinterpret_cast<const ZFP_ARRAY_TYPE*>(self.reference.array.object)->size_x()) -
         (src.reference.i + src.reference.j * reinterpret_cast<const ZFP_ARRAY_TYPE*>(src.reference.array.object)->size_x());
}

static CFP_PTR_TYPE
_t2(CFP_ARRAY_TYPE, CFP_PTR_TYPE, shift)(CFP_PTR_TYPE self, int i)
{
  uint idx = (int)(self.reference.i + self.reference.j*reinterpret_cast<const ZFP_ARRAY_TYPE*>(self.reference.array.object)->size_x()) + i;

  self.reference.i = idx % reinterpret_cast<const ZFP_ARRAY_TYPE*>(self.reference.array.object)->size_x();
  self.reference.j = idx / reinterpret_cast<const ZFP_ARRAY_TYPE*>(self.reference.array.object)->size_x();

  return self;  
}

static CFP_PTR_TYPE
_t2(CFP_ARRAY_TYPE, CFP_PTR_TYPE, inc)(CFP_PTR_TYPE self)
{
  uint idx = (int)(self.reference.i + self.reference.j*reinterpret_cast<const ZFP_ARRAY_TYPE*>(self.reference.array.object)->size_x()) + 1;

  self.reference.i = idx % reinterpret_cast<const ZFP_ARRAY_TYPE*>(self.reference.array.object)->size_x();
  self.reference.j = idx / reinterpret_cast<const ZFP_ARRAY_TYPE*>(self.reference.array.object)->size_x();

  return self;  
}

static CFP_PTR_TYPE
_t2(CFP_ARRAY_TYPE, CFP_PTR_TYPE, dec)(CFP_PTR_TYPE self)
{
  uint idx = (int)(self.reference.i + self.reference.j*reinterpret_cast<const ZFP_ARRAY_TYPE*>(self.reference.array.object)->size_x()) - 1;

  self.reference.i = idx % reinterpret_cast<const ZFP_ARRAY_TYPE*>(self.reference.array.object)->size_x();
  self.reference.j = idx / reinterpret_cast<const ZFP_ARRAY_TYPE*>(self.reference.array.object)->size_x();

  return self;  
}

static CFP_REF_TYPE
_t2(CFP_ARRAY_TYPE, CFP_PTR_TYPE, offset_ref)(CFP_PTR_TYPE self, int i)
{
  self = _t2(CFP_ARRAY_TYPE, CFP_PTR_TYPE, shift)(self, i);
  return self.reference;
}

static CFP_REF_TYPE
_t2(CFP_ARRAY_TYPE, CFP_ITER_TYPE, ref)(CFP_ITER_TYPE self)
{
  CFP_REF_TYPE r;
  r.i = self.i;
  r.j = self.j;
  r.array = self.array;
  return r;
}

static CFP_ITER_TYPE
_t2(CFP_ARRAY_TYPE, CFP_ITER_TYPE, inc)(CFP_ITER_TYPE self)
{
  CFP_ITER_TYPE it = self;
  uint nx = reinterpret_cast<const ZFP_ARRAY_TYPE*>(self.array.object)->size_x();
  uint ny = reinterpret_cast<const ZFP_ARRAY_TYPE*>(self.array.object)->size_y();

  it.i++;
  if (!(it.i & 3u) || it.i == nx) {
    it.i = (it.i - 1) & ~3u;
    it.j++;
    if (!(it.j & 3u) || it.j == ny) {
      it.j = (it.j - 1) & ~3u;
      // done with block; advance to next
      if ((it.i += 4) >= nx) {
        it.i = 0;
        if ((it.j += 4) >= ny)
          it.j = ny;
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
