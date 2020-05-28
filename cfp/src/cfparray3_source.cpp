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
  r.i = i; 
  r.j = j;
  r.k = k;
  r.array = self;
  return r;
}

static CFP_REF_TYPE
_t1(CFP_ARRAY_TYPE, ref_flat)(CFP_ARRAY_TYPE self, uint i)
{
  CFP_REF_TYPE r;
  r.i = i % reinterpret_cast<const ZFP_ARRAY_TYPE*>(self.object)->size_x();
  r.j = (i / reinterpret_cast<const ZFP_ARRAY_TYPE*>(self.object)->size_x()) % reinterpret_cast<const ZFP_ARRAY_TYPE*>(self.object)->size_y();
  r.k = i / (reinterpret_cast<const ZFP_ARRAY_TYPE*>(self.object)->size_x() * reinterpret_cast<const ZFP_ARRAY_TYPE*>(self.object)->size_y());
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

static CFP_PTR_TYPE
_t1(CFP_ARRAY_TYPE, ptr_flat)(CFP_ARRAY_TYPE self, uint i)
{
  CFP_PTR_TYPE p;
  p.reference = _t1(CFP_ARRAY_TYPE, ref_flat)(self, i);
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
  it.k = reinterpret_cast<const ZFP_ARRAY_TYPE*>(self.object)->size_z();
  it.array = self;
  return it;
}

static ZFP_SCALAR_TYPE
_t2(CFP_ARRAY_TYPE, CFP_REF_TYPE, get)(CFP_REF_TYPE self)
{
  return reinterpret_cast<const ZFP_ARRAY_TYPE*>(self.array.object)->operator()(self.i, self.j, self.k);
}

static void
_t2(CFP_ARRAY_TYPE, CFP_REF_TYPE, set)(CFP_REF_TYPE self, ZFP_SCALAR_TYPE val)
{
  reinterpret_cast<ZFP_ARRAY_TYPE*>(self.array.object)->operator()(self.i, self.j, self.k) = val;
}

static void
_t2(CFP_ARRAY_TYPE, CFP_REF_TYPE, copy)(CFP_REF_TYPE self, CFP_REF_TYPE src)
{
  reinterpret_cast<ZFP_ARRAY_TYPE*>(self.array.object)->operator()(self.i, self.j, self.k) =
    reinterpret_cast<const ZFP_ARRAY_TYPE*>(src.array.object)->operator()(src.i, src.j, src.k);
}

static int
_t2(CFP_ARRAY_TYPE, CFP_PTR_TYPE, lt)(CFP_PTR_TYPE self, CFP_PTR_TYPE src)
{
  uint selfIdx = (int)(self.reference.i + 
                   self.reference.j*reinterpret_cast<const ZFP_ARRAY_TYPE*>(self.reference.array.object)->size_x() + 
                   self.reference.k*reinterpret_cast<const ZFP_ARRAY_TYPE*>(self.reference.array.object)->size_x()*
                    reinterpret_cast<const ZFP_ARRAY_TYPE*>(self.reference.array.object)->size_y());
  uint srcIdx = (int)(src.reference.i + 
                   src.reference.j*reinterpret_cast<const ZFP_ARRAY_TYPE*>(src.reference.array.object)->size_x() + 
                   src.reference.k*reinterpret_cast<const ZFP_ARRAY_TYPE*>(src.reference.array.object)->size_x()*
                    reinterpret_cast<const ZFP_ARRAY_TYPE*>(src.reference.array.object)->size_y());

  return selfIdx < srcIdx && 
         self.reference.array.object == src.reference.array.object;
}

static int
_t2(CFP_ARRAY_TYPE, CFP_PTR_TYPE, gt)(CFP_PTR_TYPE self, CFP_PTR_TYPE src)
{
  uint selfIdx = (int)(self.reference.i + 
                   self.reference.j*reinterpret_cast<const ZFP_ARRAY_TYPE*>(self.reference.array.object)->size_x() + 
                   self.reference.k*reinterpret_cast<const ZFP_ARRAY_TYPE*>(self.reference.array.object)->size_x()*
                    reinterpret_cast<const ZFP_ARRAY_TYPE*>(self.reference.array.object)->size_y());
  uint srcIdx = (int)(src.reference.i + 
                   src.reference.j*reinterpret_cast<const ZFP_ARRAY_TYPE*>(src.reference.array.object)->size_x() + 
                   src.reference.k*reinterpret_cast<const ZFP_ARRAY_TYPE*>(src.reference.array.object)->size_x()*
                    reinterpret_cast<const ZFP_ARRAY_TYPE*>(src.reference.array.object)->size_y());

  return selfIdx > srcIdx && 
         self.reference.array.object == src.reference.array.object;
}

static int
_t2(CFP_ARRAY_TYPE, CFP_PTR_TYPE, leq)(CFP_PTR_TYPE self, CFP_PTR_TYPE src)
{
  uint selfIdx = (int)(self.reference.i + 
                   self.reference.j*reinterpret_cast<const ZFP_ARRAY_TYPE*>(self.reference.array.object)->size_x() + 
                   self.reference.k*reinterpret_cast<const ZFP_ARRAY_TYPE*>(self.reference.array.object)->size_x()*
                    reinterpret_cast<const ZFP_ARRAY_TYPE*>(self.reference.array.object)->size_y());
  uint srcIdx = (int)(src.reference.i + 
                   src.reference.j*reinterpret_cast<const ZFP_ARRAY_TYPE*>(src.reference.array.object)->size_x() + 
                   src.reference.k*reinterpret_cast<const ZFP_ARRAY_TYPE*>(src.reference.array.object)->size_x()*
                    reinterpret_cast<const ZFP_ARRAY_TYPE*>(src.reference.array.object)->size_y());

  return selfIdx <= srcIdx && 
         self.reference.array.object == src.reference.array.object;
}

static int
_t2(CFP_ARRAY_TYPE, CFP_PTR_TYPE, geq)(CFP_PTR_TYPE self, CFP_PTR_TYPE src)
{
  uint selfIdx = (int)(self.reference.i + 
                   self.reference.j*reinterpret_cast<const ZFP_ARRAY_TYPE*>(self.reference.array.object)->size_x() + 
                   self.reference.k*reinterpret_cast<const ZFP_ARRAY_TYPE*>(self.reference.array.object)->size_x()*
                    reinterpret_cast<const ZFP_ARRAY_TYPE*>(self.reference.array.object)->size_y());
  uint srcIdx = (int)(src.reference.i + 
                   src.reference.j*reinterpret_cast<const ZFP_ARRAY_TYPE*>(src.reference.array.object)->size_x() + 
                   src.reference.k*reinterpret_cast<const ZFP_ARRAY_TYPE*>(src.reference.array.object)->size_x()*
                    reinterpret_cast<const ZFP_ARRAY_TYPE*>(src.reference.array.object)->size_y());

  return selfIdx >= srcIdx && 
         self.reference.array.object == src.reference.array.object;
}

static int
_t2(CFP_ARRAY_TYPE, CFP_PTR_TYPE, eq)(CFP_PTR_TYPE self, CFP_PTR_TYPE src)
{
  return self.reference.i == src.reference.i && 
         self.reference.j == src.reference.j &&
         self.reference.k == src.reference.k &&
         self.reference.array.object == src.reference.array.object;
}

static int
_t2(CFP_ARRAY_TYPE, CFP_PTR_TYPE, neq)(CFP_PTR_TYPE self, CFP_PTR_TYPE src)
{
  return self.reference.i != src.reference.i || 
         self.reference.j != src.reference.j ||
         self.reference.k != src.reference.k ||
         self.reference.array.object != src.reference.array.object;
}

static int
_t2(CFP_ARRAY_TYPE, CFP_PTR_TYPE, distance)(CFP_PTR_TYPE self, CFP_PTR_TYPE src)
{
  return (self.reference.i + self.reference.j * reinterpret_cast<const ZFP_ARRAY_TYPE*>(self.reference.array.object)->size_x() +
          self.reference.k * reinterpret_cast<const ZFP_ARRAY_TYPE*>(self.reference.array.object)->size_x() * reinterpret_cast<const ZFP_ARRAY_TYPE*>(self.reference.array.object)->size_y()) -
         (src.reference.i + src.reference.j * reinterpret_cast<const ZFP_ARRAY_TYPE*>(src.reference.array.object)->size_x() + 
          src.reference.k * reinterpret_cast<const ZFP_ARRAY_TYPE*>(src.reference.array.object)->size_x() * reinterpret_cast<const ZFP_ARRAY_TYPE*>(src.reference.array.object)->size_y());
}

static CFP_PTR_TYPE
_t2(CFP_ARRAY_TYPE, CFP_PTR_TYPE, next)(CFP_PTR_TYPE self, ptrdiff_t d)
{
  uint idx = (int)(self.reference.i + 
                   self.reference.j*reinterpret_cast<const ZFP_ARRAY_TYPE*>(self.reference.array.object)->size_x() + 
                   self.reference.k*reinterpret_cast<const ZFP_ARRAY_TYPE*>(self.reference.array.object)->size_x()*reinterpret_cast<const ZFP_ARRAY_TYPE*>(self.reference.array.object)->size_y()) + d;

  self.reference.i = idx % reinterpret_cast<const ZFP_ARRAY_TYPE*>(self.reference.array.object)->size_x();
  self.reference.j = (idx / reinterpret_cast<const ZFP_ARRAY_TYPE*>(self.reference.array.object)->size_x()) % reinterpret_cast<const ZFP_ARRAY_TYPE*>(self.reference.array.object)->size_y();
  self.reference.k = idx / (reinterpret_cast<const ZFP_ARRAY_TYPE*>(self.reference.array.object)->size_x() * reinterpret_cast<const ZFP_ARRAY_TYPE*>(self.reference.array.object)->size_y());

  return self;  
}

static CFP_PTR_TYPE
_t2(CFP_ARRAY_TYPE, CFP_PTR_TYPE, prev)(CFP_PTR_TYPE self, ptrdiff_t d)
{
  uint idx = (int)(self.reference.i + 
                   self.reference.j*reinterpret_cast<const ZFP_ARRAY_TYPE*>(self.reference.array.object)->size_x() + 
                   self.reference.k*reinterpret_cast<const ZFP_ARRAY_TYPE*>(self.reference.array.object)->size_x()*reinterpret_cast<const ZFP_ARRAY_TYPE*>(self.reference.array.object)->size_y()) - d;

  self.reference.i = idx % reinterpret_cast<const ZFP_ARRAY_TYPE*>(self.reference.array.object)->size_x();
  self.reference.j = (idx / reinterpret_cast<const ZFP_ARRAY_TYPE*>(self.reference.array.object)->size_x()) % reinterpret_cast<const ZFP_ARRAY_TYPE*>(self.reference.array.object)->size_y();
  self.reference.k = idx / (reinterpret_cast<const ZFP_ARRAY_TYPE*>(self.reference.array.object)->size_x() * reinterpret_cast<const ZFP_ARRAY_TYPE*>(self.reference.array.object)->size_y());

  return self;  
}

static CFP_PTR_TYPE
_t2(CFP_ARRAY_TYPE, CFP_PTR_TYPE, inc)(CFP_PTR_TYPE self)
{
  uint idx = (int)(self.reference.i + 
                   self.reference.j*reinterpret_cast<const ZFP_ARRAY_TYPE*>(self.reference.array.object)->size_x() + 
                   self.reference.k*reinterpret_cast<const ZFP_ARRAY_TYPE*>(self.reference.array.object)->size_x()*reinterpret_cast<const ZFP_ARRAY_TYPE*>(self.reference.array.object)->size_y()) + 1;

  self.reference.i = idx % reinterpret_cast<const ZFP_ARRAY_TYPE*>(self.reference.array.object)->size_x();
  self.reference.j = (idx / reinterpret_cast<const ZFP_ARRAY_TYPE*>(self.reference.array.object)->size_x()) % reinterpret_cast<const ZFP_ARRAY_TYPE*>(self.reference.array.object)->size_y();
  self.reference.k = idx / (reinterpret_cast<const ZFP_ARRAY_TYPE*>(self.reference.array.object)->size_x() * reinterpret_cast<const ZFP_ARRAY_TYPE*>(self.reference.array.object)->size_y());

  return self;  
}

static CFP_PTR_TYPE
_t2(CFP_ARRAY_TYPE, CFP_PTR_TYPE, dec)(CFP_PTR_TYPE self)
{
  uint idx = (int)(self.reference.i + 
                   self.reference.j*reinterpret_cast<const ZFP_ARRAY_TYPE*>(self.reference.array.object)->size_x() + 
                   self.reference.k*reinterpret_cast<const ZFP_ARRAY_TYPE*>(self.reference.array.object)->size_x()*reinterpret_cast<const ZFP_ARRAY_TYPE*>(self.reference.array.object)->size_y()) - 1;

  self.reference.i = idx % reinterpret_cast<const ZFP_ARRAY_TYPE*>(self.reference.array.object)->size_x();
  self.reference.j = (idx / reinterpret_cast<const ZFP_ARRAY_TYPE*>(self.reference.array.object)->size_x()) % reinterpret_cast<const ZFP_ARRAY_TYPE*>(self.reference.array.object)->size_y();
  self.reference.k = idx / (reinterpret_cast<const ZFP_ARRAY_TYPE*>(self.reference.array.object)->size_x() * reinterpret_cast<const ZFP_ARRAY_TYPE*>(self.reference.array.object)->size_y());

  return self;  
}

static ZFP_SCALAR_TYPE
_t2(CFP_ARRAY_TYPE, CFP_PTR_TYPE, get)(CFP_PTR_TYPE self)
{
  return reinterpret_cast<const ZFP_ARRAY_TYPE*>(self.reference.array.object)->operator()(self.reference.i, self.reference.j, self.reference.k);
}

static ZFP_SCALAR_TYPE
_t2(CFP_ARRAY_TYPE, CFP_PTR_TYPE, get_at)(CFP_PTR_TYPE self, ptrdiff_t d)
{
  self = _t2(CFP_ARRAY_TYPE, CFP_PTR_TYPE, next)(self, d);
  return reinterpret_cast<const ZFP_ARRAY_TYPE*>(self.reference.array.object)->operator()(self.reference.i, self.reference.j, self.reference.k);
}

static void
_t2(CFP_ARRAY_TYPE, CFP_PTR_TYPE, set)(CFP_PTR_TYPE self, ZFP_SCALAR_TYPE val)
{
  reinterpret_cast<ZFP_ARRAY_TYPE*>(self.reference.array.object)->operator()(self.reference.i, self.reference.j, self.reference.k) = val;
}

static void
_t2(CFP_ARRAY_TYPE, CFP_PTR_TYPE, set_at)(CFP_PTR_TYPE self, ZFP_SCALAR_TYPE val, ptrdiff_t d)
{
  self = _t2(CFP_ARRAY_TYPE, CFP_PTR_TYPE, next)(self, d);
  reinterpret_cast<ZFP_ARRAY_TYPE*>(self.reference.array.object)->operator()(self.reference.i, self.reference.j, self.reference.k) = val;
}

static CFP_REF_TYPE
_t2(CFP_ARRAY_TYPE, CFP_ITER_TYPE, ref)(CFP_ITER_TYPE self)
{
  CFP_REF_TYPE r;
  r.i = self.i;
  r.j = self.j;
  r.k = self.k;
  r.array = self.array;
  return r;
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
  uint nx = reinterpret_cast<const ZFP_ARRAY_TYPE*>(self.array.object)->size_x();
  uint ny = reinterpret_cast<const ZFP_ARRAY_TYPE*>(self.array.object)->size_y();
  uint nz = reinterpret_cast<const ZFP_ARRAY_TYPE*>(self.array.object)->size_z();

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

static CFP_ITER_TYPE
_t2(CFP_ARRAY_TYPE, CFP_ITER_TYPE, next)(CFP_ITER_TYPE self, ptrdiff_t d)
{
  uint idx = (int)(self.i + 
                   self.j*reinterpret_cast<const ZFP_ARRAY_TYPE*>(self.array.object)->size_x() + 
                   self.k*reinterpret_cast<const ZFP_ARRAY_TYPE*>(self.array.object)->size_x()*reinterpret_cast<const ZFP_ARRAY_TYPE*>(self.array.object)->size_y()) + d;

  self.i = idx % reinterpret_cast<const ZFP_ARRAY_TYPE*>(self.array.object)->size_x();
  self.j = (idx / reinterpret_cast<const ZFP_ARRAY_TYPE*>(self.array.object)->size_x()) % reinterpret_cast<const ZFP_ARRAY_TYPE*>(self.array.object)->size_y();
  self.k = idx / (reinterpret_cast<const ZFP_ARRAY_TYPE*>(self.array.object)->size_x() * reinterpret_cast<const ZFP_ARRAY_TYPE*>(self.array.object)->size_y());

  return self;  
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
