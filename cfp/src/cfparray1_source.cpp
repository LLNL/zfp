static CFP_ARRAY_TYPE
_t1(CFP_ARRAY_TYPE, ctor)(uint n, double rate, const ZFP_SCALAR_TYPE * p, size_t csize)
{
  CFP_ARRAY_TYPE a;
  a.object = reinterpret_cast<void*>(new ZFP_ARRAY_TYPE(n, rate, p, csize));
  return a;
}

static void
_t1(CFP_ARRAY_TYPE, resize)(CFP_ARRAY_TYPE self, uint n, int clear)
{
  reinterpret_cast<ZFP_ARRAY_TYPE*>(self.object)->resize(n, clear);
}

static ZFP_SCALAR_TYPE
_t1(CFP_ARRAY_TYPE, get)(CFP_ARRAY_TYPE self, uint i)
{
  return reinterpret_cast<const ZFP_ARRAY_TYPE*>(self.object)->operator()(i);
}

static void
_t1(CFP_ARRAY_TYPE, set)(CFP_ARRAY_TYPE self, uint i, ZFP_SCALAR_TYPE val)
{
  reinterpret_cast<ZFP_ARRAY_TYPE*>(self.object)->operator()(i) = val;
}

static CFP_ITER_TYPE
_t1(CFP_ARRAY_TYPE, begin)(CFP_ARRAY_TYPE self)
{
  CFP_ITER_TYPE it;
  it.i = 0;
  it.array = self;
  return it;
}

static CFP_ITER_TYPE
_t1(CFP_ARRAY_TYPE, end)(CFP_ARRAY_TYPE self)
{
  CFP_ITER_TYPE it;
  it.i = reinterpret_cast<const ZFP_ARRAY_TYPE*>(self.object)->size_x();
  it.array = self;
  return it;
}

static CFP_REF_TYPE
_t1(CFP_ARRAY_TYPE, ref)(CFP_ARRAY_TYPE self, uint i)
{
  CFP_REF_TYPE r;
  r.i = i;
  r.array = self;
  return r;
}

static CFP_REF_TYPE
_t1(CFP_ARRAY_TYPE, flat_ref)(CFP_ARRAY_TYPE self, uint i)
{
  CFP_REF_TYPE r;
  r.i = i;
  r.array = self;
  return r;
}

static CFP_PTR_TYPE
_t1(CFP_ARRAY_TYPE, ptr)(CFP_ARRAY_TYPE self, uint i)
{
  CFP_PTR_TYPE p;
  p.reference = _t1(CFP_ARRAY_TYPE, ref)(self, i);
  return p;
}

static CFP_PTR_TYPE
_t1(CFP_ARRAY_TYPE, flat_ptr)(CFP_ARRAY_TYPE self, uint i)
{
  CFP_PTR_TYPE p;
  p.reference = _t1(CFP_ARRAY_TYPE, flat_ref)(self, i);
  return p;
}

static ZFP_SCALAR_TYPE
_t2(CFP_ARRAY_TYPE, CFP_REF_TYPE, get)(CFP_REF_TYPE self)
{
  return reinterpret_cast<const ZFP_ARRAY_TYPE*>(self.array.object)->operator()(self.i);
}

static void
_t2(CFP_ARRAY_TYPE, CFP_REF_TYPE, set)(CFP_REF_TYPE self, ZFP_SCALAR_TYPE val)
{
  reinterpret_cast<ZFP_ARRAY_TYPE*>(self.array.object)->operator()(self.i) = val;
}

static void
_t2(CFP_ARRAY_TYPE, CFP_REF_TYPE, copy)(CFP_REF_TYPE self, CFP_REF_TYPE src)
{
  reinterpret_cast<ZFP_ARRAY_TYPE*>(self.array.object)->operator()(self.i) =
    reinterpret_cast<const ZFP_ARRAY_TYPE*>(src.array.object)->operator()(src.i);
}

static int
_t2(CFP_ARRAY_TYPE, CFP_PTR_TYPE, eq)(CFP_PTR_TYPE self, CFP_PTR_TYPE src)
{
  return self.reference.i == src.reference.i && 
         self.reference.array.object == src.reference.array.object;
}

static int
_t2(CFP_ARRAY_TYPE, CFP_PTR_TYPE, diff)(CFP_PTR_TYPE self, CFP_PTR_TYPE src)
{
   return self.reference.i - src.reference.i;
}

static CFP_PTR_TYPE
_t2(CFP_ARRAY_TYPE, CFP_PTR_TYPE, shift)(CFP_PTR_TYPE self, int i)
{
  self.reference.i += i;
  return self;  
}

static CFP_PTR_TYPE
_t2(CFP_ARRAY_TYPE, CFP_PTR_TYPE, inc)(CFP_PTR_TYPE self)
{
  self.reference.i++;
  return self;  
}

static CFP_PTR_TYPE
_t2(CFP_ARRAY_TYPE, CFP_PTR_TYPE, dec)(CFP_PTR_TYPE self)
{
  self.reference.i--;
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
  r.array = self.array;
  return r;
}

static CFP_REF_TYPE
_t2(CFP_ARRAY_TYPE, CFP_ITER_TYPE, offset_ref)(CFP_ITER_TYPE self, int i)
{
  CFP_REF_TYPE r;
  r.i = self.i + i;
  r.array = self.array;
  return r;
}

static CFP_ITER_TYPE
_t2(CFP_ARRAY_TYPE, CFP_ITER_TYPE, inc)(CFP_ITER_TYPE self)
{
  self.i++;
  return self;
}

static CFP_ITER_TYPE
_t2(CFP_ARRAY_TYPE, CFP_ITER_TYPE, dec)(CFP_ITER_TYPE self)
{
  self.i--;
  return self;
}

static CFP_ITER_TYPE
_t2(CFP_ARRAY_TYPE, CFP_ITER_TYPE, shift)(CFP_ITER_TYPE self, int i)
{
  self.i += i;
  return self;
}

static int
_t2(CFP_ARRAY_TYPE, CFP_ITER_TYPE, diff)(CFP_ITER_TYPE self, CFP_ITER_TYPE src)
{
   return self.i - src.i;
}

static int
_t2(CFP_ARRAY_TYPE, CFP_ITER_TYPE, lt)(CFP_ITER_TYPE self, CFP_ITER_TYPE src)
{
  return self.i < src.i;
}

static int
_t2(CFP_ARRAY_TYPE, CFP_ITER_TYPE, gt)(CFP_ITER_TYPE self, CFP_ITER_TYPE src)
{
  return self.i > src.i;
}

static int
_t2(CFP_ARRAY_TYPE, CFP_ITER_TYPE, leq)(CFP_ITER_TYPE self, CFP_ITER_TYPE src)
{
  return self.i <= src.i;
}

static int
_t2(CFP_ARRAY_TYPE, CFP_ITER_TYPE, geq)(CFP_ITER_TYPE self, CFP_ITER_TYPE src)
{
  return self.i >= src.i;
}

static int
_t2(CFP_ARRAY_TYPE, CFP_ITER_TYPE, eq)(CFP_ITER_TYPE self, CFP_ITER_TYPE src)
{
  return (self.i == src.i &&
          self.array.object == src.array.object);
}

static uint
_t2(CFP_ARRAY_TYPE, CFP_ITER_TYPE, i)(CFP_ITER_TYPE self)
{
  return self.i;
}

