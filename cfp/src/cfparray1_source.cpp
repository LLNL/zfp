static CFP_ARRAY_TYPE
_t1(CFP_ARRAY_TYPE, ctor)(size_t n, double rate, const ZFP_SCALAR_TYPE * p, size_t csize)
{
  CFP_ARRAY_TYPE a;
  a.object = static_cast<void*>(new ZFP_ARRAY_TYPE(n, rate, p, csize));
  return a;
}

static void
_t1(CFP_ARRAY_TYPE, resize)(CFP_ARRAY_TYPE self, size_t n, int clear)
{
  static_cast<ZFP_ARRAY_TYPE*>(self.object)->resize(n, clear);
}

static ZFP_SCALAR_TYPE
_t1(CFP_ARRAY_TYPE, get)(CFP_ARRAY_TYPE self, size_t i)
{
  return static_cast<const ZFP_ARRAY_TYPE*>(self.object)->operator()(i);
}

static void
_t1(CFP_ARRAY_TYPE, set)(CFP_ARRAY_TYPE self, size_t i, ZFP_SCALAR_TYPE val)
{
  static_cast<ZFP_ARRAY_TYPE*>(self.object)->operator()(i) = val;
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
  it.i = static_cast<const ZFP_ARRAY_TYPE*>(self.object)->size_x();
  it.array = self;
  return it;
}

static CFP_REF_TYPE
_t1(CFP_ARRAY_TYPE, ref)(CFP_ARRAY_TYPE self, size_t i)
{
  CFP_REF_TYPE r;
  r.i = i;
  r.array = self;
  return r;
}

static CFP_REF_TYPE
_t1(CFP_ARRAY_TYPE, ref_flat)(CFP_ARRAY_TYPE self, size_t i)
{
  CFP_REF_TYPE r;
  r.i = i;
  r.array = self;
  return r;
}

static CFP_PTR_TYPE
_t1(CFP_ARRAY_TYPE, ptr)(CFP_ARRAY_TYPE self, size_t i)
{
  CFP_PTR_TYPE p;
  p.reference = _t1(CFP_ARRAY_TYPE, ref)(self, i);
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
  return static_cast<const ZFP_ARRAY_TYPE*>(self.array.object)->operator()(self.i);
}

static void
_t2(CFP_ARRAY_TYPE, CFP_REF_TYPE, set)(CFP_REF_TYPE self, ZFP_SCALAR_TYPE val)
{
  static_cast<ZFP_ARRAY_TYPE*>(self.array.object)->operator()(self.i) = val;
}

static void
_t2(CFP_ARRAY_TYPE, CFP_REF_TYPE, copy)(CFP_REF_TYPE self, CFP_REF_TYPE src)
{
  static_cast<ZFP_ARRAY_TYPE*>(self.array.object)->operator()(self.i) =
    static_cast<const ZFP_ARRAY_TYPE*>(src.array.object)->operator()(src.i);
}

static CFP_PTR_TYPE
_t2(CFP_ARRAY_TYPE, CFP_PTR_TYPE, next)(CFP_PTR_TYPE self, ptrdiff_t d)
{
  self.reference.i += d;
  return self;  
}

static ZFP_SCALAR_TYPE
_t2(CFP_ARRAY_TYPE, CFP_PTR_TYPE, get)(CFP_PTR_TYPE self)
{
  return static_cast<const ZFP_ARRAY_TYPE*>(self.reference.array.object)->operator()(self.reference.i);
}

static ZFP_SCALAR_TYPE
_t2(CFP_ARRAY_TYPE, CFP_PTR_TYPE, get_at)(CFP_PTR_TYPE self, ptrdiff_t d)
{
  self = _t2(CFP_ARRAY_TYPE, CFP_PTR_TYPE, next)(self, d);
  return static_cast<const ZFP_ARRAY_TYPE*>(self.reference.array.object)->operator()(self.reference.i);
}

static void
_t2(CFP_ARRAY_TYPE, CFP_PTR_TYPE, set)(CFP_PTR_TYPE self, ZFP_SCALAR_TYPE val)
{
  static_cast<ZFP_ARRAY_TYPE*>(self.reference.array.object)->operator()(self.reference.i) = val;
}

static CFP_ITER_TYPE
_t2(CFP_ARRAY_TYPE, CFP_ITER_TYPE, next)(CFP_ITER_TYPE self, ptrdiff_t d)
{
  self.i += d;
  return self;
}

static void
_t2(CFP_ARRAY_TYPE, CFP_PTR_TYPE, set_at)(CFP_PTR_TYPE self, ZFP_SCALAR_TYPE val, ptrdiff_t d)
{
  self = _t2(CFP_ARRAY_TYPE, CFP_PTR_TYPE, next)(self, d);
  static_cast<ZFP_ARRAY_TYPE*>(self.reference.array.object)->operator()(self.reference.i) = val;
}

static CFP_REF_TYPE
_t2(CFP_ARRAY_TYPE, CFP_PTR_TYPE, ref_at)(CFP_PTR_TYPE self, ptrdiff_t d)
{
  self = _t2(CFP_ARRAY_TYPE, CFP_PTR_TYPE, next)(self, d);
  return self.reference;
}

static int
_t2(CFP_ARRAY_TYPE, CFP_PTR_TYPE, lt)(CFP_PTR_TYPE self, CFP_PTR_TYPE src)
{
  return self.reference.i < src.reference.i && 
         self.reference.array.object == src.reference.array.object;
}

static int
_t2(CFP_ARRAY_TYPE, CFP_PTR_TYPE, gt)(CFP_PTR_TYPE self, CFP_PTR_TYPE src)
{
  return self.reference.i > src.reference.i && 
         self.reference.array.object == src.reference.array.object;
}

static int
_t2(CFP_ARRAY_TYPE, CFP_PTR_TYPE, leq)(CFP_PTR_TYPE self, CFP_PTR_TYPE src)
{
  return self.reference.i <= src.reference.i && 
         self.reference.array.object == src.reference.array.object;
}

static int
_t2(CFP_ARRAY_TYPE, CFP_PTR_TYPE, geq)(CFP_PTR_TYPE self, CFP_PTR_TYPE src)
{
  return self.reference.i >= src.reference.i && 
         self.reference.array.object == src.reference.array.object;
}

static int
_t2(CFP_ARRAY_TYPE, CFP_PTR_TYPE, eq)(CFP_PTR_TYPE self, CFP_PTR_TYPE src)
{
  return self.reference.i == src.reference.i && 
         self.reference.array.object == src.reference.array.object;
}

static int
_t2(CFP_ARRAY_TYPE, CFP_PTR_TYPE, neq)(CFP_PTR_TYPE self, CFP_PTR_TYPE src)
{
  return self.reference.i != src.reference.i || 
         self.reference.array.object != src.reference.array.object;
}

static ptrdiff_t
_t2(CFP_ARRAY_TYPE, CFP_PTR_TYPE, distance)(CFP_PTR_TYPE self, CFP_PTR_TYPE src)
{
   return self.reference.i - src.reference.i;
}

static CFP_PTR_TYPE
_t2(CFP_ARRAY_TYPE, CFP_PTR_TYPE, prev)(CFP_PTR_TYPE self, ptrdiff_t d)
{
  self.reference.i -= d;
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

static void
_t2(CFP_ARRAY_TYPE, CFP_ITER_TYPE, set)(CFP_ITER_TYPE self, ZFP_SCALAR_TYPE val)
{
  static_cast<ZFP_ARRAY_TYPE*>(self.array.object)->operator()(self.i) = val;
}

static void
_t2(CFP_ARRAY_TYPE, CFP_ITER_TYPE, set_at)(CFP_ITER_TYPE self, ZFP_SCALAR_TYPE val, ptrdiff_t d)
{
  self = _t2(CFP_ARRAY_TYPE, CFP_ITER_TYPE, next)(self, d);
  static_cast<ZFP_ARRAY_TYPE*>(self.array.object)->operator()(self.i) = val;
}

static ZFP_SCALAR_TYPE
_t2(CFP_ARRAY_TYPE, CFP_ITER_TYPE, get)(CFP_ITER_TYPE self)
{
  return static_cast<const ZFP_ARRAY_TYPE*>(self.array.object)->operator()(self.i);
}

static ZFP_SCALAR_TYPE
_t2(CFP_ARRAY_TYPE, CFP_ITER_TYPE, get_at)(CFP_ITER_TYPE self, ptrdiff_t d)
{
  self = _t2(CFP_ARRAY_TYPE, CFP_ITER_TYPE, next)(self, d);
  return static_cast<const ZFP_ARRAY_TYPE*>(self.array.object)->operator()(self.i);
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
_t2(CFP_ARRAY_TYPE, CFP_ITER_TYPE, ref_at)(CFP_ITER_TYPE self, ptrdiff_t d)
{
  CFP_REF_TYPE r;
  r.i = self.i + d;
  r.array = self.array;
  return r;
}

static CFP_PTR_TYPE
_t2(CFP_ARRAY_TYPE, CFP_ITER_TYPE, ptr)(CFP_ITER_TYPE self)
{
  CFP_PTR_TYPE p;
  p.reference = _t1(CFP_ARRAY_TYPE, ref)(self.array, self.i);
  return p;
}

static CFP_PTR_TYPE
_t2(CFP_ARRAY_TYPE, CFP_ITER_TYPE, ptr_at)(CFP_ITER_TYPE self, ptrdiff_t d)
{
  self = _t2(CFP_ARRAY_TYPE, CFP_ITER_TYPE, next)(self, d);
  CFP_PTR_TYPE p;
  p.reference = _t1(CFP_ARRAY_TYPE, ref)(self.array, self.i);
  return p;
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
_t2(CFP_ARRAY_TYPE, CFP_ITER_TYPE, prev)(CFP_ITER_TYPE self, ptrdiff_t d)
{
  self.i -= d;
  return self;
}

static ptrdiff_t
_t2(CFP_ARRAY_TYPE, CFP_ITER_TYPE, distance)(CFP_ITER_TYPE self, CFP_ITER_TYPE src)
{
   return self.i - src.i;
}

static int
_t2(CFP_ARRAY_TYPE, CFP_ITER_TYPE, lt)(CFP_ITER_TYPE self, CFP_ITER_TYPE src)
{
  return self.i < src.i && 
         self.array.object == src.array.object;
}

static int
_t2(CFP_ARRAY_TYPE, CFP_ITER_TYPE, gt)(CFP_ITER_TYPE self, CFP_ITER_TYPE src)
{
  return self.i > src.i && 
         self.array.object == src.array.object;
}

static int
_t2(CFP_ARRAY_TYPE, CFP_ITER_TYPE, leq)(CFP_ITER_TYPE self, CFP_ITER_TYPE src)
{
  return self.i <= src.i && 
         self.array.object == src.array.object;
}

static int
_t2(CFP_ARRAY_TYPE, CFP_ITER_TYPE, geq)(CFP_ITER_TYPE self, CFP_ITER_TYPE src)
{
  return self.i >= src.i && 
         self.array.object == src.array.object;
}

static int
_t2(CFP_ARRAY_TYPE, CFP_ITER_TYPE, eq)(CFP_ITER_TYPE self, CFP_ITER_TYPE src)
{
  return (self.i == src.i &&
          self.array.object == src.array.object);
}

static int
_t2(CFP_ARRAY_TYPE, CFP_ITER_TYPE, neq)(CFP_ITER_TYPE self, CFP_ITER_TYPE src)
{
  return (self.i != src.i ||
          self.array.object != src.array.object);
}

static size_t
_t2(CFP_ARRAY_TYPE, CFP_ITER_TYPE, i)(CFP_ITER_TYPE self)
{
  return self.i;
}
