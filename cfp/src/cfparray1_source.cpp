static CFP_ARRAY_TYPE *
_t1(CFP_ARRAY_TYPE, ctor)(uint n, double rate, const ZFP_SCALAR_TYPE * p, size_t csize)
{
  return reinterpret_cast<CFP_ARRAY_TYPE *>(new ZFP_ARRAY_TYPE(n, rate, p, csize));
}

static void
_t1(CFP_ARRAY_TYPE, resize)(CFP_ARRAY_TYPE * self, uint n, int clear)
{
  reinterpret_cast<ZFP_ARRAY_TYPE *>(self)->resize(n, clear);
}
