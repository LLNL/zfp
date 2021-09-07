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

static ZFP_SCALAR_TYPE
_t1(CFP_CONTAINER_TYPE, get)(const CFP_CONTAINER_TYPE self, size_t i, size_t j, size_t k)
{
  return static_cast<const ZFP_CONTAINER_TYPE*>(self.object)->operator()(i, j, k);
}

static void
_t1(CFP_CONTAINER_TYPE, set)(CFP_CONTAINER_TYPE self, size_t i, size_t j, size_t k, ZFP_SCALAR_TYPE val)
{
  static_cast<ZFP_CONTAINER_TYPE*>(self.object)->operator()(i, j, k) = val;
}
