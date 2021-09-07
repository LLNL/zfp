static void
_t1(CFP_CONTAINER_TYPE, dtor)(const CFP_CONTAINER_TYPE self)
{
  delete static_cast<ZFP_CONTAINER_TYPE*>(self.object);
}

static double
_t1(CFP_CONTAINER_TYPE, rate)(CFP_CONTAINER_TYPE self)
{
  return static_cast<const ZFP_CONTAINER_TYPE*>(self.object)->rate();
}

static size_t
_t1(CFP_CONTAINER_TYPE, size)(CFP_CONTAINER_TYPE self)
{
  return static_cast<const ZFP_CONTAINER_TYPE*>(self.object)->size();
}
