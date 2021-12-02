static ZFP_SCALAR_TYPE
_t1(CFP_CONTAINER_TYPE, get_flat)(const CFP_CONTAINER_TYPE self, size_t i)
{
  return static_cast<const ZFP_CONTAINER_TYPE*>(self.object)->operator[](i);
}

static void
_t1(CFP_CONTAINER_TYPE, set_flat)(CFP_CONTAINER_TYPE self, size_t i, ZFP_SCALAR_TYPE val)
{
  static_cast<ZFP_CONTAINER_TYPE*>(self.object)->operator[](i) = val;
}
