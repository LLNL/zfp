static void
_t1(CFP_CONTAINER_TYPE, partition)(const CFP_CONTAINER_TYPE self, size_t index, size_t count)
{
  static_cast<ZFP_CONTAINER_TYPE*>(self.object)->partition(index, count);
}

static void
_t1(CFP_CONTAINER_TYPE, flush_cache)(const CFP_CONTAINER_TYPE self)
{
  static_cast<const ZFP_CONTAINER_TYPE*>(self.object)->flush_cache();
}
