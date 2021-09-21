static size_t
_t1(CFP_CONTAINER_TYPE, index)(const CFP_CONTAINER_TYPE self, size_t i, size_t j)
{
  return static_cast<ZFP_CONTAINER_TYPE*>(self.object)->index(i, j);
}

static void
_t1(CFP_CONTAINER_TYPE, ij)(const CFP_CONTAINER_TYPE self, size_t* i, size_t* j, size_t index)
{
  static_cast<ZFP_CONTAINER_TYPE*>(self.object)->ij(*i, *j, index);
}
