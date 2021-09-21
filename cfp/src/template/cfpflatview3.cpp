static size_t
_t1(CFP_CONTAINER_TYPE, index)(const CFP_CONTAINER_TYPE self, size_t i, size_t j, size_t k)
{
  return static_cast<ZFP_CONTAINER_TYPE*>(self.object)->index(i, j, k);
}

static void
_t1(CFP_CONTAINER_TYPE, ijk)(const CFP_CONTAINER_TYPE self, size_t* i, size_t* j, size_t* k, size_t index)
{
  static_cast<ZFP_CONTAINER_TYPE*>(self.object)->ijk(*i, *j, *k, index);
}
