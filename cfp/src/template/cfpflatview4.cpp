static size_t
_t1(CFP_CONTAINER_TYPE, index)(const CFP_CONTAINER_TYPE self, size_t i, size_t j, size_t k, size_t l)
{
  return static_cast<ZFP_CONTAINER_TYPE*>(self.object)->index(i, j, k, l);
}

static void
_t1(CFP_CONTAINER_TYPE, ijkl)(const CFP_CONTAINER_TYPE self, size_t* i, size_t* j, size_t* k, size_t* l, size_t index)
{
  static_cast<ZFP_CONTAINER_TYPE*>(self.object)->ijkl(*i, *j, *k, *l, index);
}
