static CFP_CONTAINER_TYPE
_t1(CFP_CONTAINER_TYPE, ctor_subset)(const CFP_ARRAY_TYPE a, size_t x, size_t y, size_t z, size_t w, size_t nx, size_t ny, size_t nz, size_t nw)
{
  CFP_CONTAINER_TYPE v;
  v.object = new ZFP_CONTAINER_TYPE(static_cast<ZFP_ARRAY_TYPE *>(a.object), x, y, z, w, nx, ny, nz, nw);
  return v;
}

static size_t
_t1(CFP_CONTAINER_TYPE, global_x)(const CFP_CONTAINER_TYPE self, size_t i)
{
  return static_cast<const ZFP_CONTAINER_TYPE*>(self.object)->global_x(i);
}

static size_t
_t1(CFP_CONTAINER_TYPE, global_y)(const CFP_CONTAINER_TYPE self, size_t j)
{
  return static_cast<const ZFP_CONTAINER_TYPE*>(self.object)->global_y(j);
}

static size_t
_t1(CFP_CONTAINER_TYPE, global_z)(const CFP_CONTAINER_TYPE self, size_t k)
{
  return static_cast<const ZFP_CONTAINER_TYPE*>(self.object)->global_z(k);
}

static size_t
_t1(CFP_CONTAINER_TYPE, global_w)(const CFP_CONTAINER_TYPE self, size_t l)
{
  return static_cast<const ZFP_CONTAINER_TYPE*>(self.object)->global_w(l);
}
