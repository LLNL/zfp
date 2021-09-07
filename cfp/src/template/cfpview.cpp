static CFP_CONTAINER_TYPE
_t1(CFP_CONTAINER_TYPE, ctor)(const CFP_ARRAY_TYPE a)
{
  CFP_CONTAINER_TYPE v;
  v.object = new ZFP_CONTAINER_TYPE(*static_cast<const ZFP_CONTAINER_TYPE *>(a.object));
  return v;
}
