// 1D array views; these classes are nested within zfp::array1

// abstract view of 1D array (base class)
class preview {
public:
  typedef typename container_type::value_type value_type;

  // rate in bits per value
  double rate() const { return array->rate(); }

  // dimensions of (sub)array
  size_t size() const { return size_t(nx); }

  // local to global array index
  uint global_x(uint i) const { return x + i; }

protected:
  // construction and assignment--perform shallow copy of (sub)array
  explicit preview(array1* array) : array(array), x(0), nx(array->nx) {}
  explicit preview(array1* array, uint x, uint nx) : array(array), x(x), nx(nx) {}
  preview& operator=(array1* a)
  {
    array = a;
    x = 0;
    nx = a->nx;
    return *this;
  }

  array1* array; // underlying container
  uint x;        // offset into array
  uint nx;       // dimensions of subarray
};

// generic read-only view into a rectangular subset of a 1D array
class const_view : public preview {
protected:
  using preview::array;
  using preview::x;
  using preview::nx;
public:
  // construction--perform shallow copy of (sub)array
  const_view(array1* array) : preview(array) {}
  const_view(array1* array, uint x, uint nx) : preview(array, x, nx) {}

  // dimensions of (sub)array
  uint size_x() const { return nx; }

  // [i] accessor
  value_type operator[](uint index) const { return array->get(x + index); }

  // (i) accessor
  value_type operator()(uint i) const { return array->get(x + i); }
};

// generic read-write view into a rectangular subset of a 1D array
class view : public const_view {
protected:
  using preview::array;
  using preview::x;
  using preview::nx;
public:
  // construction--perform shallow copy of (sub)array
  view(array1* array) : const_view(array) {}
  view(array1* array, uint x, uint nx) : const_view(array, x, nx) {}

  // [i] accessor from base class
  using const_view::operator[];

  // (i) accessor from base class
  using const_view::operator();

  // [i] mutator
  reference operator[](uint index) { return reference(array, x + index); }

  // (i) mutator
  reference operator()(uint i) { return reference(array, x + i); }
};

// thread-safe read-only view of 1D (sub)array with private cache
class private_const_view : public preview {
protected:
  using preview::array;
  using preview::x;
  using preview::nx;
public:
  // construction--perform shallow copy of (sub)array
  private_const_view(array1* array, size_t cache_size = 0) :
    preview(array),
    cache(array->store, cache_size ? cache_size : array->cache.size())
  {}
  private_const_view(array1* array, uint x, uint nx, size_t cache_size = 0) :
    preview(array, x, nx),
    cache(array->store, cache_size ? cache_size : array->cache.size())
  {}

  // dimensions of (sub)array
  uint size_x() const { return nx; }

  // cache size in number of bytes
  size_t cache_size() const { return cache.size(); }

  // set minimum cache size in bytes (array dimensions must be known)
  void set_cache_size(size_t bytes) { cache.resize(bytes); }

  // empty cache without compressing modified cached blocks
  void clear_cache() const { cache.clear(); }

  // (i) accessor
  value_type operator()(uint i) const { return get(x + i); }

protected:
  // inspector
  value_type get(uint i) const { return cache.get(i); }

  mutable BlockCache1<value_type, codec_type> cache; // cache of decompressed blocks
};

// thread-safe read-write view of private 1D (sub)array
class private_view : public private_const_view {
protected:
  using preview::array;
  using preview::x;
  using preview::nx;
  using private_const_view::cache;
public:
  // private view uses its own references to access private cache
  typedef private_view container_type;
  typedef typename preview::value_type value_type;
  #include "zfp/handle1.h"
  #include "zfp/reference1.h"

  // construction--perform shallow copy of (sub)array
  private_view(array1* array, size_t cache_size = 0) : private_const_view(array, cache_size) {}
  private_view(array1* array, uint x, uint nx, size_t cache_size = 0) : private_const_view(array, x, nx, cache_size) {}

  // partition view into count block-aligned pieces, with 0 <= index < count
  void partition(uint index, uint count)
  {
    partition(x, nx, index, count);
  }

  // flush cache by compressing all modified cached blocks
  void flush_cache() const { cache.flush(); }

  // (i) accessor from base class
  using private_const_view::operator();

  // (i) mutator
  reference operator()(uint i) { return reference(this, x + i); }

protected:
  // block-aligned partition of [offset, offset + size): index out of count
  static void partition(uint& offset, uint& size, uint index, uint count)
  {
    uint bmin = offset / 4;
    uint bmax = (offset + size + 3) / 4;
    uint xmin = std::max(offset +    0, 4 * (bmin + (bmax - bmin) * (index + 0) / count));
    uint xmax = std::min(offset + size, 4 * (bmin + (bmax - bmin) * (index + 1) / count));
    offset = xmin;
    size = xmax - xmin;
  }

  // mutator
  void set(uint i, value_type val) { cache.set(i, val); }

  // in-place updates
  void add(uint i, value_type val) { cache.ref(i) += val; }
  void sub(uint i, value_type val) { cache.ref(i) -= val; }
  void mul(uint i, value_type val) { cache.ref(i) *= val; }
  void div(uint i, value_type val) { cache.ref(i) /= val; }
};
