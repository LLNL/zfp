// 1D array views; these classes are nested within zfp::array1

// abstract view of 1D array (base class)
class preview {
public:
  // rate in bits per value
  double rate() const { return array->rate(); }

  // dimensions of (sub)array
  size_t size() const { return nx; }

  // local to global array index
  size_t global_x(size_t i) const { return x + i; }

protected:
  // construction and assignment--perform shallow copy of (sub)array
  explicit preview(array1* array) : array(array), x(0), nx(array->nx) {}
  explicit preview(array1* array, size_t x, size_t nx) : array(array), x(x), nx(nx) {}
  preview& operator=(array1* a)
  {
    array = a;
    x = 0;
    nx = a->nx;
    return *this;
  }

  array1* array; // underlying container
  size_t x;      // offset into array
  size_t nx;     // dimensions of subarray
};

// generic read-only view into a rectangular subset of a 1D array
class const_view : public preview {
protected:
  using preview::array;
  using preview::x;
  using preview::nx;
public:
  typedef typename container_type::value_type value_type;
  typedef typename container_type::const_reference const_reference;
  typedef typename container_type::const_pointer const_pointer;

  // construction--perform shallow copy of (sub)array
  const_view(array1* array) : preview(array) {}
  const_view(array1* array, size_t x, size_t nx) : preview(array, x, nx) {}

  // dimensions of (sub)array
  size_t size_x() const { return nx; }

  // [i] accessor
  const_reference operator[](size_t index) const { return const_reference(array, x + index); }

  // (i) accessor
  const_reference operator()(size_t i) const { return const_reference(array, x + i); }
};

// generic read-write view into a rectangular subset of a 1D array
class view : public const_view {
protected:
  using preview::array;
  using preview::x;
  using preview::nx;
public:
  typedef typename container_type::value_type value_type;
  typedef typename container_type::reference reference;
  typedef typename container_type::pointer pointer;

  // construction--perform shallow copy of (sub)array
  view(array1* array) : const_view(array) {}
  view(array1* array, size_t x, size_t nx) : const_view(array, x, nx) {}

  // [i] accessor from base class
  using const_view::operator[];

  // (i) accessor from base class
  using const_view::operator();

  // [i] mutator
  reference operator[](size_t index) { return reference(array, x + index); }

  // (i) mutator
  reference operator()(size_t i) { return reference(array, x + i); }
};

// thread-safe read-only view of 1D (sub)array with private cache
class private_const_view : public preview {
protected:
  using preview::array;
  using preview::x;
  using preview::nx;
public:
  // private view uses its own references to access private cache
  typedef typename container_type::value_type value_type;
  typedef private_const_view container_type;
  class const_pointer;
  #include "zfp/handle1.h"
  #include "zfp/reference1.h"
  #include "zfp/pointer1.h"

  // construction--perform shallow copy of (sub)array
  private_const_view(array1* array, size_t cache_size = 0) :
    preview(array),
    cache(array->store, cache_size ? cache_size : array->cache.size())
  {}
  private_const_view(array1* array, size_t x, size_t nx, size_t cache_size = 0) :
    preview(array, x, nx),
    cache(array->store, cache_size ? cache_size : array->cache.size())
  {}

  // dimensions of (sub)array
  size_t size_x() const { return nx; }

  // cache size in number of bytes
  size_t cache_size() const { return cache.size(); }

  // set minimum cache size in bytes (array dimensions must be known)
  void set_cache_size(size_t bytes) { cache.resize(bytes); }

  // empty cache without compressing modified cached blocks
  void clear_cache() const { cache.clear(); }

  // (i) accessor
  const_reference operator()(size_t i) const { return const_reference(const_cast<container_type*>(this), x + i); }

protected:
  // inspector
  value_type get(size_t i) const { return cache.get(i); }

  BlockCache1<value_type, codec_type> cache; // cache of decompressed blocks
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
  typedef typename container_type::value_type value_type;
  typedef private_view container_type;
  class const_pointer;
  class pointer;
  #include "zfp/handle1.h"
  #include "zfp/reference1.h"
  #include "zfp/pointer1.h"

  // construction--perform shallow copy of (sub)array
  private_view(array1* array, size_t cache_size = 0) : private_const_view(array, cache_size) {}
  private_view(array1* array, size_t x, size_t nx, size_t cache_size = 0) : private_const_view(array, x, nx, cache_size) {}

  // partition view into count block-aligned pieces, with 0 <= index < count
  void partition(size_t index, size_t count)
  {
    partition(x, nx, index, count);
  }

  // flush cache by compressing all modified cached blocks
  void flush_cache() const { cache.flush(); }

  // (i) accessor from base class
  using private_const_view::operator();

  // (i) mutator
  reference operator()(size_t i) { return reference(this, x + i); }

protected:
  // block-aligned partition of [offset, offset + size): index out of count
  static void partition(size_t& offset, size_t& size, size_t index, size_t count)
  {
    size_t bmin = offset / 4;
    size_t bmax = (offset + size + 3) / 4;
    size_t xmin = std::max(offset +    0, 4 * (bmin + (bmax - bmin) * (index + 0) / count));
    size_t xmax = std::min(offset + size, 4 * (bmin + (bmax - bmin) * (index + 1) / count));
    offset = xmin;
    size = xmax - xmin;
  }

  // mutator
  void set(size_t i, value_type val) { cache.set(i, val); }

  // in-place updates
  void add(size_t i, value_type val) { cache.ref(i) += val; }
  void sub(size_t i, value_type val) { cache.ref(i) -= val; }
  void mul(size_t i, value_type val) { cache.ref(i) *= val; }
  void div(size_t i, value_type val) { cache.ref(i) /= val; }
};
