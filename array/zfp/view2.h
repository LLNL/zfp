// 2D array views; these classes are nested within zfp::container_type

// abstract view of 2D array (base class)
class preview {
public:
  // rate in bits per value
  double rate() const { return array->rate(); }

  // dimensions of (sub)array
  size_t size() const { return nx * ny; }

  // local to global array indices
  size_t global_x(size_t i) const { return x + i; }
  size_t global_y(size_t j) const { return y + j; }

protected:
  // construction and assignment--perform shallow copy of (sub)array
  explicit preview(container_type* array) : array(array), x(0), y(0), nx(array->nx), ny(array->ny) {}
  explicit preview(container_type* array, size_t x, size_t y, size_t nx, size_t ny) : array(array), x(x), y(y), nx(nx), ny(ny) {}
  preview& operator=(container_type* a)
  {
    array = a;
    x = y = 0;
    nx = a->nx;
    ny = a->ny;
    return *this;
  }

  container_type* array; // underlying container
  size_t x, y;           // offset into array
  size_t nx, ny;         // dimensions of subarray
};

// generic read-only view into a rectangular subset of a 2D array
class const_view : public preview {
protected:
  using preview::array;
  using preview::x;
  using preview::y;
  using preview::nx;
  using preview::ny;
public:
  typedef typename container_type::value_type value_type;
  typedef typename container_type::const_reference const_reference;
  typedef typename container_type::const_pointer const_pointer;

  // construction--perform shallow copy of (sub)array
  const_view(container_type* array) : preview(array) {}
  const_view(container_type* array, size_t x, size_t y, size_t nx, size_t ny) : preview(array, x, y, nx, ny) {}

  // dimensions of (sub)array
  size_t size_x() const { return nx; }
  size_t size_y() const { return ny; }

  // (i, j) accessor
  const_reference operator()(size_t i, size_t j) const { return const_reference(array, x + i, y + j); }
};

// generic read-write view into a rectangular subset of a 2D array
class view : public const_view {
protected:
  using preview::array;
  using preview::x;
  using preview::y;
  using preview::nx;
  using preview::ny;
public:
  typedef typename container_type::value_type value_type;
  typedef typename container_type::reference reference;
  typedef typename container_type::pointer pointer;

  // construction--perform shallow copy of (sub)array
  view(container_type* array) : const_view(array) {}
  view(container_type* array, size_t x, size_t y, size_t nx, size_t ny) : const_view(array, x, y, nx, ny) {}

  // (i, j) accessor from base class
  using const_view::operator();

  // (i, j) mutator
  reference operator()(size_t i, size_t j) { return reference(array, x + i, y + j); }
};

// flat view of 2D array (operator[] returns scalar)
class flat_view : public view {
protected:
  using preview::array;
  using preview::x;
  using preview::y;
  using preview::nx;
  using preview::ny;
public:
  typedef typename container_type::value_type value_type;
  typedef typename container_type::const_reference const_reference;
  typedef typename container_type::reference reference;
  typedef typename container_type::const_pointer const_pointer;
  typedef typename container_type::pointer pointer;

  // construction--perform shallow copy of (sub)array
  flat_view(container_type* array) : view(array) {}
  flat_view(container_type* array, size_t x, size_t y, size_t nx, size_t ny) : view(array, x, y, nx, ny) {}

  // convert (i, j) index to flat index
  size_t index(size_t i, size_t j) const { return i + nx * j; }

  // convert flat index to (i, j) index
  void ij(size_t& i, size_t& j, size_t index) const
  {
    i = index % nx; index /= nx;
    j = index;
  }

  // flat index accessors
  const_reference operator[](size_t index) const
  {
    size_t i, j;
    ij(i, j, index);
    return const_reference(array, x + i, y + j);
  }
  reference operator[](size_t index)
  {
    size_t i, j;
    ij(i, j, index);
    return reference(array, x + i, y + j);
  }
};

// forward declaration of friends
class nested_view1;
class nested_view2;

// nested view into a 1D rectangular subset of a 2D array
class nested_view1 : public preview {
protected:
  using preview::array;
  using preview::x;
  using preview::y;
  using preview::nx;
  using preview::ny;
public:
  typedef typename container_type::value_type value_type;
  typedef typename container_type::const_reference const_reference;
  typedef typename container_type::reference reference;
  typedef typename container_type::const_pointer const_pointer;
  typedef typename container_type::pointer pointer;

  // dimensions of (sub)array
  size_t size_x() const { return nx; }

  // [i] accessor and mutator
  const_reference operator[](size_t index) const { return const_reference(array, x + index, y); }
  reference operator[](size_t index) { return reference(array, x + index, y); }

  // (i) accessor and mutator
  const_reference operator()(size_t i) const { return const_reference(array, x + i, y); }
  reference operator()(size_t i) { return reference(array, x + i, y); }

protected:
  // construction--perform shallow copy of (sub)array
  friend class nested_view2;
  explicit nested_view1(container_type* array) : preview(array) {}
  explicit nested_view1(container_type* array, size_t x, size_t y, size_t nx, size_t ny) : preview(array, x, y, nx, ny) {}
};

// nested view into a 2D rectangular subset of a 2D array
class nested_view2 : public preview {
protected:
  using preview::array;
  using preview::x;
  using preview::y;
  using preview::nx;
  using preview::ny;
public:
  typedef typename container_type::value_type value_type;
  typedef typename container_type::const_reference const_reference;
  typedef typename container_type::reference reference;
  typedef typename container_type::const_pointer const_pointer;
  typedef typename container_type::pointer pointer;

  // construction--perform shallow copy of (sub)array
  nested_view2(container_type* array) : preview(array) {}
  nested_view2(container_type* array, size_t x, size_t y, size_t nx, size_t ny) : preview(array, x, y, nx, ny) {}

  // dimensions of (sub)array
  size_t size_x() const { return nx; }
  size_t size_y() const { return ny; }

  // 1D view
  nested_view1 operator[](size_t index) const { return nested_view1(array, x, y + index, nx, 1); }

  // (i, j) accessor and mutator
  const_reference operator()(size_t i, size_t j) const { return const_reference(array, x + i, y + j); }
  reference operator()(size_t i, size_t j) { return reference(array, x + i, y + j); }
};

typedef nested_view2 nested_view;

// thread-safe read-only view of 2D (sub)array with private cache
class private_const_view : public preview {
protected:
  using preview::array;
  using preview::x;
  using preview::y;
  using preview::nx;
  using preview::ny;
public:
  // construction--perform shallow copy of (sub)array
  private_const_view(container_type* array, size_t cache_size = 0) :
    preview(array),
    cache(array->store, cache_size ? cache_size : array->cache.size())
  {}
  private_const_view(container_type* array, size_t x, size_t y, size_t nx, size_t ny, size_t cache_size = 0) :
    preview(array, x, y, nx, ny),
    cache(array->store, cache_size ? cache_size : array->cache.size())
  {}

  // private view uses its own references to access private cache
  typedef typename container_type::value_type value_type;
  typedef private_const_view container_type;
  class const_pointer;
  #include "zfp/handle2.h"
  #include "zfp/reference2.h"
  #include "zfp/pointer2.h"

  // dimensions of (sub)array
  size_t size_x() const { return nx; }
  size_t size_y() const { return ny; }

  // cache size in number of bytes
  size_t cache_size() const { return cache.size(); }

  // set minimum cache size in bytes (array dimensions must be known)
  void set_cache_size(size_t bytes) { cache.resize(bytes); }

  // empty cache without compressing modified cached blocks
  void clear_cache() const { cache.clear(); }

  // (i, j) accessor
  const_reference operator()(size_t i, size_t j) const { return const_reference(const_cast<container_type*>(this), x + i, y + j); }

protected:
  // inspector
  value_type get(size_t i, size_t j) const { return cache.get(i, j); }

  BlockCache2<value_type, codec_type> cache; // cache of decompressed blocks
};

// thread-safe read-write view of private 2D (sub)array
class private_view : public private_const_view {
protected:
  using preview::array;
  using preview::x;
  using preview::y;
  using preview::nx;
  using preview::ny;
  using private_const_view::cache;
public:
  // construction--perform shallow copy of (sub)array
  private_view(container_type* array, size_t cache_size = 0) : private_const_view(array, cache_size) {}
  private_view(container_type* array, size_t x, size_t y, size_t nx, size_t ny, size_t cache_size = 0) : private_const_view(array, x, y, nx, ny, cache_size) {}

  // private view uses its own references to access private cache
  typedef typename container_type::value_type value_type;
  typedef private_view container_type;
  class const_pointer;
  class pointer;
  #include "zfp/handle2.h"
  #include "zfp/reference2.h"
  #include "zfp/pointer2.h"

  // partition view into count block-aligned pieces, with 0 <= index < count
  void partition(size_t index, size_t count)
  {
    if (nx > ny)
      partition(x, nx, index, count);
    else
      partition(y, ny, index, count);
  }

  // flush cache by compressing all modified cached blocks
  void flush_cache() const { cache.flush(); }

  // (i, j) accessor from base class
  using private_const_view::operator();

  // (i, j) mutator
  reference operator()(size_t i, size_t j) { return reference(this, x + i, y + j); }

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
  void set(size_t i, size_t j, value_type val) { cache.set(i, j, val); }

  // in-place updates
  void add(size_t i, size_t j, value_type val) { cache.ref(i, j) += val; }
  void sub(size_t i, size_t j, value_type val) { cache.ref(i, j) -= val; }
  void mul(size_t i, size_t j, value_type val) { cache.ref(i, j) *= val; }
  void div(size_t i, size_t j, value_type val) { cache.ref(i, j) /= val; }
};
