// 2D array views; these classes are nested within zfp::array2

// abstract view of 2D array (base class)
class preview {
public:
  typedef container_type::value_type value_type;

  // rate in bits per value
  double rate() const { return array->rate(); }

  // dimensions of (sub)array
  size_t size() const { return size_t(nx) * size_t(ny); }

  // local to global array indices
  uint global_x(uint i) const { return x + i; }
  uint global_y(uint j) const { return y + j; }

protected:
  // construction and assignment--perform shallow copy of (sub)array
  explicit preview(array2* array) : array(array), x(0), y(0), nx(array->nx), ny(array->ny) {}
  explicit preview(array2* array, uint x, uint y, uint nx, uint ny) : array(array), x(x), y(y), nx(nx), ny(ny) {}
  preview& operator=(array2* a)
  {
    array = a;
    x = y = 0;
    nx = a->nx;
    ny = a->ny;
    return *this;
  }

  array2* array; // underlying container
  uint x, y;     // offset into array
  uint nx, ny;   // dimensions of subarray
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
  // construction--perform shallow copy of (sub)array
  const_view(array2* array) : preview(array) {}
  const_view(array2* array, uint x, uint y, uint nx, uint ny) : preview(array, x, y, nx, ny) {}

  // dimensions of (sub)array
  uint size_x() const { return nx; }
  uint size_y() const { return ny; }

  // (i, j) accessor
  value_type operator()(uint i, uint j) const { return array->get(x + i, y + j); }
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
  // construction--perform shallow copy of (sub)array
  view(array2* array) : const_view(array) {}
  view(array2* array, uint x, uint y, uint nx, uint ny) : const_view(array, x, y, nx, ny) {}

  // (i, j) accessor from base class
  using const_view::operator();

  // (i, j) mutator
  reference operator()(uint i, uint j) { return reference(array, x + i, y + j); }
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
  // construction--perform shallow copy of (sub)array
  flat_view(array2* array) : view(array) {}
  flat_view(array2* array, uint x, uint y, uint nx, uint ny) : view(array, x, y, nx, ny) {}

  // convert (i, j) index to flat index
  uint index(uint i, uint j) const { return i + nx * j; }

  // convert flat index to (i, j) index
  void ij(uint& i, uint& j, uint index) const
  {
    i = index % nx; index /= nx;
    j = index;
  }

  // flat index accessors
  value_type operator[](uint index) const
  {
    uint i, j;
    ij(i, j, index);
    return array->get(x + i, y + j);
  }
  reference operator[](uint index)
  {
    uint i, j;
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
  // dimensions of (sub)array
  uint size_x() const { return nx; }

  // [i] accessor and mutator
  value_type operator[](uint index) const { return array->get(x + index, y); }
  reference operator[](uint index) { return reference(array, x + index, y); }

  // (i) accessor and mutator
  value_type operator()(uint i) const { return array->get(x + i, y); }
  reference operator()(uint i) { return reference(array, x + i, y); }

protected:
  // construction--perform shallow copy of (sub)array
  friend class nested_view2;
  explicit nested_view1(array2* array) : preview(array) {}
  explicit nested_view1(array2* array, uint x, uint y, uint nx, uint ny) : preview(array, x, y, nx, ny) {}
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
  // construction--perform shallow copy of (sub)array
  nested_view2(array2* array) : preview(array) {}
  nested_view2(array2* array, uint x, uint y, uint nx, uint ny) : preview(array, x, y, nx, ny) {}

  // dimensions of (sub)array
  uint size_x() const { return nx; }
  uint size_y() const { return ny; }

  // 1D view
  nested_view1 operator[](uint index) const { return nested_view1(array, x, y + index, nx, 1); }

  // (i, j) accessor and mutator
  value_type operator()(uint i, uint j) const { return array->get(x + i, y + j); }
  reference operator()(uint i, uint j) { return reference(array, x + i, y + j); }
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
  private_const_view(array2* array, size_t cache_size = 0) :
    preview(array),
    cache(array->store, cache_size ? cache_size : array->cache.size())
  {}
  private_const_view(array2* array, uint x, uint y, uint nx, uint ny, size_t cache_size = 0) :
    preview(array, x, y, nx, ny),
    cache(array->store, cache_size ? cache_size : array->cache.size())
  {}

  // dimensions of (sub)array
  uint size_x() const { return nx; }
  uint size_y() const { return ny; }

  // cache size in number of bytes
  size_t cache_size() const { return cache.size(); }

  // set minimum cache size in bytes (array dimensions must be known)
  void set_cache_size(size_t bytes) { cache.resize(bytes); }

  // empty cache without compressing modified cached blocks
  void clear_cache() const { cache.clear(); }

  // (i, j) accessor
  value_type operator()(uint i, uint j) const { return get(x + i, y + j); }

protected:
  // inspector
  value_type get(uint i, uint j) const { return cache.get(i, j); }

  mutable BlockCache2<value_type, codec_type> cache; // cache of decompressed blocks
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
  // private view uses its own references to access private cache
  typedef private_view container_type;
  typedef typename preview::value_type value_type;
  #include "zfp/handle2.h"
  #include "zfp/reference2.h"

  // construction--perform shallow copy of (sub)array
  private_view(array2* array, size_t cache_size = 0) : private_const_view(array, cache_size) {}
  private_view(array2* array, uint x, uint y, uint nx, uint ny, size_t cache_size = 0) : private_const_view(array, x, y, nx, ny, cache_size) {}

  // partition view into count block-aligned pieces, with 0 <= index < count
  void partition(uint index, uint count)
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
  reference operator()(uint i, uint j) { return reference(this, x + i, y + j); }

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
  void set(uint i, uint j, value_type val) { cache.set(i, j, val); }

  // in-place updates
  void add(uint i, uint j, value_type val) { cache.ref(i, j) += val; }
  void sub(uint i, uint j, value_type val) { cache.ref(i, j) -= val; }
  void mul(uint i, uint j, value_type val) { cache.ref(i, j) *= val; }
  void div(uint i, uint j, value_type val) { cache.ref(i, j) /= val; }
};
