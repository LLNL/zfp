#ifndef ZFP_VIEW2_HPP
#define ZFP_VIEW2_HPP

// 2D array views

namespace zfp {
namespace internal {
namespace dim2 {

// abstract view of 2D array (base class)
template <class Container>
class preview {
public:
  typedef Container container_type;
  typedef typename container_type::value_type value_type;

  // rate in bits per value
  double rate() const { return array->rate(); }

  // dimensions of (sub)array
  size_t size() const { return nx * ny; }

  // local to global array indices
  size_t global_x(size_t i) const { return x + i; }
  size_t global_y(size_t j) const { return y + j; }

protected:
  // construction and assignment--perform shallow copy of (sub)array
  explicit preview(container_type* array) : array(array), x(0), y(0), nx(array->size_x()), ny(array->size_y()) {}
  explicit preview(container_type* array, size_t x, size_t y, size_t nx, size_t ny) : array(array), x(x), y(y), nx(nx), ny(ny) {}
  preview& operator=(container_type* a)
  {
    array = a;
    x = y = 0;
    nx = a->nx;
    ny = a->ny;
    return *this;
  }

  // global index bounds for iterators
  size_t min_x() const { return x; }
  size_t max_x() const { return x + nx; }
  size_t min_y() const { return y; }
  size_t max_y() const { return y + ny; }

  container_type* array; // underlying container
  size_t x, y;           // offset into array
  size_t nx, ny;         // dimensions of subarray
};

// generic read-only view into a rectangular subset of a 2D array
template <class Container>
class const_view : public preview<Container> {
public:
  typedef Container container_type;
  typedef typename container_type::value_type value_type;
  typedef typename zfp::internal::dim2::const_reference<const_view> const_reference;
  typedef typename zfp::internal::dim2::const_pointer<const_view> const_pointer;
  typedef typename zfp::internal::dim2::const_iterator<const_view> const_iterator;

  // construction--perform shallow copy of (sub)array
  const_view(container_type* array) : preview<Container>(array) {}
  const_view(container_type* array, size_t x, size_t y, size_t nx, size_t ny) : preview<Container>(array, x, y, nx, ny) {}

  // dimensions of (sub)array
  size_t size_x() const { return nx; }
  size_t size_y() const { return ny; }

  // (i, j) inspector
  const_reference operator()(size_t i, size_t j) const { return const_reference(this, x + i, y + j); }

  // random access iterators
  const_iterator cbegin() const { return const_iterator(this, x, y); }
  const_iterator cend() const { return const_iterator(this, x, y + ny); }
  const_iterator begin() const { return cbegin(); }
  const_iterator end() const { return cend(); }

protected:
  friend class zfp::internal::dim2::const_handle<const_view>;
  friend class zfp::internal::dim2::const_pointer<const_view>;
  friend class zfp::internal::dim2::const_iterator<const_view>;

  using preview<Container>::min_x;
  using preview<Container>::max_x;
  using preview<Container>::min_y;
  using preview<Container>::max_y;
  using preview<Container>::array;
  using preview<Container>::x;
  using preview<Container>::y;
  using preview<Container>::nx;
  using preview<Container>::ny;

  // inspector
  value_type get(size_t x, size_t y) const { return array->get(x, y); }
};

// generic read-write view into a rectangular subset of a 2D array
template <class Container>
class view : public const_view<Container> {
public:
  typedef Container container_type;
  typedef typename container_type::value_type value_type;
  typedef typename zfp::internal::dim2::const_reference<view> const_reference;
  typedef typename zfp::internal::dim2::const_pointer<view> const_pointer;
  typedef typename zfp::internal::dim2::const_iterator<view> const_iterator;
  typedef typename zfp::internal::dim2::reference<view> reference;
  typedef typename zfp::internal::dim2::pointer<view> pointer;
  typedef typename zfp::internal::dim2::iterator<view> iterator;

  // construction--perform shallow copy of (sub)array
  view(container_type* array) : const_view<Container>(array) {}
  view(container_type* array, size_t x, size_t y, size_t nx, size_t ny) : const_view<Container>(array, x, y, nx, ny) {}

  // (i, j) inspector
  const_reference operator()(size_t i, size_t j) const { return const_reference(this, x + i, y + j); }

  // (i, j) mutator
  reference operator()(size_t i, size_t j) { return reference(this, x + i, y + j); }

  // random access iterators
  const_iterator cbegin() const { return const_iterator(this, x, y); }
  const_iterator cend() const { return const_iterator(this, x, y + ny); }
  const_iterator begin() const { return cbegin(); }
  const_iterator end() const { return cend(); }
  iterator begin() { return iterator(this, x, y); }
  iterator end() { return iterator(this, x, y + ny); }

protected:
  friend class zfp::internal::dim2::const_handle<view>;
  friend class zfp::internal::dim2::const_pointer<view>;
  friend class zfp::internal::dim2::const_iterator<view>;
  friend class zfp::internal::dim2::reference<view>;
  friend class zfp::internal::dim2::pointer<view>;
  friend class zfp::internal::dim2::iterator<view>;

  using const_view<Container>::min_x;
  using const_view<Container>::max_x;
  using const_view<Container>::min_y;
  using const_view<Container>::max_y;
  using const_view<Container>::get;
  using const_view<Container>::array;
  using const_view<Container>::x;
  using const_view<Container>::y;
  using const_view<Container>::nx;
  using const_view<Container>::ny;

  // mutator
  void set(size_t x, size_t y, value_type val) { array->set(x, y, val); }

  // in-place updates
  void add(size_t x, size_t y, value_type val) { array->add(x, y, val); }
  void sub(size_t x, size_t y, value_type val) { array->sub(x, y, val); }
  void mul(size_t x, size_t y, value_type val) { array->mul(x, y, val); }
  void div(size_t x, size_t y, value_type val) { array->div(x, y, val); }
};

// flat view of 2D array (operator[] returns scalar)
template <class Container>
class flat_view : public view<Container> {
public:
  typedef Container container_type;
  typedef typename container_type::value_type value_type;
  typedef typename zfp::internal::dim2::const_reference<flat_view> const_reference;
  typedef typename zfp::internal::dim2::const_pointer<flat_view> const_pointer;
  typedef typename zfp::internal::dim2::reference<flat_view> reference;
  typedef typename zfp::internal::dim2::pointer<flat_view> pointer;

  // construction--perform shallow copy of (sub)array
  flat_view(container_type* array) : view<Container>(array) {}
  flat_view(container_type* array, size_t x, size_t y, size_t nx, size_t ny) : view<Container>(array, x, y, nx, ny) {}

  // convert (i, j) index to flat index
  size_t index(size_t i, size_t j) const { return i + nx * j; }

  // convert flat index to (i, j) index
  void ij(size_t& i, size_t& j, size_t index) const
  {
    i = index % nx; index /= nx;
    j = index;
  }

  // flat index [] inspector
  const_reference operator[](size_t index) const
  {
    size_t i, j;
    ij(i, j, index);
    return const_reference(this, x + i, y + j);
  }

  // flat index [] mutator
  reference operator[](size_t index)
  {
    size_t i, j;
    ij(i, j, index);
    return reference(this, x + i, y + j);
  }

  // (i, j) inspector
  const_reference operator()(size_t i, size_t j) const { return const_reference(this, x + i, y + j); }

  // (i, j) mutator
  reference operator()(size_t i, size_t j) { return reference(this, x + i, y + j); }

protected:
  friend class zfp::internal::dim2::const_handle<flat_view>;
  friend class zfp::internal::dim2::const_pointer<flat_view>;
  friend class zfp::internal::dim2::reference<flat_view>;
  friend class zfp::internal::dim2::pointer<flat_view>;

  using view<Container>::array;
  using view<Container>::x;
  using view<Container>::y;
  using view<Container>::nx;
  using view<Container>::ny;

  // inspector
  value_type get(size_t x, size_t y) const { return array->get(x, y); }

  // mutator
  void set(size_t x, size_t y, value_type val) { array->set(x, y, val); }

  // in-place updates
  void add(size_t x, size_t y, value_type val) { array->add(x, y, val); }
  void sub(size_t x, size_t y, value_type val) { array->sub(x, y, val); }
  void mul(size_t x, size_t y, value_type val) { array->mul(x, y, val); }
  void div(size_t x, size_t y, value_type val) { array->div(x, y, val); }
};

// forward declaration of friends
template <class Container> class nested_view1;
template <class Container> class nested_view2;

// nested view into a 1D rectangular subset of a 2D array
template <class Container>
class nested_view1 : public preview<Container> {
public:
  typedef Container container_type;
  typedef typename container_type::value_type value_type;
  typedef typename zfp::internal::dim2::const_reference<nested_view1> const_reference;
  typedef typename zfp::internal::dim2::const_pointer<nested_view1> const_pointer;
  typedef typename zfp::internal::dim2::reference<nested_view1> reference;
  typedef typename zfp::internal::dim2::pointer<nested_view1> pointer;

  // dimensions of (sub)array
  size_t size_x() const { return nx; }

  // [i] inspector and mutator
  const_reference operator[](size_t index) const { return const_reference(this, x + index, y); }
  reference operator[](size_t index) { return reference(this, x + index, y); }

  // (i) inspector and mutator
  const_reference operator()(size_t i) const { return const_reference(this, x + i, y); }
  reference operator()(size_t i) { return reference(this, x + i, y); }

protected:
  friend class zfp::internal::dim2::const_handle<nested_view1>;
  friend class zfp::internal::dim2::const_pointer<nested_view1>;
  friend class zfp::internal::dim2::reference<nested_view1>;
  friend class zfp::internal::dim2::pointer<nested_view1>;

  using preview<Container>::array;
  using preview<Container>::x;
  using preview<Container>::y;
  using preview<Container>::nx;
  using preview<Container>::ny;

  // construction--perform shallow copy of (sub)array
  friend class nested_view2<Container>;
  explicit nested_view1(container_type* array) : preview<Container>(array) {}
  explicit nested_view1(container_type* array, size_t x, size_t y, size_t nx, size_t ny) : preview<Container>(array, x, y, nx, ny) {}

  // inspector
  value_type get(size_t x, size_t y) const { return array->get(x, y); }

  // mutator
  void set(size_t x, size_t y, value_type val) { array->set(x, y, val); }

  // in-place updates
  void add(size_t x, size_t y, value_type val) { array->add(x, y, val); }
  void sub(size_t x, size_t y, value_type val) { array->sub(x, y, val); }
  void mul(size_t x, size_t y, value_type val) { array->mul(x, y, val); }
  void div(size_t x, size_t y, value_type val) { array->div(x, y, val); }
};

// nested view into a 2D rectangular subset of a 2D array
template <class Container>
class nested_view2 : public preview<Container> {
public:
  typedef Container container_type;
  typedef typename container_type::value_type value_type;
  typedef typename zfp::internal::dim2::const_reference<nested_view2> const_reference;
  typedef typename zfp::internal::dim2::const_pointer<nested_view2> const_pointer;
  typedef typename zfp::internal::dim2::reference<nested_view2> reference;
  typedef typename zfp::internal::dim2::pointer<nested_view2> pointer;

  // construction--perform shallow copy of (sub)array
  nested_view2(container_type* array) : preview<Container>(array) {}
  nested_view2(container_type* array, size_t x, size_t y, size_t nx, size_t ny) : preview<Container>(array, x, y, nx, ny) {}

  // dimensions of (sub)array
  size_t size_x() const { return nx; }
  size_t size_y() const { return ny; }

  // 1D view
  nested_view1<Container> operator[](size_t index) const { return nested_view1<Container>(array, x, y + index, nx, 1); }

  // (i, j) inspector and mutator
  const_reference operator()(size_t i, size_t j) const { return const_reference(this, x + i, y + j); }
  reference operator()(size_t i, size_t j) { return reference(this, x + i, y + j); }

protected:
  friend class zfp::internal::dim2::const_handle<nested_view2>;
  friend class zfp::internal::dim2::const_pointer<nested_view2>;
  friend class zfp::internal::dim2::reference<nested_view2>;
  friend class zfp::internal::dim2::pointer<nested_view2>;

  using preview<Container>::array;
  using preview<Container>::x;
  using preview<Container>::y;
  using preview<Container>::nx;
  using preview<Container>::ny;

  // inspector
  value_type get(size_t x, size_t y) const { return array->get(x, y); }

  // mutator
  void set(size_t x, size_t y, value_type val) { array->set(x, y, val); }

  // in-place updates
  void add(size_t x, size_t y, value_type val) { array->add(x, y, val); }
  void sub(size_t x, size_t y, value_type val) { array->sub(x, y, val); }
  void mul(size_t x, size_t y, value_type val) { array->mul(x, y, val); }
  void div(size_t x, size_t y, value_type val) { array->div(x, y, val); }
};

// thread-safe read-only view of 2D (sub)array with private cache
template <class Container>
class private_const_view : public preview<Container> {
public:
  typedef Container container_type;
  typedef typename container_type::value_type value_type;
  typedef typename container_type::store_type store_type;
  typedef typename zfp::internal::dim2::const_reference<private_const_view> const_reference;
  typedef typename zfp::internal::dim2::const_pointer<private_const_view> const_pointer;
  typedef typename zfp::internal::dim2::const_iterator<private_const_view> const_iterator;

  // construction--perform shallow copy of (sub)array
  private_const_view(container_type* array, size_t cache_size = 0) :
    preview<Container>(array),
    cache(array->store, cache_size ? cache_size : array->cache.size())
  {
    array->store.reference();
  }
  private_const_view(container_type* array, size_t x, size_t y, size_t nx, size_t ny, size_t cache_size = 0) :
    preview<Container>(array, x, y, nx, ny),
    cache(array->store, cache_size ? cache_size : array->cache.size())
  {
    array->store.reference();
  }

  // destructor
  ~private_const_view()
  {
    array->store.unreference();
  }

  // dimensions of (sub)array
  size_t size_x() const { return nx; }
  size_t size_y() const { return ny; }

  // cache size in number of bytes
  size_t cache_size() const { return cache.size(); }

  // set minimum cache size in bytes (array dimensions must be known)
  void set_cache_size(size_t bytes) { cache.resize(bytes); }

  // empty cache without compressing modified cached blocks
  void clear_cache() const { cache.clear(); }

  // (i, j) inspector
  const_reference operator()(size_t i, size_t j) const { return const_reference(this, x + i, y + j); }

  // random access iterators
  const_iterator cbegin() const { return const_iterator(this, x, y); }
  const_iterator cend() const { return const_iterator(this, x, y + ny); }
  const_iterator begin() const { return cbegin(); }
  const_iterator end() const { return cend(); }

protected:
  friend class zfp::internal::dim2::const_handle<private_const_view>;
  friend class zfp::internal::dim2::const_pointer<private_const_view>;
  friend class zfp::internal::dim2::const_iterator<private_const_view>;

  using preview<Container>::min_x;
  using preview<Container>::max_x;
  using preview<Container>::min_y;
  using preview<Container>::max_y;
  using preview<Container>::array;
  using preview<Container>::x;
  using preview<Container>::y;
  using preview<Container>::nx;
  using preview<Container>::ny;

  // inspector
  value_type get(size_t x, size_t y) const { return cache.get(x, y); }

  BlockCache2<value_type, store_type> cache; // cache of decompressed blocks
};

// thread-safe read-write view of private 2D (sub)array
template <class Container>
class private_view : public private_const_view<Container> {
public:
  typedef Container container_type;
  typedef typename container_type::value_type value_type;
  typedef typename zfp::internal::dim2::const_reference<private_view> const_reference;
  typedef typename zfp::internal::dim2::const_pointer<private_view> const_pointer;
  typedef typename zfp::internal::dim2::const_iterator<private_view> const_iterator;
  typedef typename zfp::internal::dim2::reference<private_view> reference;
  typedef typename zfp::internal::dim2::pointer<private_view> pointer;
  typedef typename zfp::internal::dim2::iterator<private_view> iterator;

  // construction--perform shallow copy of (sub)array
  private_view(container_type* array, size_t cache_size = 0) : private_const_view<Container>(array, cache_size) {}
  private_view(container_type* array, size_t x, size_t y, size_t nx, size_t ny, size_t cache_size = 0) : private_const_view<Container>(array, x, y, nx, ny, cache_size) {}

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

  // (i, j) inspector
  const_reference operator()(size_t i, size_t j) const { return const_reference(this, x + i, y + j); }

  // (i, j) mutator
  reference operator()(size_t i, size_t j) { return reference(this, x + i, y + j); }

  // random access iterators
  const_iterator cbegin() const { return const_iterator(this, x, y); }
  const_iterator cend() const { return const_iterator(this, x, y + ny); }
  const_iterator begin() const { return cbegin(); }
  const_iterator end() const { return cend(); }
  iterator begin() { return iterator(this, x, y); }
  iterator end() { return iterator(this, x, y + ny); }

protected:
  friend class zfp::internal::dim2::const_handle<private_view>;
  friend class zfp::internal::dim2::const_pointer<private_view>;
  friend class zfp::internal::dim2::const_iterator<private_view>;
  friend class zfp::internal::dim2::reference<private_view>;
  friend class zfp::internal::dim2::pointer<private_view>;
  friend class zfp::internal::dim2::iterator<private_view>;

  using private_const_view<Container>::min_x;
  using private_const_view<Container>::max_x;
  using private_const_view<Container>::min_y;
  using private_const_view<Container>::max_y;
  using private_const_view<Container>::get;
  using private_const_view<Container>::array;
  using private_const_view<Container>::x;
  using private_const_view<Container>::y;
  using private_const_view<Container>::nx;
  using private_const_view<Container>::ny;
  using private_const_view<Container>::cache;

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
  void set(size_t x, size_t y, value_type val) { cache.set(x, y, val); }

  // in-place updates
  void add(size_t x, size_t y, value_type val) { cache.ref(x, y) += val; }
  void sub(size_t x, size_t y, value_type val) { cache.ref(x, y) -= val; }
  void mul(size_t x, size_t y, value_type val) { cache.ref(x, y) *= val; }
  void div(size_t x, size_t y, value_type val) { cache.ref(x, y) /= val; }
};

} // dim2
} // internal
} // zfp

#endif
