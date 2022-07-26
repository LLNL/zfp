#ifndef ZFP_VIEW4_HPP
#define ZFP_VIEW4_HPP

// 4D array views

namespace zfp {
namespace internal {
namespace dim4 {

// abstract view of 4D array (base class)
template <class Container>
class preview {
public:
  typedef Container container_type;
  typedef typename container_type::value_type value_type;

  // rate in bits per value
  double rate() const { return array->rate(); }

  // dimensions of (sub)array
  size_t size() const { return nx * ny * nz * nw; }

  // local to global array indices
  size_t global_x(size_t i) const { return x + i; }
  size_t global_y(size_t j) const { return y + j; }
  size_t global_z(size_t k) const { return z + k; }
  size_t global_w(size_t l) const { return w + l; }

protected:
  // construction and assignment--perform shallow copy of (sub)array
  explicit preview(container_type* array) : array(array), x(0), y(0), z(0), w(0), nx(array->size_x()), ny(array->size_y()), nz(array->size_z()), nw(array->size_w()) {}
  explicit preview(container_type* array, size_t x, size_t y, size_t z, size_t w, size_t nx, size_t ny, size_t nz, size_t nw) : array(array), x(x), y(y), z(z), w(w), nx(nx), ny(ny), nz(nz), nw(nw) {}
  preview& operator=(container_type* a)
  {
    array = a;
    x = y = z = w = 0;
    nx = a->nx;
    ny = a->ny;
    nz = a->nz;
    nw = a->nw;
    return *this;
  }

  // global index bounds for iterators
  size_t min_x() const { return x; }
  size_t max_x() const { return x + nx; }
  size_t min_y() const { return y; }
  size_t max_y() const { return y + ny; }
  size_t min_z() const { return z; }
  size_t max_z() const { return z + nz; }
  size_t min_w() const { return w; }
  size_t max_w() const { return w + nw; }

  container_type* array; // underlying container
  size_t x, y, z, w;     // offset into array
  size_t nx, ny, nz, nw; // dimensions of subarray
};

// generic read-only view into a rectangular subset of a 4D array
template <class Container>
class const_view : public preview<Container> {
public:
  typedef Container container_type;
  typedef typename container_type::value_type value_type;
  typedef typename zfp::internal::dim4::const_reference<const_view> const_reference;
  typedef typename zfp::internal::dim4::const_pointer<const_view> const_pointer;
  typedef typename zfp::internal::dim4::const_iterator<const_view> const_iterator;

  // construction--perform shallow copy of (sub)array
  const_view(container_type* array) : preview<Container>(array) {}
  const_view(container_type* array, size_t x, size_t y, size_t z, size_t w, size_t nx, size_t ny, size_t nz, size_t nw) : preview<Container>(array, x, y, z, w, nx, ny, nz, nw) {}

  // dimensions of (sub)array
  size_t size_x() const { return nx; }
  size_t size_y() const { return ny; }
  size_t size_z() const { return nz; }
  size_t size_w() const { return nw; }

  // (i, j, k, l) inspector
  const_reference operator()(size_t i, size_t j, size_t k, size_t l) const { return const_reference(this, x + i, y + j, z + k, w + l); }

  // random access iterators
  const_iterator cbegin() const { return const_iterator(this, x, y, z, w); }
  const_iterator cend() const { return const_iterator(this, x, y, z, w + nw); }
  const_iterator begin() const { return cbegin(); }
  const_iterator end() const { return cend(); }

protected:
  friend class zfp::internal::dim4::const_handle<const_view>;
  friend class zfp::internal::dim4::const_pointer<const_view>;
  friend class zfp::internal::dim4::const_iterator<const_view>;

  using preview<Container>::min_x;
  using preview<Container>::max_x;
  using preview<Container>::min_y;
  using preview<Container>::max_y;
  using preview<Container>::min_z;
  using preview<Container>::max_z;
  using preview<Container>::min_w;
  using preview<Container>::max_w;
  using preview<Container>::array;
  using preview<Container>::x;
  using preview<Container>::y;
  using preview<Container>::z;
  using preview<Container>::w;
  using preview<Container>::nx;
  using preview<Container>::ny;
  using preview<Container>::nz;
  using preview<Container>::nw;

  // inspector
  value_type get(size_t x, size_t y, size_t z, size_t w) const { return array->get(x, y, z, w); }
};

// generic read-write view into a rectangular subset of a 4D array
template <class Container>
class view : public const_view<Container> {
public:
  typedef Container container_type;
  typedef typename container_type::value_type value_type;
  typedef typename zfp::internal::dim4::const_reference<view> const_reference;
  typedef typename zfp::internal::dim4::const_pointer<view> const_pointer;
  typedef typename zfp::internal::dim4::const_iterator<view> const_iterator;
  typedef typename zfp::internal::dim4::reference<view> reference;
  typedef typename zfp::internal::dim4::pointer<view> pointer;
  typedef typename zfp::internal::dim4::iterator<view> iterator;

  // construction--perform shallow copy of (sub)array
  view(container_type* array) : const_view<Container>(array) {}
  view(container_type* array, size_t x, size_t y, size_t z, size_t w, size_t nx, size_t ny, size_t nz, size_t nw) : const_view<Container>(array, x, y, z, w, nx, ny, nz, nw) {}

  // (i, j, k, l) inspector
  const_reference operator()(size_t i, size_t j, size_t k, size_t l) const { return const_reference(this, x + i, y + j, z + k, w + l); }

  // (i, j, k, l) mutator
  reference operator()(size_t i, size_t j, size_t k, size_t l) { return reference(this, x + i, y + j, z + k, w + l); }

  // random access iterators
  const_iterator cbegin() const { return const_iterator(this, x, y, z, w); }
  const_iterator cend() const { return const_iterator(this, x, y, z, w + nw); }
  const_iterator begin() const { return cbegin(); }
  const_iterator end() const { return cend(); }
  iterator begin() { return iterator(this, x, y, z, w); }
  iterator end() { return iterator(this, x, y, z, w + nw); }

protected:
  friend class zfp::internal::dim4::const_handle<view>;
  friend class zfp::internal::dim4::const_pointer<view>;
  friend class zfp::internal::dim4::const_iterator<view>;
  friend class zfp::internal::dim4::reference<view>;
  friend class zfp::internal::dim4::pointer<view>;
  friend class zfp::internal::dim4::iterator<view>;

  using const_view<Container>::min_x;
  using const_view<Container>::max_x;
  using const_view<Container>::min_y;
  using const_view<Container>::max_y;
  using const_view<Container>::min_z;
  using const_view<Container>::max_z;
  using const_view<Container>::min_w;
  using const_view<Container>::max_w;
  using const_view<Container>::get;
  using const_view<Container>::array;
  using const_view<Container>::x;
  using const_view<Container>::y;
  using const_view<Container>::z;
  using const_view<Container>::w;
  using const_view<Container>::nx;
  using const_view<Container>::ny;
  using const_view<Container>::nz;
  using const_view<Container>::nw;

  // mutator
  void set(size_t x, size_t y, size_t z, size_t w, value_type val) { array->set(x, y, z, w, val); }

  // in-place updates
  void add(size_t x, size_t y, size_t z, size_t w, value_type val) { array->add(x, y, z, w, val); }
  void sub(size_t x, size_t y, size_t z, size_t w, value_type val) { array->sub(x, y, z, w, val); }
  void mul(size_t x, size_t y, size_t z, size_t w, value_type val) { array->mul(x, y, z, w, val); }
  void div(size_t x, size_t y, size_t z, size_t w, value_type val) { array->div(x, y, z, w, val); }
};

// flat view of 4D array (operator[] returns scalar)
template <class Container>
class flat_view : public view<Container> {
public:
  typedef Container container_type;
  typedef typename container_type::value_type value_type;
  typedef typename zfp::internal::dim4::const_reference<flat_view> const_reference;
  typedef typename zfp::internal::dim4::const_pointer<flat_view> const_pointer;
  typedef typename zfp::internal::dim4::reference<flat_view> reference;
  typedef typename zfp::internal::dim4::pointer<flat_view> pointer;

  // construction--perform shallow copy of (sub)array
  flat_view(container_type* array) : view<Container>(array) {}
  flat_view(container_type* array, size_t x, size_t y, size_t z, size_t w, size_t nx, size_t ny, size_t nz, size_t nw) : view<Container>(array, x, y, z, w, nx, ny, nz, nw) {}

  // convert (i, j, k, l) index to flat index
  size_t index(size_t i, size_t j, size_t k, size_t l) const { return i + nx * (j + ny * (k + nz * l)); }

  // convert flat index to (i, j, k, l) index
  void ijkl(size_t& i, size_t& j, size_t& k, size_t& l, size_t index) const
  {
    i = index % nx; index /= nx;
    j = index % ny; index /= ny;
    k = index % nz; index /= nz;
    l = index;
  }

  // flat index [] inspector
  const_reference operator[](size_t index) const
  {
    size_t i, j, k, l;
    ijkl(i, j, k, l, index);
    return const_reference(this, x + i, y + j, z + k, w + l);
  }

  // flat index [] mutator
  reference operator[](size_t index)
  {
    size_t i, j, k, l;
    ijkl(i, j, k, l, index);
    return reference(this, x + i, y + j, z + k, w + l);
  }

  // (i, j, k, l) inspector
  const_reference operator()(size_t i, size_t j, size_t k, size_t l) const { return const_reference(this, x + i, y + j, z + k, w + l); }

  // (i, j, k, l) mutator
  reference operator()(size_t i, size_t j, size_t k, size_t l) { return reference(this, x + i, y + j, z + k, w + l); }

protected:
  friend class zfp::internal::dim4::const_handle<flat_view>;
  friend class zfp::internal::dim4::const_pointer<flat_view>;
  friend class zfp::internal::dim4::reference<flat_view>;
  friend class zfp::internal::dim4::pointer<flat_view>;

  using view<Container>::array;
  using view<Container>::x;
  using view<Container>::y;
  using view<Container>::z;
  using view<Container>::w;
  using view<Container>::nx;
  using view<Container>::ny;
  using view<Container>::nz;
  using view<Container>::nw;

  // inspector
  value_type get(size_t x, size_t y, size_t z, size_t w) const { return array->get(x, y, z, w); }

  // mutator
  void set(size_t x, size_t y, size_t z, size_t w, value_type val) { array->set(x, y, z, w, val); }

  // in-place updates
  void add(size_t x, size_t y, size_t z, size_t w, value_type val) { array->add(x, y, z, w, val); }
  void sub(size_t x, size_t y, size_t z, size_t w, value_type val) { array->sub(x, y, z, w, val); }
  void mul(size_t x, size_t y, size_t z, size_t w, value_type val) { array->mul(x, y, z, w, val); }
  void div(size_t x, size_t y, size_t z, size_t w, value_type val) { array->div(x, y, z, w, val); }
};

// forward declaration of friends
template <class Container> class nested_view1;
template <class Container> class nested_view2;
template <class Container> class nested_view3;
template <class Container> class nested_view4;

// nested view into a 1D rectangular subset of a 4D array
template <class Container>
class nested_view1 : public preview<Container> {
public:
  typedef Container container_type;
  typedef typename container_type::value_type value_type;
  typedef typename zfp::internal::dim4::const_reference<nested_view1> const_reference;
  typedef typename zfp::internal::dim4::const_pointer<nested_view1> const_pointer;
  typedef typename zfp::internal::dim4::reference<nested_view1> reference;
  typedef typename zfp::internal::dim4::pointer<nested_view1> pointer;

  // dimensions of (sub)array
  size_t size_x() const { return nx; }

  // [i] inspector and mutator
  const_reference operator[](size_t index) const { return const_reference(this, x + index, y, z, w); }
  reference operator[](size_t index) { return reference(this, x + index, y, z, w); }

  // (i) inspector and mutator
  const_reference operator()(size_t i) const { return const_reference(this, x + i, y, z, w); }
  reference operator()(size_t i) { return reference(this, x + i, y, z, w); }

protected:
  friend class zfp::internal::dim4::const_handle<nested_view1>;
  friend class zfp::internal::dim4::const_pointer<nested_view1>;
  friend class zfp::internal::dim4::reference<nested_view1>;
  friend class zfp::internal::dim4::pointer<nested_view1>;

  using preview<Container>::array;
  using preview<Container>::x;
  using preview<Container>::y;
  using preview<Container>::z;
  using preview<Container>::w;
  using preview<Container>::nx;
  using preview<Container>::ny;
  using preview<Container>::nz;
  using preview<Container>::nw;

  // construction--perform shallow copy of (sub)array
  friend class nested_view2<Container>;
  explicit nested_view1(container_type* array) : preview<Container>(array) {}
  explicit nested_view1(container_type* array, size_t x, size_t y, size_t z, size_t w, size_t nx, size_t ny, size_t nz, size_t nw) : preview<Container>(array, x, y, z, w, nx, ny, nz, nw) {}

  // inspector
  value_type get(size_t x, size_t y, size_t z, size_t w) const { return array->get(x, y, z, w); }

  // mutator
  void set(size_t x, size_t y, size_t z, size_t w, value_type val) { array->set(x, y, z, w, val); }

  // in-place updates
  void add(size_t x, size_t y, size_t z, size_t w, value_type val) { array->add(x, y, z, w, val); }
  void sub(size_t x, size_t y, size_t z, size_t w, value_type val) { array->sub(x, y, z, w, val); }
  void mul(size_t x, size_t y, size_t z, size_t w, value_type val) { array->mul(x, y, z, w, val); }
  void div(size_t x, size_t y, size_t z, size_t w, value_type val) { array->div(x, y, z, w, val); }
};

// nested view into a 2D rectangular subset of a 4D array
template <class Container>
class nested_view2 : public preview<Container> {
public:
  typedef Container container_type;
  typedef typename container_type::value_type value_type;
  typedef typename zfp::internal::dim4::const_reference<nested_view2> const_reference;
  typedef typename zfp::internal::dim4::const_pointer<nested_view2> const_pointer;
  typedef typename zfp::internal::dim4::reference<nested_view2> reference;
  typedef typename zfp::internal::dim4::pointer<nested_view2> pointer;

  // construction--perform shallow copy of (sub)array
  nested_view2(container_type* array) : preview<Container>(array) {}
  nested_view2(container_type* array, size_t x, size_t y, size_t z, size_t w, size_t nx, size_t ny, size_t nz, size_t nw) : preview<Container>(array, x, y, z, w, nx, ny, nz, nw) {}

  // dimensions of (sub)array
  size_t size_x() const { return nx; }
  size_t size_y() const { return ny; }

  // 1D view
  nested_view1<Container> operator[](size_t index) const { return nested_view1<Container>(array, x, y + index, z, w, nx, 1, 1, 1); }

  // (i, j) inspector and mutator
  const_reference operator()(size_t i, size_t j) const { return const_reference(this, x + i, y + j, z, w); }
  reference operator()(size_t i, size_t j) { return reference(this, x + i, y + j, z, w); }

protected:
  friend class zfp::internal::dim4::const_handle<nested_view2>;
  friend class zfp::internal::dim4::const_pointer<nested_view2>;
  friend class zfp::internal::dim4::reference<nested_view2>;
  friend class zfp::internal::dim4::pointer<nested_view2>;

  using preview<Container>::array;
  using preview<Container>::x;
  using preview<Container>::y;
  using preview<Container>::z;
  using preview<Container>::w;
  using preview<Container>::nx;
  using preview<Container>::ny;
  using preview<Container>::nz;
  using preview<Container>::nw;

  // inspector
  value_type get(size_t x, size_t y, size_t z, size_t w) const { return array->get(x, y, z, w); }

  // mutator
  void set(size_t x, size_t y, size_t z, size_t w, value_type val) { array->set(x, y, z, w, val); }

  // in-place updates
  void add(size_t x, size_t y, size_t z, size_t w, value_type val) { array->add(x, y, z, w, val); }
  void sub(size_t x, size_t y, size_t z, size_t w, value_type val) { array->sub(x, y, z, w, val); }
  void mul(size_t x, size_t y, size_t z, size_t w, value_type val) { array->mul(x, y, z, w, val); }
  void div(size_t x, size_t y, size_t z, size_t w, value_type val) { array->div(x, y, z, w, val); }
};

// nested view into a 3D rectangular subset of a 4D array
template <class Container>
class nested_view3 : public preview<Container> {
public:
  typedef Container container_type;
  typedef typename container_type::value_type value_type;
  typedef typename zfp::internal::dim4::const_reference<nested_view3> const_reference;
  typedef typename zfp::internal::dim4::const_pointer<nested_view3> const_pointer;
  typedef typename zfp::internal::dim4::reference<nested_view3> reference;
  typedef typename zfp::internal::dim4::pointer<nested_view3> pointer;

  // construction--perform shallow copy of (sub)array
  nested_view3(container_type* array) : preview<Container>(array) {}
  nested_view3(container_type* array, size_t x, size_t y, size_t z, size_t w, size_t nx, size_t ny, size_t nz, size_t nw) : preview<Container>(array, x, y, z, w, nx, ny, nz, nw) {}

  // dimensions of (sub)array
  size_t size_x() const { return nx; }
  size_t size_y() const { return ny; }
  size_t size_z() const { return nz; }

  // 2D view
  nested_view2<Container> operator[](size_t index) const { return nested_view2<Container>(array, x, y, z + index, w, nx, ny, 1, 1); }

  // (i, j, k) inspector and mutator
  const_reference operator()(size_t i, size_t j, size_t k) const { return const_reference(this, x + i, y + j, z + k, w); }
  reference operator()(size_t i, size_t j, size_t k) { return reference(this, x + i, y + j, z + k, w); }

protected:
  friend class zfp::internal::dim4::const_handle<nested_view3>;
  friend class zfp::internal::dim4::const_pointer<nested_view3>;
  friend class zfp::internal::dim4::reference<nested_view3>;
  friend class zfp::internal::dim4::pointer<nested_view3>;

  using preview<Container>::array;
  using preview<Container>::x;
  using preview<Container>::y;
  using preview<Container>::z;
  using preview<Container>::w;
  using preview<Container>::nx;
  using preview<Container>::ny;
  using preview<Container>::nz;
  using preview<Container>::nw;

  // inspector
  value_type get(size_t x, size_t y, size_t z, size_t w) const { return array->get(x, y, z, w); }

  // mutator
  void set(size_t x, size_t y, size_t z, size_t w, value_type val) { array->set(x, y, z, w, val); }

  // in-place updates
  void add(size_t x, size_t y, size_t z, size_t w, value_type val) { array->add(x, y, z, w, val); }
  void sub(size_t x, size_t y, size_t z, size_t w, value_type val) { array->sub(x, y, z, w, val); }
  void mul(size_t x, size_t y, size_t z, size_t w, value_type val) { array->mul(x, y, z, w, val); }
  void div(size_t x, size_t y, size_t z, size_t w, value_type val) { array->div(x, y, z, w, val); }
};

// nested view into a 4D rectangular subset of a 4D array
template <class Container>
class nested_view4 : public preview<Container> {
public:
  typedef Container container_type;
  typedef typename container_type::value_type value_type;
  typedef typename zfp::internal::dim4::const_reference<nested_view4> const_reference;
  typedef typename zfp::internal::dim4::const_pointer<nested_view4> const_pointer;
  typedef typename zfp::internal::dim4::reference<nested_view4> reference;
  typedef typename zfp::internal::dim4::pointer<nested_view4> pointer;

  // construction--perform shallow copy of (sub)array
  nested_view4(container_type* array) : preview<Container>(array) {}
  nested_view4(container_type* array, size_t x, size_t y, size_t z, size_t w, size_t nx, size_t ny, size_t nz, size_t nw) : preview<Container>(array, x, y, z, w, nx, ny, nz, nw) {}

  // dimensions of (sub)array
  size_t size_x() const { return nx; }
  size_t size_y() const { return ny; }
  size_t size_z() const { return nz; }
  size_t size_w() const { return nw; }

  // 3D view
  nested_view3<Container> operator[](size_t index) const { return nested_view3<Container>(array, x, y, z, w + index, nx, ny, nz, 1); }

  // (i, j, k, l) inspector and mutator
  const_reference operator()(size_t i, size_t j, size_t k, size_t l) const { return const_reference(this, x + i, y + j, z + k, w + l); }
  reference operator()(size_t i, size_t j, size_t k, size_t l) { return reference(this, x + i, y + j, z + k, w + l); }

protected:
  friend class zfp::internal::dim4::const_handle<nested_view4>;
  friend class zfp::internal::dim4::const_pointer<nested_view4>;
  friend class zfp::internal::dim4::reference<nested_view4>;
  friend class zfp::internal::dim4::pointer<nested_view4>;

  using preview<Container>::array;
  using preview<Container>::x;
  using preview<Container>::y;
  using preview<Container>::z;
  using preview<Container>::w;
  using preview<Container>::nx;
  using preview<Container>::ny;
  using preview<Container>::nz;
  using preview<Container>::nw;

  // inspector
  value_type get(size_t x, size_t y, size_t z, size_t w) const { return array->get(x, y, z, w); }

  // mutator
  void set(size_t x, size_t y, size_t z, size_t w, value_type val) { array->set(x, y, z, w, val); }

  // in-place updates
  void add(size_t x, size_t y, size_t z, size_t w, value_type val) { array->add(x, y, z, w, val); }
  void sub(size_t x, size_t y, size_t z, size_t w, value_type val) { array->sub(x, y, z, w, val); }
  void mul(size_t x, size_t y, size_t z, size_t w, value_type val) { array->mul(x, y, z, w, val); }
  void div(size_t x, size_t y, size_t z, size_t w, value_type val) { array->div(x, y, z, w, val); }
};

// thread-safe read-only view of 4D (sub)array with private cache
template <class Container>
class private_const_view : public preview<Container> {
public:
  typedef Container container_type;
  typedef typename container_type::value_type value_type;
  typedef typename container_type::store_type store_type;
  typedef typename zfp::internal::dim4::const_reference<private_const_view> const_reference;
  typedef typename zfp::internal::dim4::const_pointer<private_const_view> const_pointer;
  typedef typename zfp::internal::dim4::const_iterator<private_const_view> const_iterator;

  // construction--perform shallow copy of (sub)array
  private_const_view(container_type* array, size_t cache_size = 0) :
    preview<Container>(array),
    cache(array->store, cache_size ? cache_size : array->cache.size())
  {
    array->store.reference();
  }
  private_const_view(container_type* array, size_t x, size_t y, size_t z, size_t w, size_t nx, size_t ny, size_t nz, size_t nw, size_t cache_size = 0) :
    preview<Container>(array, x, y, z, w, nx, ny, nz, nw),
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
  size_t size_z() const { return nz; }
  size_t size_w() const { return nw; }

  // cache size in number of bytes
  size_t cache_size() const { return cache.size(); }

  // set minimum cache size in bytes (array dimensions must be known)
  void set_cache_size(size_t bytes) { cache.resize(bytes); }

  // empty cache without compressing modified cached blocks
  void clear_cache() const { cache.clear(); }

  // (i, j, k) inspector
  const_reference operator()(size_t i, size_t j, size_t k, size_t l) const { return const_reference(this, x + i, y + j, z + k, w + l); }

  // random access iterators
  const_iterator cbegin() const { return const_iterator(this, x, y, z, w); }
  const_iterator cend() const { return const_iterator(this, x, y, z, w + nw); }
  const_iterator begin() const { return cbegin(); }
  const_iterator end() const { return cend(); }

protected:
  friend class zfp::internal::dim4::const_handle<private_const_view>;
  friend class zfp::internal::dim4::const_pointer<private_const_view>;
  friend class zfp::internal::dim4::const_iterator<private_const_view>;

  using preview<Container>::min_x;
  using preview<Container>::max_x;
  using preview<Container>::min_y;
  using preview<Container>::max_y;
  using preview<Container>::min_z;
  using preview<Container>::max_z;
  using preview<Container>::min_w;
  using preview<Container>::max_w;
  using preview<Container>::array;
  using preview<Container>::x;
  using preview<Container>::y;
  using preview<Container>::z;
  using preview<Container>::w;
  using preview<Container>::nx;
  using preview<Container>::ny;
  using preview<Container>::nz;
  using preview<Container>::nw;

  // inspector
  value_type get(size_t x, size_t y, size_t z, size_t w) const { return cache.get(x, y, z, w); }

  BlockCache4<value_type, store_type> cache; // cache of decompressed blocks
};

// thread-safe read-write view of private 4D (sub)array
template <class Container>
class private_view : public private_const_view<Container> {
public:
  typedef Container container_type;
  typedef typename container_type::value_type value_type;
  typedef typename zfp::internal::dim4::const_reference<private_view> const_reference;
  typedef typename zfp::internal::dim4::const_pointer<private_view> const_pointer;
  typedef typename zfp::internal::dim4::const_iterator<private_view> const_iterator;
  typedef typename zfp::internal::dim4::reference<private_view> reference;
  typedef typename zfp::internal::dim4::pointer<private_view> pointer;
  typedef typename zfp::internal::dim4::iterator<private_view> iterator;

  // construction--perform shallow copy of (sub)array
  private_view(container_type* array, size_t cache_size = 0) : private_const_view<Container>(array, cache_size) {}
  private_view(container_type* array, size_t x, size_t y, size_t z, size_t w, size_t nx, size_t ny, size_t nz, size_t nw, size_t cache_size = 0) : private_const_view<Container>(array, x, y, z, w, nx, ny, nz, nw, cache_size) {}

  // partition view into count block-aligned pieces, with 0 <= index < count
  void partition(size_t index, size_t count)
  {
    if (std::max(nx, ny) > std::max(nz, nw)) {
      if (nx > ny)
        partition(x, nx, index, count);
      else
        partition(y, ny, index, count);
    }
    else {
      if (nz > nw)
        partition(z, nz, index, count);
      else
        partition(w, nw, index, count);
    }
  }

  // flush cache by compressing all modified cached blocks
  void flush_cache() const { cache.flush(); }

  // (i, j, k, l) inspector
  const_reference operator()(size_t i, size_t j, size_t k, size_t l) const { return const_reference(this, x + i, y + j, z + k, w + l); }

  // (i, j, k, l) mutator
  reference operator()(size_t i, size_t j, size_t k, size_t l) { return reference(this, x + i, y + j, z + k, w + l); }

  // random access iterators
  const_iterator cbegin() const { return const_iterator(this, x, y, z, w); }
  const_iterator cend() const { return const_iterator(this, x, y, z, w + nw); }
  const_iterator begin() const { return cbegin(); }
  const_iterator end() const { return cend(); }
  iterator begin() { return iterator(this, x, y, z, w); }
  iterator end() { return iterator(this, x, y, z, w + nw); }

protected:
  friend class zfp::internal::dim4::const_handle<private_view>;
  friend class zfp::internal::dim4::const_pointer<private_view>;
  friend class zfp::internal::dim4::const_iterator<private_view>;
  friend class zfp::internal::dim4::reference<private_view>;
  friend class zfp::internal::dim4::pointer<private_view>;
  friend class zfp::internal::dim4::iterator<private_view>;

  using private_const_view<Container>::min_x;
  using private_const_view<Container>::max_x;
  using private_const_view<Container>::min_y;
  using private_const_view<Container>::max_y;
  using private_const_view<Container>::min_z;
  using private_const_view<Container>::max_z;
  using private_const_view<Container>::min_w;
  using private_const_view<Container>::max_w;
  using private_const_view<Container>::get;
  using private_const_view<Container>::array;
  using private_const_view<Container>::x;
  using private_const_view<Container>::y;
  using private_const_view<Container>::z;
  using private_const_view<Container>::w;
  using private_const_view<Container>::nx;
  using private_const_view<Container>::ny;
  using private_const_view<Container>::nz;
  using private_const_view<Container>::nw;
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
  void set(size_t x, size_t y, size_t z, size_t w, value_type val) { cache.set(x, y, z, w, val); }

  // in-place updates
  void add(size_t x, size_t y, size_t z, size_t w, value_type val) { cache.ref(x, y, z, w) += val; }
  void sub(size_t x, size_t y, size_t z, size_t w, value_type val) { cache.ref(x, y, z, w) -= val; }
  void mul(size_t x, size_t y, size_t z, size_t w, value_type val) { cache.ref(x, y, z, w) *= val; }
  void div(size_t x, size_t y, size_t z, size_t w, value_type val) { cache.ref(x, y, z, w) /= val; }
};

} // dim4
} // internal
} // zfp

#endif
