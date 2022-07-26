#ifndef ZFP_VIEW1_HPP
#define ZFP_VIEW1_HPP

// 1D array views

namespace zfp {
namespace internal {
namespace dim1 {

// abstract view of 1D array (base class)
template <class Container>
class preview {
public:
  typedef Container container_type;
  typedef typename container_type::value_type value_type;

  // rate in bits per value
  double rate() const { return array->rate(); }

  // dimensions of (sub)array
  size_t size() const { return nx; }

  // local to global array index
  size_t global_x(size_t i) const { return x + i; }

protected:
  // construction and assignment--perform shallow copy of (sub)array
  explicit preview(container_type* array) : array(array), x(0), nx(array->size_x()) {}
  explicit preview(container_type* array, size_t x, size_t nx) : array(array), x(x), nx(nx) {}
  preview& operator=(container_type* a)
  {
    array = a;
    x = 0;
    nx = a->nx;
    return *this;
  }

  // global index bounds for iterators
  size_t min_x() const { return x; }
  size_t max_x() const { return x + nx; }

  container_type* array; // underlying container
  size_t x;              // offset into array
  size_t nx;             // dimensions of subarray
};

// generic read-only view into a rectangular subset of a 1D array
template <class Container>
class const_view : public preview<Container> {
public:
  typedef Container container_type;
  typedef typename container_type::value_type value_type;
  typedef typename zfp::internal::dim1::const_reference<const_view> const_reference;
  typedef typename zfp::internal::dim1::const_pointer<const_view> const_pointer;
  typedef typename zfp::internal::dim1::const_iterator<const_view> const_iterator;

  // construction--perform shallow copy of (sub)array
  const_view(container_type* array) : preview<Container>(array) {}
  const_view(container_type* array, size_t x, size_t nx) : preview<Container>(array, x, nx) {}

  // dimensions of (sub)array
  size_t size_x() const { return nx; }

  // [i] inspector
  const_reference operator[](size_t index) const { return const_reference(this, x + index); }

  // (i) inspector
  const_reference operator()(size_t i) const { return const_reference(this, x + i); }

  // random access iterators
  const_iterator cbegin() const { return const_iterator(this, x); }
  const_iterator cend() const { return const_iterator(this, x + nx); }
  const_iterator begin() const { return cbegin(); }
  const_iterator end() const { return cend(); }

protected:
  friend class zfp::internal::dim1::const_handle<const_view>;
  friend class zfp::internal::dim1::const_pointer<const_view>;
  friend class zfp::internal::dim1::const_iterator<const_view>;

  using preview<Container>::min_x;
  using preview<Container>::max_x;
  using preview<Container>::array;
  using preview<Container>::x;
  using preview<Container>::nx;

  // inspector
  value_type get(size_t x) const { return array->get(x); }
};

// generic read-write view into a rectangular subset of a 1D array
template <class Container>
class view : public const_view<Container> {
public:
  typedef Container container_type;
  typedef typename container_type::value_type value_type;
  typedef typename zfp::internal::dim1::const_reference<view> const_reference;
  typedef typename zfp::internal::dim1::const_pointer<view> const_pointer;
  typedef typename zfp::internal::dim1::const_iterator<view> const_iterator;
  typedef typename zfp::internal::dim1::reference<view> reference;
  typedef typename zfp::internal::dim1::pointer<view> pointer;
  typedef typename zfp::internal::dim1::iterator<view> iterator;

  // construction--perform shallow copy of (sub)array
  view(container_type* array) : const_view<Container>(array) {}
  view(container_type* array, size_t x, size_t nx) : const_view<Container>(array, x, nx) {}

  // [i] inspector
  const_reference operator[](size_t index) const { return const_reference(this, x + index); }

  // (i) inspector
  const_reference operator()(size_t i) const { return const_reference(this, x + i); }

  // [i] mutator
  reference operator[](size_t index) { return reference(this, x + index); }

  // (i) mutator
  reference operator()(size_t i) { return reference(this, x + i); }

  // random access iterators
  const_iterator cbegin() const { return const_iterator(this, x); }
  const_iterator cend() const { return const_iterator(this, x + nx); }
  const_iterator begin() const { return cbegin(); }
  const_iterator end() const { return cend(); }
  iterator begin() { return iterator(this, x); }
  iterator end() { return iterator(this, x + nx); }

protected:
  friend class zfp::internal::dim1::const_handle<view>;
  friend class zfp::internal::dim1::const_pointer<view>;
  friend class zfp::internal::dim1::const_iterator<view>;
  friend class zfp::internal::dim1::reference<view>;
  friend class zfp::internal::dim1::pointer<view>;
  friend class zfp::internal::dim1::iterator<view>;

  using const_view<Container>::min_x;
  using const_view<Container>::max_x;
  using const_view<Container>::get;
  using const_view<Container>::array;
  using const_view<Container>::x;
  using const_view<Container>::nx;

  // mutator
  void set(size_t x, value_type val) { array->set(x, val); }

  // in-place updates
  void add(size_t x, value_type val) { array->add(x, val); }
  void sub(size_t x, value_type val) { array->sub(x, val); }
  void mul(size_t x, value_type val) { array->mul(x, val); }
  void div(size_t x, value_type val) { array->div(x, val); }
};

// thread-safe read-only view of 1D (sub)array with private cache
template <class Container>
class private_const_view : public preview<Container> {
public:
  typedef Container container_type;
  typedef typename container_type::value_type value_type;
  typedef typename container_type::store_type store_type;
  typedef typename zfp::internal::dim1::const_reference<private_const_view> const_reference;
  typedef typename zfp::internal::dim1::const_pointer<private_const_view> const_pointer;
  typedef typename zfp::internal::dim1::const_iterator<private_const_view> const_iterator;

  // construction--perform shallow copy of (sub)array
  private_const_view(container_type* array, size_t cache_size = 0) :
    preview<Container>(array),
    cache(array->store, cache_size ? cache_size : array->cache.size())
  {
    array->store.reference();
  }
  private_const_view(container_type* array, size_t x, size_t nx, size_t cache_size = 0) :
    preview<Container>(array, x, nx),
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

  // cache size in number of bytes
  size_t cache_size() const { return cache.size(); }

  // set minimum cache size in bytes (array dimensions must be known)
  void set_cache_size(size_t bytes) { cache.resize(bytes); }

  // empty cache without compressing modified cached blocks
  void clear_cache() const { cache.clear(); }

  // (i) inspector
  const_reference operator()(size_t i) const { return const_reference(this, x + i); }

  // random access iterators
  const_iterator cbegin() const { return const_iterator(this, x); }
  const_iterator cend() const { return const_iterator(this, x + nx); }
  const_iterator begin() const { return cbegin(); }
  const_iterator end() const { return cend(); }

protected:
  friend class zfp::internal::dim1::const_handle<private_const_view>;
  friend class zfp::internal::dim1::const_pointer<private_const_view>;
  friend class zfp::internal::dim1::const_iterator<private_const_view>;

  using preview<Container>::min_x;
  using preview<Container>::max_x;
  using preview<Container>::array;
  using preview<Container>::x;
  using preview<Container>::nx;

  // inspector
  value_type get(size_t x) const { return cache.get(x); }

  BlockCache1<value_type, store_type> cache; // cache of decompressed blocks
};

// thread-safe read-write view of private 1D (sub)array
template <class Container>
class private_view : public private_const_view<Container> {
public:
  typedef Container container_type;
  typedef typename container_type::value_type value_type;
  typedef typename zfp::internal::dim1::const_reference<private_view> const_reference;
  typedef typename zfp::internal::dim1::const_pointer<private_view> const_pointer;
  typedef typename zfp::internal::dim1::const_iterator<private_view> const_iterator;
  typedef typename zfp::internal::dim1::reference<private_view> reference;
  typedef typename zfp::internal::dim1::pointer<private_view> pointer;
  typedef typename zfp::internal::dim1::iterator<private_view> iterator;

  // construction--perform shallow copy of (sub)array
  private_view(container_type* array, size_t cache_size = 0) : private_const_view<Container>(array, cache_size) {}
  private_view(container_type* array, size_t x, size_t nx, size_t cache_size = 0) : private_const_view<Container>(array, x, nx, cache_size) {}

  // partition view into count block-aligned pieces, with 0 <= index < count
  void partition(size_t index, size_t count)
  {
    partition(x, nx, index, count);
  }

  // flush cache by compressing all modified cached blocks
  void flush_cache() const { cache.flush(); }

  // (i) inspector
  const_reference operator()(size_t i) const { return const_reference(this, x + i); }

  // (i) mutator
  reference operator()(size_t i) { return reference(this, x + i); }

  // random access iterators
  const_iterator cbegin() const { return const_iterator(this, x); }
  const_iterator cend() const { return const_iterator(this, x + nx); }
  const_iterator begin() const { return cbegin(); }
  const_iterator end() const { return cend(); }
  iterator begin() { return iterator(this, x); }
  iterator end() { return iterator(this, x + nx); }

protected:
  friend class zfp::internal::dim1::const_handle<private_view>;
  friend class zfp::internal::dim1::const_pointer<private_view>;
  friend class zfp::internal::dim1::const_iterator<private_view>;
  friend class zfp::internal::dim1::reference<private_view>;
  friend class zfp::internal::dim1::pointer<private_view>;
  friend class zfp::internal::dim1::iterator<private_view>;

  using private_const_view<Container>::min_x;
  using private_const_view<Container>::max_x;
  using private_const_view<Container>::get;
  using private_const_view<Container>::array;
  using private_const_view<Container>::x;
  using private_const_view<Container>::nx;
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
  void set(size_t x, value_type val) { cache.set(x, val); }

  // in-place updates
  void add(size_t x, value_type val) { cache.ref(x) += val; }
  void sub(size_t x, value_type val) { cache.ref(x) -= val; }
  void mul(size_t x, value_type val) { cache.ref(x) *= val; }
  void div(size_t x, value_type val) { cache.ref(x) /= val; }
};

} // dim1
} // internal
} // zfp

#endif
