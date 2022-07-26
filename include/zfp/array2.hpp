#ifndef ZFP_ARRAY2_HPP
#define ZFP_ARRAY2_HPP

#include <cstddef>
#include <cstring>
#include <iterator>
#include "zfp/array.hpp"
#include "zfp/index.hpp"
#include "zfp/codec/zfpcodec.hpp"
#include "zfp/internal/array/cache2.hpp"
#include "zfp/internal/array/handle2.hpp"
#include "zfp/internal/array/iterator2.hpp"
#include "zfp/internal/array/pointer2.hpp"
#include "zfp/internal/array/reference2.hpp"
#include "zfp/internal/array/store2.hpp"
#include "zfp/internal/array/view2.hpp"

namespace zfp {

// compressed 2D array of scalars
template <
  typename Scalar,
  class Codec = zfp::codec::zfp2<Scalar>,
  class Index = zfp::index::implicit
>
class array2 : public array {
public:
  // types utilized by nested classes
  typedef array2 container_type;
  typedef Scalar value_type;
  typedef Codec codec_type;
  typedef Index index_type;
  typedef zfp::internal::BlockStore2<value_type, codec_type, index_type> store_type;
  typedef zfp::internal::BlockCache2<value_type, store_type> cache_type;
  typedef typename Codec::header header;

  // accessor classes
  typedef zfp::internal::dim2::const_reference<array2> const_reference;
  typedef zfp::internal::dim2::const_pointer<array2> const_pointer;
  typedef zfp::internal::dim2::const_iterator<array2> const_iterator;
  typedef zfp::internal::dim2::const_view<array2> const_view;
  typedef zfp::internal::dim2::private_const_view<array2> private_const_view;
  typedef zfp::internal::dim2::reference<array2> reference;
  typedef zfp::internal::dim2::pointer<array2> pointer;
  typedef zfp::internal::dim2::iterator<array2> iterator;
  typedef zfp::internal::dim2::view<array2> view;
  typedef zfp::internal::dim2::flat_view<array2> flat_view;
  typedef zfp::internal::dim2::nested_view1<array2> nested_view1;
  typedef zfp::internal::dim2::nested_view2<array2> nested_view2;
  typedef zfp::internal::dim2::nested_view2<array2> nested_view;
  typedef zfp::internal::dim2::private_view<array2> private_view;

  // default constructor
  array2() :
    array(2, Codec::type),
    cache(store)
  {}

  // constructor of nx * ny array using rate bits per value, at least
  // cache_size bytes of cache, and optionally initialized from flat array p
  array2(size_t nx, size_t ny, double rate, const value_type* p = 0, size_t cache_size = 0) :
    array(2, Codec::type),
    store(nx, ny, zfp_config_rate(rate, true)),
    cache(store, cache_size)
  {
    this->nx = nx;
    this->ny = ny;
    if (p)
      set(p);
  }

  // constructor, from previously-serialized compressed array
  array2(const zfp::array::header& header, const void* buffer = 0, size_t buffer_size_bytes = 0) :
    array(2, Codec::type, header),
    store(header.size_x(), header.size_y(), zfp_config_rate(header.rate(), true)),
    cache(store)
  {
    if (buffer) {
      if (buffer_size_bytes && buffer_size_bytes < store.compressed_size())
        throw zfp::exception("buffer size is smaller than required");
      std::memcpy(store.compressed_data(), buffer, store.compressed_size());
    }
  }

  // copy constructor--performs a deep copy
  array2(const array2& a) :
    array(),
    cache(store)
  {
    deep_copy(a);
  }

  // construction from view--perform deep copy of (sub)array
  template <class View>
  array2(const View& v) :
    array(2, Codec::type),
    store(v.size_x(), v.size_y(), zfp_config_rate(v.rate(), true)),
    cache(store)
  {
    this->nx = v.size_x();
    this->ny = v.size_y();
    // initialize array in its preferred order
    for (iterator it = begin(); it != end(); ++it)
      *it = v(it.i(), it.j());
  }

  // virtual destructor
  virtual ~array2() {}

  // assignment operator--performs a deep copy
  array2& operator=(const array2& a)
  {
    if (this != &a)
      deep_copy(a);
    return *this;
  }

  // total number of elements in array
  size_t size() const { return nx * ny; }

  // array dimensions
  size_t size_x() const { return nx; }
  size_t size_y() const { return ny; }

  // resize the array (all previously stored data will be lost)
  void resize(size_t nx, size_t ny, bool clear = true)
  {
    cache.clear();
    this->nx = nx;
    this->ny = ny;
    store.resize(nx, ny, clear);
  }

  // rate in bits per value
  double rate() const { return store.rate(); }

  // set rate in bits per value
  double set_rate(double rate)
  {
    cache.clear();
    return store.set_rate(rate, true);
  }

  // byte size of array data structure components indicated by mask
  size_t size_bytes(uint mask = ZFP_DATA_ALL) const
  {
    size_t size = 0;
    size += store.size_bytes(mask);
    size += cache.size_bytes(mask);
    if (mask & ZFP_DATA_META)
      size += sizeof(*this);
    return size;
  }

  // number of bytes of compressed data
  size_t compressed_size() const { return store.compressed_size(); }

  // pointer to compressed data for read or write access
  void* compressed_data() const
  {
    cache.flush();
    return store.compressed_data();
  }

  // cache size in number of bytes
  size_t cache_size() const { return cache.size(); }

  // set minimum cache size in bytes (array dimensions must be known)
  void set_cache_size(size_t bytes)
  {
    cache.flush();
    cache.resize(bytes);
  }

  // empty cache without compressing modified cached blocks
  void clear_cache() const { cache.clear(); }

  // flush cache by compressing all modified cached blocks
  void flush_cache() const { cache.flush(); }

  // decompress array and store at p
  void get(value_type* p) const
  {
    const size_t bx = store.block_size_x();
    const size_t by = store.block_size_y();
    const ptrdiff_t sx = 1;
    const ptrdiff_t sy = static_cast<ptrdiff_t>(nx);
    size_t block_index = 0;
    for (size_t j = 0; j < by; j++, p += 4 * sx * ptrdiff_t(nx - bx))
      for (size_t i = 0; i < bx; i++, p += 4)
        cache.get_block(block_index++, p, sx, sy);
  }

  // initialize array by copying and compressing data stored at p
  void set(const value_type* p)
  {
    const size_t bx = store.block_size_x();
    const size_t by = store.block_size_y();
    size_t block_index = 0;
    if (p) {
      // compress data stored at p
      const ptrdiff_t sx = 1;
      const ptrdiff_t sy = static_cast<ptrdiff_t>(nx);
      for (size_t j = 0; j < by; j++, p += 4 * sx * ptrdiff_t(nx - bx))
        for (size_t i = 0; i < bx; i++, p += 4)
          cache.put_block(block_index++, p, sx, sy);
    }
    else {
      // zero-initialize array
      const value_type block[4 * 4] = {};
      while (block_index < bx * by)
        cache.put_block(block_index++, block, 1, 4);
    }
  }

  // (i, j) accessors
  const_reference operator()(size_t i, size_t j) const { return const_reference(const_cast<container_type*>(this), i, j); }
  reference operator()(size_t i, size_t j) { return reference(this, i, j); }

  // flat index accessors
  const_reference operator[](size_t index) const
  {
    size_t i, j;
    ij(i, j, index);
    return const_reference(const_cast<container_type*>(this), i, j);
  }
  reference operator[](size_t index)
  {
    size_t i, j;
    ij(i, j, index);
    return reference(this, i, j);
  }

  // random access iterators
  const_iterator cbegin() const { return const_iterator(this, 0, 0); }
  const_iterator cend() const { return const_iterator(this, 0, ny); }
  const_iterator begin() const { return cbegin(); }
  const_iterator end() const { return cend(); }
  iterator begin() { return iterator(this, 0, 0); }
  iterator end() { return iterator(this, 0, ny); }

protected:
  friend class zfp::internal::dim2::const_handle<array2>;
  friend class zfp::internal::dim2::const_reference<array2>;
  friend class zfp::internal::dim2::const_pointer<array2>;
  friend class zfp::internal::dim2::const_iterator<array2>;
  friend class zfp::internal::dim2::const_view<array2>;
  friend class zfp::internal::dim2::private_const_view<array2>;
  friend class zfp::internal::dim2::reference<array2>;
  friend class zfp::internal::dim2::pointer<array2>;
  friend class zfp::internal::dim2::iterator<array2>;
  friend class zfp::internal::dim2::view<array2>;
  friend class zfp::internal::dim2::flat_view<array2>;
  friend class zfp::internal::dim2::nested_view1<array2>;
  friend class zfp::internal::dim2::nested_view2<array2>;
  friend class zfp::internal::dim2::private_view<array2>;

  // perform a deep copy
  void deep_copy(const array2& a)
  {
    // copy base class members
    array::deep_copy(a);
    // copy persistent storage
    store.deep_copy(a.store);
    // copy cached data
    cache.deep_copy(a.cache);
  }

  // global index bounds
  size_t min_x() const { return 0; }
  size_t max_x() const { return nx; }
  size_t min_y() const { return 0; }
  size_t max_y() const { return ny; }

  // inspector
  value_type get(size_t i, size_t j) const { return cache.get(i, j); }

  // mutators (called from proxy reference)
  void set(size_t i, size_t j, value_type val) { cache.set(i, j, val); }
  void add(size_t i, size_t j, value_type val) { cache.ref(i, j) += val; }
  void sub(size_t i, size_t j, value_type val) { cache.ref(i, j) -= val; }
  void mul(size_t i, size_t j, value_type val) { cache.ref(i, j) *= val; }
  void div(size_t i, size_t j, value_type val) { cache.ref(i, j) /= val; }

  // convert flat index to (i, j)
  void ij(size_t& i, size_t& j, size_t index) const
  {
    i = index % nx; index /= nx;
    j = index;
  }

  store_type store; // persistent storage of compressed blocks
  cache_type cache; // cache of decompressed blocks
};

typedef array2<float> array2f;
typedef array2<double> array2d;

}

#endif
