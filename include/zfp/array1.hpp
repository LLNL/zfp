#ifndef ZFP_ARRAY1_HPP
#define ZFP_ARRAY1_HPP

#include <cstddef>
#include <cstring>
#include <iterator>
#include "zfp/array.hpp"
#include "zfp/index.hpp"
#include "zfp/codec/zfpcodec.hpp"
#include "zfp/internal/array/cache1.hpp"
#include "zfp/internal/array/handle1.hpp"
#include "zfp/internal/array/iterator1.hpp"
#include "zfp/internal/array/pointer1.hpp"
#include "zfp/internal/array/reference1.hpp"
#include "zfp/internal/array/store1.hpp"
#include "zfp/internal/array/view1.hpp"

namespace zfp {

// compressed 2D array of scalars
template <
  typename Scalar,
  class Codec = zfp::codec::zfp1<Scalar>,
  class Index = zfp::index::implicit
>
class array1 : public array {
public:
  // types utilized by nested classes
  typedef array1 container_type;
  typedef Scalar value_type;
  typedef Codec codec_type;
  typedef Index index_type;
  typedef zfp::internal::BlockStore1<value_type, codec_type, index_type> store_type;
  typedef zfp::internal::BlockCache1<value_type, store_type> cache_type;
  typedef typename Codec::header header;

  // accessor classes
  typedef zfp::internal::dim1::const_reference<array1> const_reference;
  typedef zfp::internal::dim1::const_pointer<array1> const_pointer;
  typedef zfp::internal::dim1::const_iterator<array1> const_iterator;
  typedef zfp::internal::dim1::const_view<array1> const_view;
  typedef zfp::internal::dim1::private_const_view<array1> private_const_view;
  typedef zfp::internal::dim1::reference<array1> reference;
  typedef zfp::internal::dim1::pointer<array1> pointer;
  typedef zfp::internal::dim1::iterator<array1> iterator;
  typedef zfp::internal::dim1::view<array1> view;
  typedef zfp::internal::dim1::private_view<array1> private_view;

  // default constructor
  array1() :
    array(1, Codec::type),
    cache(store)
  {}

  // constructor of nx-element array using rate bits per value, at least
  // cache_size bytes of cache, and optionally initialized from flat array p
  array1(size_t nx, double rate, const value_type* p = 0, size_t cache_size = 0) :
    array(1, Codec::type),
    store(nx, zfp_config_rate(rate, true)),
    cache(store, cache_size)
  {
    this->nx = nx;
    if (p)
      set(p);
  }

  // constructor, from previously-serialized compressed array
  array1(const zfp::array::header& header, const void* buffer = 0, size_t buffer_size_bytes = 0) :
    array(1, Codec::type, header),
    store(header.size_x(), zfp_config_rate(header.rate(), true)),
    cache(store)
  {
    if (buffer) {
      if (buffer_size_bytes && buffer_size_bytes < store.compressed_size())
        throw zfp::exception("buffer size is smaller than required");
      std::memcpy(store.compressed_data(), buffer, store.compressed_size());
    }
  }

  // copy constructor--performs a deep copy
  array1(const array1& a) :
    array(),
    cache(store)
  {
    deep_copy(a);
  }

  // construction from view--perform deep copy of (sub)array
  template <class View>
  array1(const View& v) :
    array(1, Codec::type),
    store(v.size_x(), zfp_config_rate(v.rate(), true)),
    cache(store)
  {
    this->nx = v.size_x();
    // initialize array in its preferred order
    for (iterator it = begin(); it != end(); ++it)
      *it = v(it.i());
  }

  // virtual destructor
  virtual ~array1() {}

  // assignment operator--performs a deep copy
  array1& operator=(const array1& a)
  {
    if (this != &a)
      deep_copy(a);
    return *this;
  }

  // total number of elements in array
  size_t size() const { return nx; }

  // array dimensions
  size_t size_x() const { return nx; }

  // resize the array (all previously stored data will be lost)
  void resize(size_t nx, bool clear = true)
  {
    cache.clear();
    this->nx = nx;
    store.resize(nx, clear);
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
    const ptrdiff_t sx = 1;
    size_t block_index = 0;
    for (size_t i = 0; i < bx; i++, p += 4)
      cache.get_block(block_index++, p, sx);
  }

  // initialize array by copying and compressing data stored at p
  void set(const value_type* p)
  {
    const size_t bx = store.block_size_x();
    size_t block_index = 0;
    if (p) {
      // compress data stored at p
      const ptrdiff_t sx = 1;
      for (size_t i = 0; i < bx; i++, p += 4)
        cache.put_block(block_index++, p, sx);
    }
    else {
      // zero-initialize array
      const value_type block[4] = {};
      while (block_index < bx)
        cache.put_block(block_index++, block, 1);
    }
  }

  // accessors
  const_reference operator()(size_t i) const { return const_reference(const_cast<container_type*>(this), i); }
  reference operator()(size_t i) { return reference(this, i); }

  // flat index accessors
  const_reference operator[](size_t index) const { return const_reference(const_cast<container_type*>(this), index); }
  reference operator[](size_t index) { return reference(this, index); }

  // random access iterators
  const_iterator cbegin() const { return const_iterator(this, 0); }
  const_iterator cend() const { return const_iterator(this, nx); }
  const_iterator begin() const { return cbegin(); }
  const_iterator end() const { return cend(); }
  iterator begin() { return iterator(this, 0); }
  iterator end() { return iterator(this, nx); }

protected:
  friend class zfp::internal::dim1::const_handle<array1>;
  friend class zfp::internal::dim1::const_reference<array1>;
  friend class zfp::internal::dim1::const_pointer<array1>;
  friend class zfp::internal::dim1::const_iterator<array1>;
  friend class zfp::internal::dim1::const_view<array1>;
  friend class zfp::internal::dim1::private_const_view<array1>;
  friend class zfp::internal::dim1::reference<array1>;
  friend class zfp::internal::dim1::pointer<array1>;
  friend class zfp::internal::dim1::iterator<array1>;
  friend class zfp::internal::dim1::view<array1>;
  friend class zfp::internal::dim1::private_view<array1>;

  // perform a deep copy
  void deep_copy(const array1& a)
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

  // inspector
  value_type get(size_t i) const { return cache.get(i); }

  // mutators (called from proxy reference)
  void set(size_t i, value_type val) { cache.set(i, val); }
  void add(size_t i, value_type val) { cache.ref(i) += val; }
  void sub(size_t i, value_type val) { cache.ref(i) -= val; }
  void mul(size_t i, value_type val) { cache.ref(i) *= val; }
  void div(size_t i, value_type val) { cache.ref(i) /= val; }

  store_type store; // persistent storage of compressed blocks
  cache_type cache; // cache of decompressed blocks
};

typedef array1<float> array1f;
typedef array1<double> array1d;

}

#endif
