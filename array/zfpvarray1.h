#ifndef ZFP_VARRAY1_H
#define ZFP_VARRAY1_H

#include <cstddef>
#include <cstring>
#include <iterator>
#include "zfpvarray.h"
#include "zfpcodec.h"
#include "zfp/cache1.h"
#include "zfp/tilestore1.h"
#include "zfp/handle1.h"
#include "zfp/reference1.h"
#include "zfp/pointer1.h"
#include "zfp/iterator1.h"
#include "zfp/view1.h"

namespace zfp {

// variable-rate compressed 1D array of scalars
template <
  typename Scalar,
  class Codec = zfp::codec::zfp1<Scalar>
>
class var_array1 : public var_array {
public:
  // types utilized by nested classes
  typedef var_array1 container_type;
  typedef Scalar value_type;
  typedef Codec codec_type;
  typedef zfp::internal::TileStore1<value_type, codec_type> store_type;
  typedef BlockCache1<value_type, store_type, true> cache_type;
  typedef typename Codec::header header;

  // accessor classes
  typedef zfp::internal::dim1::const_reference<var_array1> const_reference;
  typedef zfp::internal::dim1::const_pointer<var_array1> const_pointer;
  typedef zfp::internal::dim1::const_iterator<var_array1> const_iterator;
  typedef zfp::internal::dim1::const_view<var_array1> const_view;
  typedef zfp::internal::dim1::private_const_view<var_array1> private_const_view;
  typedef zfp::internal::dim1::reference<var_array1> reference;
  typedef zfp::internal::dim1::pointer<var_array1> pointer;
  typedef zfp::internal::dim1::iterator<var_array1> iterator;
  typedef zfp::internal::dim1::view<var_array1> view;
  typedef zfp::internal::dim1::private_view<var_array1> private_view;

  // default constructor
  var_array1() :
    var_array(1, Codec::type),
    cache(store)
  {}

  // constructor of nx-element array using given configuration, at least
  // cache_size bytes of cache, and optionally initialized from flat array p
  var_array1(size_t nx, const zfp_config& config, const value_type* p = 0, size_t cache_size = 0) :
    var_array(1, Codec::type),
    store(nx, config),
    cache(store, cache_size)
  {
    this->nx = nx;
    if (p)
      set(p);
  }

  // copy constructor--performs a deep copy
  var_array1(const var_array1& a) :
    cache(store)
  {
    deep_copy(a);
  }

  // virtual destructor
  virtual ~var_array1() {}

  // assignment operator--performs a deep copy
  var_array1& operator=(const var_array1& a)
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
  void resize(size_t nx)
  {
    cache.clear();
    this->nx = nx;
    store.resize(nx);
  }

  // rate in compressed bits per value
  double rate() const { return store.rate(); }

  // precision in uncompressed bits per value
  uint precision() const { return store.precision(); }

  // accuracy as absolute error tolerance
  double accuracy() const { return store.accuracy(); }

  // set rate in compressed bits per value
  double set_rate(double rate)
  {
    cache.clear();
    return store.set_rate(rate);
  }

  // set precision in uncompressed bits per value
  uint set_precision(uint precision)
  {
    cache.clear();
    return store.set_precision(precision);
  }

  // set accuracy as absolute error tolerance
  double set_accuracy(double tolerance)
  {
    cache.clear();
    return store.set_accuracy(tolerance);
  }

  virtual size_t size_bytes(uint mask = ZFP_DATA_ALL) const
  {
    size_t size = 0;
    size += store.size_bytes(mask);
    size += cache.size_bytes(mask);
    if (mask & ZFP_DATA_META)
      size += sizeof(*this);
    return size;
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
    const ptrdiff_t sx = 1;
    size_t block_index = 0;
    for (size_t i = 0; i < bx; i++, p += 4)
      cache.put_block(block_index++, p, sx);
  }

  // accessors
  const_reference operator()(size_t i) const { return const_reference(const_cast<container_type*>(this), i); }
  reference operator()(size_t i) { return reference(this, i); }

  // flat index accessors
  const_reference operator[](size_t index) const { return const_reference(const_cast<container_type*>(this), index); }
  reference operator[](size_t index) { return reference(this, index); }

  // random access iterators
  const_iterator cbegin() const { return const_iterator(this, 0, 0); }
  const_iterator cend() const { return const_iterator(this, 0, ny); }
  const_iterator begin() const { return cbegin(); }
  const_iterator end() const { return cend(); }
  iterator begin() { return iterator(this, 0, 0); }
  iterator end() { return iterator(this, 0, ny); }

protected:
  friend class zfp::internal::dim1::const_handle<var_array1>;
  friend class zfp::internal::dim1::const_reference<var_array1>;
  friend class zfp::internal::dim1::const_pointer<var_array1>;
  friend class zfp::internal::dim1::const_iterator<var_array1>;
  friend class zfp::internal::dim1::const_view<var_array1>;
  friend class zfp::internal::dim1::private_const_view<var_array1>;
  friend class zfp::internal::dim1::reference<var_array1>;
  friend class zfp::internal::dim1::pointer<var_array1>;
  friend class zfp::internal::dim1::iterator<var_array1>;
  friend class zfp::internal::dim1::view<var_array1>;
  friend class zfp::internal::dim1::private_view<var_array1>;

  // perform a deep copy
  void deep_copy(const var_array1& a)
  {
    // copy base class members
    var_array::deep_copy(a);
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

typedef var_array1<float> var_array1f;
typedef var_array1<double> var_array1d;

}

#endif
