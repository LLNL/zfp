#ifndef ZFP_VARRAY2_H
#define ZFP_VARRAY2_H

#include <cstddef>
#include <cstring>
#include <iterator>
#include "zfpvarray.h"
#include "zfpcodec.h"
#include "zfp/cache2.h"
#include "zfp/tilestore2.h"
#include "zfp/handle2.h"
#include "zfp/reference2.h"
#include "zfp/pointer2.h"
#include "zfp/iterator2.h"
#include "zfp/view2.h"

namespace zfp {

// variable-rate compressed 2D array of scalars
template <
  typename Scalar,
  class Codec = zfp::codec::zfp2<Scalar>
>
class var_array2 : public var_array {
public:
  // types utilized by nested classes
  typedef var_array2 container_type;
  typedef Scalar value_type;
  typedef Codec codec_type;
  typedef zfp::internal::TileStore2<value_type, codec_type> store_type;
  typedef BlockCache2<value_type, store_type, true> cache_type;
  typedef typename Codec::header header;

  // accessor classes (TODO: views)
  typedef zfp::internal::dim2::const_reference<var_array2> const_reference;
  typedef zfp::internal::dim2::const_pointer<var_array2> const_pointer;
  typedef zfp::internal::dim2::const_iterator<var_array2> const_iterator;
  typedef zfp::internal::dim2::const_view<var_array2> const_view;
  typedef zfp::internal::dim2::private_const_view<var_array2> private_const_view;
  typedef zfp::internal::dim2::reference<var_array2> reference;
  typedef zfp::internal::dim2::pointer<var_array2> pointer;
  typedef zfp::internal::dim2::iterator<var_array2> iterator;
  typedef zfp::internal::dim2::view<var_array2> view;
  typedef zfp::internal::dim2::flat_view<var_array2> flat_view;
  typedef zfp::internal::dim2::nested_view1<var_array2> nested_view1;
  typedef zfp::internal::dim2::nested_view2<var_array2> nested_view2;
  typedef zfp::internal::dim2::nested_view2<var_array2> nested_view;
  typedef zfp::internal::dim2::private_view<var_array2> private_view;

  // default constructor
  var_array2() :
    var_array(2, Codec::type),
    cache(store)
  {}

  // constructor of nx * ny array using given configuration, at least
  // cache_size bytes of cache, and optionally initialized from flat array p
  var_array2(size_t nx, size_t ny, const zfp_config& config, const value_type* p = 0, size_t cache_size = 0) :
    var_array(2, Codec::type),
    store(nx, ny, config),
    cache(store, cache_size)
  {
    this->nx = nx;
    this->ny = ny;
    if (p)
      set(p);
  }

  // copy constructor--performs a deep copy
  var_array2(const var_array2& a) :
    cache(store)
  {
    deep_copy(a);
  }

  // virtual destructor
  virtual ~var_array2() {}

  // assignment operator--performs a deep copy
  var_array2& operator=(const var_array2& a)
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
  void resize(size_t nx, size_t ny)
  {
    cache.clear();
    this->nx = nx;
    this->ny = ny;
    store.resize(nx, ny);
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
    const ptrdiff_t sx = 1;
    const ptrdiff_t sy = static_cast<ptrdiff_t>(nx);
    size_t block_index = 0;
    for (size_t j = 0; j < by; j++, p += 4 * sx * ptrdiff_t(nx - bx))
      for (size_t i = 0; i < bx; i++, p += 4)
        cache.put_block(block_index++, p, sx, sy);
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
  friend class zfp::internal::dim2::const_handle<var_array2>;
  friend class zfp::internal::dim2::const_reference<var_array2>;
  friend class zfp::internal::dim2::const_pointer<var_array2>;
  friend class zfp::internal::dim2::const_iterator<var_array2>;
  friend class zfp::internal::dim2::const_view<var_array2>;
  friend class zfp::internal::dim2::private_const_view<var_array2>;
  friend class zfp::internal::dim2::reference<var_array2>;
  friend class zfp::internal::dim2::pointer<var_array2>;
  friend class zfp::internal::dim2::iterator<var_array2>;
  friend class zfp::internal::dim2::view<var_array2>;
  friend class zfp::internal::dim2::flat_view<var_array2>;
  friend class zfp::internal::dim2::nested_view1<var_array2>;
  friend class zfp::internal::dim2::nested_view2<var_array2>;
  friend class zfp::internal::dim2::private_view<var_array2>;

  // perform a deep copy
  void deep_copy(const var_array2& a)
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

typedef var_array2<float> var_array2f;
typedef var_array2<double> var_array2d;

}

#endif
