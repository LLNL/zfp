#ifndef ZFP_CONSTARRAY1_HPP
#define ZFP_CONSTARRAY1_HPP

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

// compressed 1D array of scalars
template <
  typename Scalar,
  class Codec = zfp::codec::zfp1<Scalar>,
  class Index = zfp::index::hybrid4
>
class const_array1 : public array {
public:
  // types utilized by nested classes
  typedef const_array1 container_type;
  typedef Scalar value_type;
  typedef Codec codec_type;
  typedef Index index_type;
  typedef zfp::internal::BlockStore1<value_type, codec_type, index_type> store_type;
  typedef zfp::internal::BlockCache1<value_type, store_type> cache_type;
  typedef typename Codec::header header;

  // accessor classes
  typedef zfp::internal::dim1::const_reference<const_array1> const_reference;
  typedef zfp::internal::dim1::const_pointer<const_array1> const_pointer;
  typedef zfp::internal::dim1::const_iterator<const_array1> const_iterator;
  typedef zfp::internal::dim1::const_view<const_array1> const_view;
  typedef zfp::internal::dim1::private_const_view<const_array1> private_const_view;

  // default constructor
  const_array1() :
    array(1, Codec::type),
    cache(store)
  {}

  // constructor of nx-element array using given configuration, at least
  // cache_size bytes of cache, and optionally initialized from flat array p
  const_array1(size_t nx, const zfp_config& config, const value_type* p = 0, size_t cache_size = 0) :
    array(1, Codec::type),
    store(nx, config),
    cache(store, cache_size)
  {
    this->nx = nx;
    set(p);
  }

  // copy constructor--performs a deep copy
  const_array1(const const_array1& a) :
    cache(store)
  {
    deep_copy(a);
  }

  // virtual destructor
  virtual ~const_array1() {}

  // assignment operator--performs a deep copy
  const_array1& operator=(const const_array1& a)
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

  // compression mode
  zfp_mode mode() const { return store.mode(); }

  // rate in compressed bits per value (fixed-rate mode only)
  double rate() const { return store.rate(); }

  // precision in uncompressed bits per value (fixed-precision mode only)
  uint precision() const { return store.precision(); }

  // accuracy as absolute error tolerance (fixed-accuracy mode only)
  double accuracy() const { return store.accuracy(); }

  // compression parameters (all compression modes)
  void params(uint* minbits, uint* maxbits, uint* maxprec, int* minexp) const { return store.params(minbits, maxbits, maxprec, minexp); }

  // set rate in compressed bits per value
  double set_rate(double rate)
  {
    cache.clear();
    return store.set_rate(rate, false);
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

  // enable reversible (lossless) mode
  void set_reversible()
  {
    cache.clear();
    store.set_reversible();
  }

  // set expert mode compression parameters
  bool set_params(uint minbits, uint maxbits, uint maxprec, int minexp)
  {
    cache.clear();
    return store.set_params(minbits, maxbits, maxprec, minexp);
  }

  // set compression mode and parameters
  void set_config(const zfp_config& config)
  {
    cache.clear();
    store.set_config(config);
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
  void set(const value_type* p, bool compact = true)
  {
    cache.clear();
    store.clear();
    const size_t bx = store.block_size_x();
    size_t block_index = 0;
    if (p) {
      // compress data stored at p
      const ptrdiff_t sx = 1;
      for (size_t i = 0; i < bx; i++, p += 4)
        store.encode(block_index++, p, sx);
    }
    else {
      // zero-initialize array
      const value_type block[4] = {};
      while (block_index < bx)
        store.encode(block_index++, block);
    }
    store.flush();
    if (compact)
      store.compact();
  }

  // accessor
  const_reference operator()(size_t i) const { return const_reference(const_cast<container_type*>(this), i); }

  // flat index accessor
  const_reference operator[](size_t index) const { return const_reference(const_cast<container_type*>(this), index); }

  // random access iterators
  const_iterator cbegin() const { return const_iterator(this, 0); }
  const_iterator cend() const { return const_iterator(this, nx); }
  const_iterator begin() const { return cbegin(); }
  const_iterator end() const { return cend(); }

protected:
  friend class zfp::internal::dim1::const_handle<const_array1>;
  friend class zfp::internal::dim1::const_reference<const_array1>;
  friend class zfp::internal::dim1::const_pointer<const_array1>;
  friend class zfp::internal::dim1::const_iterator<const_array1>;
  friend class zfp::internal::dim1::const_view<const_array1>;
  friend class zfp::internal::dim1::private_const_view<const_array1>;

  // perform a deep copy
  void deep_copy(const const_array1& a)
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

  store_type store; // persistent storage of compressed blocks
  cache_type cache; // cache of decompressed blocks
};

typedef const_array1<float> const_array1f;
typedef const_array1<double> const_array1d;

}

#endif
