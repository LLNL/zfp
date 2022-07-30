#ifndef ZFP_CONSTARRAY4_HPP
#define ZFP_CONSTARRAY4_HPP

#include <cstddef>
#include <cstring>
#include <iterator>
#include "zfp/array.hpp"
#include "zfp/index.hpp"
#include "zfp/codec/zfpcodec.hpp"
#include "zfp/internal/array/cache4.hpp"
#include "zfp/internal/array/handle4.hpp"
#include "zfp/internal/array/iterator4.hpp"
#include "zfp/internal/array/pointer4.hpp"
#include "zfp/internal/array/reference4.hpp"
#include "zfp/internal/array/store4.hpp"
#include "zfp/internal/array/view4.hpp"

namespace zfp {

// compressed 4D array of scalars
template <
  typename Scalar,
  class Codec = zfp::codec::zfp4<Scalar>,
  class Index = zfp::index::hybrid4
>
class const_array4 : public array {
public:
  // types utilized by nested classes
  typedef const_array4 container_type;
  typedef Scalar value_type;
  typedef Codec codec_type;
  typedef Index index_type;
  typedef zfp::internal::BlockStore4<value_type, codec_type, index_type> store_type;
  typedef zfp::internal::BlockCache4<value_type, store_type> cache_type;
  typedef typename Codec::header header;

  // accessor classes
  typedef zfp::internal::dim4::const_reference<const_array4> const_reference;
  typedef zfp::internal::dim4::const_pointer<const_array4> const_pointer;
  typedef zfp::internal::dim4::const_iterator<const_array4> const_iterator;
  typedef zfp::internal::dim4::const_view<const_array4> const_view;
  typedef zfp::internal::dim4::private_const_view<const_array4> private_const_view;

  // default constructor
  const_array4() :
    array(4, Codec::type),
    cache(store)
  {}

  // constructor of nx * ny * nz * nw array using given configuration, at least
  // cache_size bytes of cache, and optionally initialized from flat array p
  const_array4(size_t nx, size_t ny, size_t nz, size_t nw, const zfp_config& config, const value_type* p = 0, size_t cache_size = 0) :
    array(4, Codec::type),
    store(nx, ny, nz, nw, config),
    cache(store, cache_size)
  {
    this->nx = nx;
    this->ny = ny;
    this->nz = nz;
    this->nw = nw;
    set(p);
  }

  // copy constructor--performs a deep copy
  const_array4(const const_array4& a) :
    cache(store)
  {
    deep_copy(a);
  }

  // virtual destructor
  virtual ~const_array4() {}

  // assignment operator--performs a deep copy
  const_array4& operator=(const const_array4& a)
  {
    if (this != &a)
      deep_copy(a);
    return *this;
  }

  // total number of elements in array
  size_t size() const { return nx * ny * nz * nw; }

  // array dimensions
  size_t size_x() const { return nx; }
  size_t size_y() const { return ny; }
  size_t size_z() const { return nz; }
  size_t size_w() const { return nw; }

  // resize the array (all previously stored data will be lost)
  void resize(size_t nx, size_t ny, size_t nz, size_t nw, bool clear = true)
  {
    cache.clear();
    this->nx = nx;
    this->ny = ny;
    this->nz = nz;
    this->nw = nw;
    store.resize(nx, ny, nz, nw, clear);
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
    const size_t by = store.block_size_y();
    const size_t bz = store.block_size_z();
    const size_t bw = store.block_size_w();
    const ptrdiff_t sx = 1;
    const ptrdiff_t sy = static_cast<ptrdiff_t>(nx);
    const ptrdiff_t sz = static_cast<ptrdiff_t>(nx * ny);
    const ptrdiff_t sw = static_cast<ptrdiff_t>(nx * ny * nz);
    size_t block_index = 0;
    for (size_t l = 0; l < bw; l++, p += 4 * sz * (nz - bz))
      for (size_t k = 0; k < bz; k++, p += 4 * sy * (ny - by))
        for (size_t j = 0; j < by; j++, p += 4 * sx * (nx - bx))
          for (size_t i = 0; i < bx; i++, p += 4)
            cache.get_block(block_index++, p, sx, sy, sz, sw);
  }

  // initialize array by copying and compressing data stored at p
  void set(const value_type* p, bool compact = true)
  {
    cache.clear();
    store.clear();
    const size_t bx = store.block_size_x();
    const size_t by = store.block_size_y();
    const size_t bz = store.block_size_z();
    const size_t bw = store.block_size_w();
    size_t block_index = 0;
    if (p) {
      // compress data stored at p
      const ptrdiff_t sx = 1;
      const ptrdiff_t sy = static_cast<ptrdiff_t>(nx);
      const ptrdiff_t sz = static_cast<ptrdiff_t>(nx * ny);
      const ptrdiff_t sw = static_cast<ptrdiff_t>(nx * ny * nz);
      for (size_t l = 0; l < bw; l++, p += 4 * sz * (nz - bz))
        for (size_t k = 0; k < bz; k++, p += 4 * sy * (ny - by))
          for (size_t j = 0; j < by; j++, p += 4 * sx * (nx - bx))
            for (size_t i = 0; i < bx; i++, p += 4)
              store.encode(block_index++, p, sx, sy, sz, sw);
    }
    else {
      // zero-initialize array
      const value_type block[4 * 4 * 4 * 4] = {};
      while (block_index < bx * by * bz * bw)
        store.encode(block_index++, block);
    }
    store.flush();
    if (compact)
      store.compact();
  }

  // (i, j, k, l) accessor
  const_reference operator()(size_t i, size_t j, size_t k, size_t l) const { return const_reference(const_cast<container_type*>(this), i, j, k, l); }

  // flat index accessor
  const_reference operator[](size_t index) const
  {
    size_t i, j, k, l;
    ijkl(i, j, k, l, index);
    return const_reference(const_cast<container_type*>(this), i, j, k, l);
  }

  // random access iterators
  const_iterator cbegin() const { return const_iterator(this, 0, 0, 0, 0); }
  const_iterator cend() const { return const_iterator(this, 0, 0, 0, nw); }
  const_iterator begin() const { return cbegin(); }
  const_iterator end() const { return cend(); }

protected:
  friend class zfp::internal::dim4::const_handle<const_array4>;
  friend class zfp::internal::dim4::const_reference<const_array4>;
  friend class zfp::internal::dim4::const_pointer<const_array4>;
  friend class zfp::internal::dim4::const_iterator<const_array4>;
  friend class zfp::internal::dim4::const_view<const_array4>;
  friend class zfp::internal::dim4::private_const_view<const_array4>;

  // perform a deep copy
  void deep_copy(const const_array4& a)
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
  size_t min_z() const { return 0; }
  size_t max_z() const { return nz; }
  size_t min_w() const { return 0; }
  size_t max_w() const { return nw; }

  // inspector
  value_type get(size_t i, size_t j, size_t k, size_t l) const { return cache.get(i, j, k, l); }

  // convert flat index to (i, j, k, l)
  void ijkl(size_t& i, size_t& j, size_t& k, size_t& l, size_t index) const
  {
    i = index % nx; index /= nx;
    j = index % ny; index /= ny;
    k = index % nz; index /= nz;
    l = index;
  }

  store_type store; // persistent storage of compressed blocks
  cache_type cache; // cache of decompressed blocks
};

typedef const_array4<float> const_array4f;
typedef const_array4<double> const_array4d;

}

#endif
