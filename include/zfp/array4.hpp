#ifndef ZFP_ARRAY4_HPP
#define ZFP_ARRAY4_HPP

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

// compressed 3D array of scalars
template <
  typename Scalar,
  class Codec = zfp::codec::zfp4<Scalar>,
  class Index = zfp::index::implicit
>
class array4 : public array {
public:
  // types utilized by nested classes
  typedef array4 container_type;
  typedef Scalar value_type;
  typedef Codec codec_type;
  typedef Index index_type;
  typedef zfp::internal::BlockStore4<value_type, codec_type, index_type> store_type;
  typedef zfp::internal::BlockCache4<value_type, store_type> cache_type;
  typedef typename Codec::header header;

  // accessor classes
  typedef zfp::internal::dim4::const_reference<array4> const_reference;
  typedef zfp::internal::dim4::const_pointer<array4> const_pointer;
  typedef zfp::internal::dim4::const_iterator<array4> const_iterator;
  typedef zfp::internal::dim4::const_view<array4> const_view;
  typedef zfp::internal::dim4::private_const_view<array4> private_const_view;
  typedef zfp::internal::dim4::reference<array4> reference;
  typedef zfp::internal::dim4::pointer<array4> pointer;
  typedef zfp::internal::dim4::iterator<array4> iterator;
  typedef zfp::internal::dim4::view<array4> view;
  typedef zfp::internal::dim4::flat_view<array4> flat_view;
  typedef zfp::internal::dim4::nested_view1<array4> nested_view1;
  typedef zfp::internal::dim4::nested_view2<array4> nested_view2;
  typedef zfp::internal::dim4::nested_view3<array4> nested_view3;
  typedef zfp::internal::dim4::nested_view4<array4> nested_view4;
  typedef zfp::internal::dim4::nested_view4<array4> nested_view;
  typedef zfp::internal::dim4::private_view<array4> private_view;

  // default constructor
  array4() :
    array(4, Codec::type),
    cache(store)
  {}

  // constructor of nx * ny * nz * nw array using rate bits per value, at least
  // cache_size bytes of cache, and optionally initialized from flat array p
  array4(size_t nx, size_t ny, size_t nz, size_t nw, double rate, const value_type* p = 0, size_t cache_size = 0) :
    array(4, Codec::type),
    store(nx, ny, nz, nw, zfp_config_rate(rate, true)),
    cache(store, cache_size)
  {
    this->nx = nx;
    this->ny = ny;
    this->nz = nz;
    this->nw = nw;
    if (p)
      set(p);
  }

  // constructor, from previously-serialized compressed array
  array4(const zfp::array::header& header, const void* buffer = 0, size_t buffer_size_bytes = 0) :
    array(4, Codec::type, header),
    store(header.size_x(), header.size_y(), header.size_z(), header.size_w(), zfp_config_rate(header.rate(), true)),
    cache(store)
  {
    if (buffer) {
      if (buffer_size_bytes && buffer_size_bytes < store.compressed_size())
        throw zfp::exception("buffer size is smaller than required");
      std::memcpy(store.compressed_data(), buffer, store.compressed_size());
    }
  }

  // copy constructor--performs a deep copy
  array4(const array4& a) :
    array(),
    cache(store)
  {
    deep_copy(a);
  }

  // construction from view--perform deep copy of (sub)array
  template <class View>
  array4(const View& v) :
    array(4, Codec::type),
    store(v.size_x(), v.size_y(), v.size_z(), v.size_w(), zfp_config_rate(v.rate(), true)),
    cache(store)
  {
    this->nx = v.size_x();
    this->ny = v.size_y();
    this->nz = v.size_z();
    this->nw = v.size_w();
    // initialize array in its preferred order
    for (iterator it = begin(); it != end(); ++it)
      *it = v(it.i(), it.j(), it.k(), it.l());
  }

  // virtual destructor
  virtual ~array4() {}

  // assignment operator--performs a deep copy
  array4& operator=(const array4& a)
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
    const size_t bz = store.block_size_z();
    const size_t bw = store.block_size_w();
    const ptrdiff_t sx = 1;
    const ptrdiff_t sy = static_cast<ptrdiff_t>(nx);
    const ptrdiff_t sz = static_cast<ptrdiff_t>(nx * ny);
    const ptrdiff_t sw = static_cast<ptrdiff_t>(nx * ny * nz);
    size_t block_index = 0;
    for (size_t l = 0; l < bw; l++, p += 4 * sz * ptrdiff_t(nz - bz))
      for (size_t k = 0; k < bz; k++, p += 4 * sy * ptrdiff_t(ny - by))
        for (size_t j = 0; j < by; j++, p += 4 * sx * ptrdiff_t(nx - bx))
          for (size_t i = 0; i < bx; i++, p += 4)
            cache.get_block(block_index++, p, sx, sy, sz, sw);
  }

  // initialize array by copying and compressing data stored at p
  void set(const value_type* p)
  {
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
      for (size_t l = 0; l < bw; l++, p += 4 * sz * ptrdiff_t(nz - bz))
        for (size_t k = 0; k < bz; k++, p += 4 * sy * ptrdiff_t(ny - by))
          for (size_t j = 0; j < by; j++, p += 4 * sx * ptrdiff_t(nx - bx))
            for (size_t i = 0; i < bx; i++, p += 4)
              cache.put_block(block_index++, p, sx, sy, sz, sw);
    }
    else {
      // zero-initialize array
      const value_type block[4 * 4 * 4 * 4] = {};
      while (block_index < bx * by * bz * bw)
        cache.put_block(block_index++, block, 1, 4, 16, 64);
    }
  }

  // (i, j, k) accessors
  const_reference operator()(size_t i, size_t j, size_t k, size_t l) const { return const_reference(const_cast<container_type*>(this), i, j, k, l); }
  reference operator()(size_t i, size_t j, size_t k, size_t l) { return reference(this, i, j, k, l); }

  // flat index accessors
  const_reference operator[](size_t index) const
  {
    size_t i, j, k, l;
    ijkl(i, j, k, l, index);
    return const_reference(const_cast<container_type*>(this), i, j, k, l);
  }
  reference operator[](size_t index)
  {
    size_t i, j, k, l;
    ijkl(i, j, k, l, index);
    return reference(this, i, j, k, l);
  }

  // random access iterators
  const_iterator cbegin() const { return const_iterator(this, 0, 0, 0, 0); }
  const_iterator cend() const { return const_iterator(this, 0, 0, 0, nw); }
  const_iterator begin() const { return cbegin(); }
  const_iterator end() const { return cend(); }
  iterator begin() { return iterator(this, 0, 0, 0, 0); }
  iterator end() { return iterator(this, 0, 0, 0, nw); }

protected:
  friend class zfp::internal::dim4::const_handle<array4>;
  friend class zfp::internal::dim4::const_reference<array4>;
  friend class zfp::internal::dim4::const_pointer<array4>;
  friend class zfp::internal::dim4::const_iterator<array4>;
  friend class zfp::internal::dim4::const_view<array4>;
  friend class zfp::internal::dim4::private_const_view<array4>;
  friend class zfp::internal::dim4::reference<array4>;
  friend class zfp::internal::dim4::pointer<array4>;
  friend class zfp::internal::dim4::iterator<array4>;
  friend class zfp::internal::dim4::view<array4>;
  friend class zfp::internal::dim4::flat_view<array4>;
  friend class zfp::internal::dim4::nested_view1<array4>;
  friend class zfp::internal::dim4::nested_view2<array4>;
  friend class zfp::internal::dim4::nested_view3<array4>;
  friend class zfp::internal::dim4::nested_view4<array4>;
  friend class zfp::internal::dim4::private_view<array4>;

  // perform a deep copy
  void deep_copy(const array4& a)
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

  // mutators (called from proxy reference)
  void set(size_t i, size_t j, size_t k, size_t l, value_type val) { cache.set(i, j, k, l, val); }
  void add(size_t i, size_t j, size_t k, size_t l, value_type val) { cache.ref(i, j, k, l) += val; }
  void sub(size_t i, size_t j, size_t k, size_t l, value_type val) { cache.ref(i, j, k, l) -= val; }
  void mul(size_t i, size_t j, size_t k, size_t l, value_type val) { cache.ref(i, j, k, l) *= val; }
  void div(size_t i, size_t j, size_t k, size_t l, value_type val) { cache.ref(i, j, k, l) /= val; }

  // convert flat index to (i, j, k)
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

typedef array4<float> array4f;
typedef array4<double> array4d;

}

#endif
