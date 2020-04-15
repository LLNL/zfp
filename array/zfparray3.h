#ifndef ZFP_ARRAY3_H
#define ZFP_ARRAY3_H

#include <cstddef>
#include <cstring>
#include <iterator>
#include "zfparray.h"
#include "zfp/cache3.h"
#include "zfp/store3.h"

namespace zfp {

// compressed 3D array of scalars
template < typename Scalar, class Codec = zfp::codec<Scalar, 3> >
class array3 : public array {
public:
  // types utilized by nested classes
  typedef array3 container_type;
  typedef Scalar value_type;
  typedef Codec codec_type;

  // forward declarations
  class const_reference;
  class const_pointer;
  class const_iterator;
  class reference;
  class pointer;
  class iterator;
  class view;

  #include "zfp/handle3.h"
  #include "zfp/reference3.h"
  #include "zfp/pointer3.h"
  #include "zfp/iterator3.h"
  #include "zfp/view3.h"

  // default constructor
  array3() :
    array(3, Codec::type),
    cache(store, 0)
  {}

  // constructor of nx * ny * nz array using rate bits per value, at least
  // cache_size bytes of cache, and optionally initialized from flat array p
  array3(uint nx, uint ny, uint nz, double rate, const Scalar* p = 0, size_t cache_size = 0) :
    array(3, Codec::type),
    store(nx, ny, nz, rate),
    cache(store, cache_size)
  {
    this->nx = nx;
    this->ny = ny;
    this->nz = nz;
    if (p)
      set(p);
  }

  // constructor, from previously-serialized compressed array
  array3(const zfp::array::header& h, const void* = 0, size_t buffer_size_bytes = 0) :
    array(3, Codec::type, h, buffer_size_bytes),
    cache(store)
  {
#if 0
    resize(nx, ny, nz, false);
    if (buffer)
      std::memcpy(data, buffer, bytes);
#else
    // must construct store and cache
    throw std::runtime_error("(de)serialization not supported");
#endif
  }

  // copy constructor--performs a deep copy
  array3(const array3& a) :
    cache(store)
  {
    deep_copy(a);
  }

  // construction from view--perform deep copy of (sub)array
  template <class View>
  array3(const View& v) :
    array(3, Codec::type),
    store(v.size_x(), v.size_y(), v.size_z(), v.rate()),
    cache(store, 0)
  {
    // initialize array in its preferred order
    for (iterator it = begin(); it != end(); ++it)
      *it = v(it.i(), it.j(), it.k());
  }

  // virtual destructor
  virtual ~array3() {}

  // assignment operator--performs a deep copy
  array3& operator=(const array3& a)
  {
    if (this != &a)
      deep_copy(a);
    return *this;
  }

  // total number of elements in array
  size_t size() const { return size_t(nx) * size_t(ny) * size_t(nz); }

  // array dimensions
  uint size_x() const { return nx; }
  uint size_y() const { return ny; }
  uint size_z() const { return nz; }

  // resize the array (all previously stored data will be lost)
  void resize(uint nx, uint ny, uint nz, bool clear = true)
  {
    this->nx = nx;
    this->ny = ny;
    this->nz = nz;
    store.resize(nx, ny, nz, clear);
    cache.clear();
  }

  // rate in bits per value
  double rate() const { return store.rate(); }

  // set rate in bits per value
  double set_rate(double rate)
  {
    rate = store.set_rate(rate);
    cache.clear();
    return rate;
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
  void get(Scalar* p) const
  {
    const uint bx = store.block_size_x();
    const uint by = store.block_size_y();
    const uint bz = store.block_size_z();
    uint block_index = 0;
    for (uint k = 0; k < bz; k++, p += 4 * nx * (ny - by))
      for (uint j = 0; j < by; j++, p += 4 * (nx - bx))
        for (uint i = 0; i < bx; i++, p += 4)
          cache.get_block(block_index++, p, 1, nx, nx * ny);
  }

  // initialize array by copying and compressing data stored at p
  void set(const Scalar* p)
  {
    const uint bx = store.block_size_x();
    const uint by = store.block_size_y();
    const uint bz = store.block_size_z();
    uint block_index = 0;
    for (uint k = 0; k < bz; k++, p += 4 * nx * (ny - by))
      for (uint j = 0; j < by; j++, p += 4 * (nx - bx))
        for (uint i = 0; i < bx; i++, p += 4)
          store.encode(block_index++, p, 1, nx, nx * ny);
    cache.clear();
  }

  // (i, j, k) accessors
  Scalar operator()(uint i, uint j, uint k) const { return get(i, j, k); }
  reference operator()(uint i, uint j, uint k) { return reference(this, i, j, k); }

  // flat index accessors
  Scalar operator[](uint index) const
  {
    uint i, j, k;
    ijk(i, j, k, index);
    return get(i, j, k);
  }
  reference operator[](uint index)
  {
    uint i, j, k;
    ijk(i, j, k, index);
    return reference(this, i, j, k);
  }

  // sequential iterators
  const_iterator cbegin() const { return const_iterator(this, 0, 0, 0); }
  const_iterator cend() const { return const_iterator(this, 0, 0, nz); }
  const_iterator begin() const { return cbegin(); }
  const_iterator end() const { return cend(); }
  iterator begin() { return iterator(this, 0, 0, 0); }
  iterator end() { return iterator(this, 0, 0, nz); }

protected:
  // perform a deep copy
  void deep_copy(const array3& a)
  {
    // copy base class members
    array::deep_copy(a);
    // copy persistent storage
    store.deep_copy(a.store);
    // copy cached data
    cache.deep_copy(a.cache);
  }

  // inspector
  Scalar get(uint i, uint j, uint k) const { return cache.get(i, j, k); }

  // mutators (called from proxy reference)
  void set(uint i, uint j, uint k, Scalar val) { cache.set(i, j, k, val); }
  void add(uint i, uint j, uint k, Scalar val) { cache.ref(i, j, k) += val; }
  void sub(uint i, uint j, uint k, Scalar val) { cache.ref(i, j, k) -= val; }
  void mul(uint i, uint j, uint k, Scalar val) { cache.ref(i, j, k) *= val; }
  void div(uint i, uint j, uint k, Scalar val) { cache.ref(i, j, k) /= val; }

  // convert flat index to (i, j, k)
  void ijk(uint& i, uint& j, uint& k, uint index) const
  {
    i = index % nx; index /= nx;
    j = index % ny; index /= ny;
    k = index;
  }

  BlockStore3<Scalar, Codec> store; // persistent storage of compressed blocks
  BlockCache3<Scalar, Codec> cache; // cache of decompressed blocks
};

typedef array3<float> array3f;
typedef array3<double> array3d;

}

#endif
