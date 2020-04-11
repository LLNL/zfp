#ifndef ZFP_ARRAY1_H
#define ZFP_ARRAY1_H

#include <cstddef>
#include <cstring>
#include <iterator>
#include "zfparray.h"
#include "zfp/cache1.h"
#include "zfp/block1.h"

namespace zfp {

// compressed 2D array of scalars
template < typename Scalar, class Codec = zfp::codec<Scalar, 1> >
class array1 : public array {
public:
  // types utilized by nested classes
  typedef array1 container_type;
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

  #include "zfp/handle1.h"
  #include "zfp/reference1.h"
  #include "zfp/pointer1.h"
  #include "zfp/iterator1.h"
  #include "zfp/view1.h"

  // default constructor
  array1() :
    array(1, Codec::type),
    cache(storage, 0)
  {}

  // constructor of nx-element array using rate bits per value, at least
  // cache_size bytes of cache, and optionally initialized from flat array p
  array1(uint nx, double rate, const Scalar* p = 0, size_t cache_size = 0) :
    array(1, Codec::type),
    storage(nx, rate),
    cache(storage, cache_size)
  {
    this->nx = nx;
    if (p)
      set(p);
  }

  // constructor, from previously-serialized compressed array
  array1(const zfp::array::header& h, const void* = 0, size_t buffer_size_bytes = 0) :
    array(1, Codec::type, h, buffer_size_bytes),
    cache(storage)
  {
#if 0
    resize(nx, false);
    if (buffer)
      std::memcpy(data, buffer, bytes);
#else
    // must construct storage and cache
    throw std::runtime_error("(de)serialization not supported");
#endif
  }

  // copy constructor--performs a deep copy
  array1(const array1& a) :
    cache(storage)
  {
    deep_copy(a);
  }

  // construction from view--perform deep copy of (sub)array
  template <class View>
  array1(const View& v) :
    array(1, Codec::type),
    storage(v.size_x(), v.rate()),
    cache(storage, 0)
  {
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
  size_t size() const { return size_t(nx); }

  // array dimensions
  uint size_x() const { return nx; }

  // resize the array (all previously stored data will be lost)
  void resize(uint nx, bool clear = true)
  {
    this->nx = nx;
    storage.resize(nx, clear);
    cache.clear();
  }

  // rate in bits per value
  double rate() const { return storage.rate(); }

  // set rate in bits per value
  double set_rate(double rate)
  {
    rate = storage.set_rate(rate);
    cache.clear();
    return rate;
  }

  // number of bytes of compressed data
  size_t compressed_size() const { return storage.compressed_size(); }

  // pointer to compressed data for read or write access
  void* compressed_data() const
  {
    cache.flush();
    return storage.compressed_data();
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
    const uint bx = storage.block_size_x();
    uint block_index = 0;
    for (uint i = 0; i < bx; i++, p += 4)
      cache.get_block(block_index++, p, 1);
  }

  // initialize array by copying and compressing data stored at p
  void set(const Scalar* p)
  {
    const uint bx = storage.block_size_x();
    uint block_index = 0;
    for (uint i = 0; i < bx; i++, p += 4)
      storage.encode(block_index++, p, 1);
    cache.clear();
  }

  // (i, j) accessors
  Scalar operator()(uint i) const { return get(i); }
  reference operator()(uint i) { return reference(this, i); }

  // flat index accessors
  Scalar operator[](uint index) const { return get(index); }
  reference operator[](uint index) { return reference(this, index); }

  // random access iterators
  const_iterator cbegin() const { return const_iterator(this, 0); }
  const_iterator cend() const { return const_iterator(this, nx); }
  const_iterator begin() const { return cbegin(); }
  const_iterator end() const { return cend(); }
  iterator begin() { return iterator(this, 0); }
  iterator end() { return iterator(this, nx); }

protected:
  // perform a deep copy
  void deep_copy(const array1& a)
  {
    // copy base class members
    array::deep_copy(a);
    // copy persistent storage
    storage.deep_copy(a.storage);
    // copy cached data
    cache.deep_copy(a.cache);
  }

  // inspector
  Scalar get(uint i) const { return cache.get(i); }

  // mutators (called from proxy reference)
  void set(uint i, Scalar val) { cache.set(i, val); }
  void add(uint i, Scalar val) { cache.ref(i) += val; }
  void sub(uint i, Scalar val) { cache.ref(i) -= val; }
  void mul(uint i, Scalar val) { cache.ref(i) *= val; }
  void div(uint i, Scalar val) { cache.ref(i) /= val; }

  BlockStorage1<Scalar, Codec> storage; // persistent storage of compressed blocks
  BlockCache1<Scalar, Codec> cache;     // cache of decompressed blocks
};

typedef array1<float> array1f;
typedef array1<double> array1d;

}

#endif
