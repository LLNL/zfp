#ifndef ARRAY2D_HPP
#define ARRAY2D_HPP

#include <climits>
#include <vector>

typedef unsigned int uint;

// uncompressed 2D double-precision array (for comparison)
namespace raw {
class array2d {
public:
  // constructors
  array2d() : nx(0), ny(0) {}
  array2d(size_t nx, size_t ny, double = 0.0, const double* = 0, size_t = 0) : nx(nx), ny(ny), data(nx * ny, 0.0) {}

  // array size
  size_t size() const { return data.size(); }
  size_t size_x() const { return nx; }
  size_t size_y() const { return ny; }
  void resize(size_t nx, size_t ny) { this->nx = nx; this->ny = ny; data.resize(nx * ny, 0.0); }

  // rate in bits/value
  double rate() const { return CHAR_BIT * sizeof(double); }

  // cache size in bytes
  size_t cache_size() const { return 0; }

  // byte size of data structures
  size_t size_bytes(uint mask = ZFP_DATA_ALL) const
  {
    size_t size = 0;
    if (mask & ZFP_DATA_META)
      size += sizeof(*this);
    if (mask & ZFP_DATA_PAYLOAD)
      size += data.size() * sizeof(double);
    return size;
  }

  // accessors
  double& operator()(size_t x, size_t y) { return data[x + nx * y]; }
  const double& operator()(size_t x, size_t y) const { return data[x + nx * y]; }
  double& operator[](size_t index) { return data[index]; }
  const double& operator[](size_t index) const { return data[index]; }

  // minimal-functionality forward iterator
  class iterator {
  public:
    double& operator*() const { return array->operator[](index); }
    iterator& operator++() { index++; return *this; }
    iterator operator++(int) { iterator p = *this; index++; return p; }
    bool operator==(const iterator& it) const { return array == it.array && index == it.index; }
    bool operator!=(const iterator& it) const { return !operator==(it); }
    size_t i() const { return index % array->nx; }
    size_t j() const { return index / array->nx; }
  protected:
    friend class array2d;
    iterator(array2d* array, size_t index) : array(array), index(index) {}
    array2d* array;
    size_t index;
  };

  iterator begin() { return iterator(this, 0); }
  iterator end() { return iterator(this, nx * ny); }

protected:
  size_t nx, ny;
  std::vector<double> data;
};
}

#endif
