#ifndef ARRAY2D_H
#define ARRAY2D_H

#include <climits>
#include <vector>

#define unused_(x) ((void)(x))

typedef unsigned int uint;

// uncompressed 2D double-precision array (for comparison)
namespace raw {
class array2d {
public:
  array2d() : nx(0), ny(0) {}
  array2d(uint nx, uint ny, double rate = 0.0, const double* p = 0, size_t csize = 0) : nx(nx), ny(ny), data(nx * ny, 0.0)
  {
    unused_(rate);
    unused_(p);
    unused_(csize);
  }
  void resize(uint nx, uint ny) { this->nx = nx; this->ny = ny; data.resize(nx * ny, 0.0); }
  size_t size() const { return data.size(); }
  size_t size_x() const { return nx; }
  size_t size_y() const { return ny; }
  double rate() const { return CHAR_BIT * sizeof(double); }
  size_t cache_size() const { return 0; }
  double& operator()(uint x, uint y) { return data[x + nx * y]; }
  const double& operator()(uint x, uint y) const { return data[x + nx * y]; }
  double& operator[](uint i) { return data[i]; }
  const double& operator[](uint i) const { return data[i]; }
  class iterator {
  public:
    double& operator*() const { return array->operator[](index); }
    iterator& operator++() { index++; return *this; }
    iterator operator++(int) { iterator p = *this; index++; return p; }
    bool operator==(const iterator& it) const { return array == it.array && index == it.index; }
    bool operator!=(const iterator& it) const { return !operator==(it); }
    uint i() const { return index % array->nx; }
    uint j() const { return index / array->nx; }
  protected:
    friend class array2d;
    iterator(array2d* array, uint index) : array(array), index(index) {}
    array2d* array;
    uint index;
  };
  iterator begin() { return iterator(this, 0); }
  iterator end() { return iterator(this, nx * ny); }
protected:
  uint nx;
  uint ny;
  std::vector<double> data;
};
}

#undef unused_

#endif
