#ifndef ARRAY2D_H
#define ARRAY2D_H

#include <vector>

typedef unsigned int uint;

// uncompressed 2D double-precision array (for comparison)
class Array2d {
public:
  Array2d(uint nx, uint ny, uint precision) : nx(nx), ny(ny), data(nx * ny, 0.0) {}
  size_t size() const { return data.size(); }
  double& operator()(uint x, uint y) { return data[x + nx * y]; }
  const double& operator()(uint x, uint y) const { return data[x + nx * y]; }
  double& operator[](uint i) { return data[i]; }
  const double& operator[](uint i) const { return data[i]; }
protected:
  uint nx;
  uint ny;
  std::vector<double> data;
};

#endif
