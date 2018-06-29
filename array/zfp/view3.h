// 3D array views; these classes are nested within zfp::array3

// abstract view of 3D array (base class)
class preview {
public:
  // rate in bits per value
  double rate() const { return array->rate(); }

  // dimensions of (sub)array
  size_t size() const { return size_t(nx) * size_t(ny) * size_t(nz); }

protected:
  // construction and assignment--perform shallow copy of (sub)array
  explicit preview(array3* array) : array(array), x(0), y(0), z(0), nx(array->nx), ny(array->ny), nz(array->nz) {}
  explicit preview(array3* array, uint x, uint y, uint z, uint nx, uint ny, uint nz) : array(array), x(x), y(y), z(z), nx(nx), ny(ny), nz(nz) {}
  preview& operator=(array3* a)
  {
    array = a;
    x = y = z = 0;
    nx = a->nx;
    ny = a->ny;
    nz = a->nz;
    return *this;
  }

  array3* array;   // underlying container
  uint x, y, z;    // offset into array
  uint nx, ny, nz; // dimensions of subarray
};

// generic read-only view into a rectangular subset of a 3D array
class const_view : public preview {
protected:
  using preview::array;
  using preview::x;
  using preview::y;
  using preview::z;
  using preview::nx;
  using preview::ny;
  using preview::nz;
public:
  // construction--perform shallow copy of (sub)array
  const_view(array3* array) : preview(array) {}
  const_view(array3* array, uint x, uint y, uint z, uint nx, uint ny, uint nz) : preview(array, x, y, z, nx, ny, nz) {}

  // dimensions of (sub)array
  uint size_x() const { return nx; }
  uint size_y() const { return ny; }
  uint size_z() const { return nz; }

  // (i, j, k) accessor
  Scalar operator()(uint i, uint j, uint k) const { return array->get(x + i, y + j, z + k); }
};

// generic read-write view into a rectangular subset of a 3D array (base class)
class view : public const_view {
protected:
  using preview::array;
  using preview::x;
  using preview::y;
  using preview::z;
  using preview::nx;
  using preview::ny;
  using preview::nz;
public:
  // construction--perform shallow copy of (sub)array
  view(array3* array) : const_view(array) {}
  view(array3* array, uint x, uint y, uint z, uint nx, uint ny, uint nz) : const_view(array, x, y, z, nx, ny, nz) {}

  // dimensions of (sub)array
  uint size_x() const { return nx; }
  uint size_y() const { return ny; }
  uint size_z() const { return nz; }

  // (i, j, k) mutator
  reference operator()(uint i, uint j, uint k) { return reference(array, x + i, y + j, z + k); }
};

// flat view of 3D array (operator[] returns scalar)
class flat_view : public view {
protected:
  using preview::array;
  using preview::x;
  using preview::y;
  using preview::z;
  using preview::nx;
  using preview::ny;
  using preview::nz;
public:
  // construction--perform shallow copy of (sub)array
  flat_view(array3* array) : view(array) {}
  flat_view(array3* array, uint x, uint y, uint z, uint nx, uint ny, uint nz) : view(array, x, y, z, nx, ny, nz) {}

  // convert (i, j, k) index to flat index
  uint index(uint i, uint j, uint k) const { return i + nx * (j + ny * k); }

  // convert flat index to (i, j, k) index
  void ijk(uint& i, uint& j, uint& k, uint index) const
  {
    i = index % nx; index /= nx;
    j = index % ny; index /= ny;
    k = index;
  }

  // flat index accessors
  Scalar operator[](uint index) const
  {
    uint i, j, k;
    ijk(i, j, k, index);
    return array->get(x + i, y + j, z + k);
  }
  reference operator[](uint index)
  {
    uint i, j, k;
    ijk(i, j, k, index);
    return reference(array, x + i, y + j, z + k);
  }
};

// forward declaration of friends
class nested_view1;
class nested_view2;
class nested_view3;

// nested view into a 1D rectangular subset of a 3D array
class nested_view1 : public preview {
protected:
  using preview::array;
  using preview::x;
  using preview::y;
  using preview::z;
  using preview::nx;
  using preview::ny;
  using preview::nz;
public:
  // dimensions of (sub)array
  uint size_x() const { return nx; }

  // [i] accessor and mutator
  Scalar operator[](uint index) const { return array->get(x + index, y, z); }
  reference operator[](uint index) { return reference(array, x + index, y, z); }

  // (i) accessor and mutator
  Scalar operator()(uint i) const { return array->get(x + i, y, z); }
  reference operator()(uint i) { return reference(array, x + i, y, z); }

protected:
  // construction--perform shallow copy of (sub)array
  friend class array3::nested_view2;
  explicit nested_view1(array3* array) : preview(array) {}
  explicit nested_view1(array3* array, uint x, uint y, uint z, uint nx, uint ny, uint nz) : preview(array, x, y, z, nx, ny, nz) {}
};

// nested view into a 2D rectangular subset of a 3D array
class nested_view2 : public preview {
protected:
  using preview::array;
  using preview::x;
  using preview::y;
  using preview::z;
  using preview::nx;
  using preview::ny;
  using preview::nz;
public:
  // dimensions of (sub)array
  uint size_x() const { return nx; }
  uint size_y() const { return ny; }

  // 1D view
  nested_view1 operator[](uint index) const { return nested_view1(array, x, y + index, z, nx, 1, 1); }

  // (i, j) accessor and mutator
  Scalar operator()(uint i, uint j) const { return array->get(x + i, y + j, z); }
  reference operator()(uint i, uint j) { return reference(array, x + i, y + j, z); }

protected:
  // construction--perform shallow copy of (sub)array
  friend class array3::nested_view3;
  explicit nested_view2(array3* array) : preview(array) {}
  explicit nested_view2(array3* array, uint x, uint y, uint z, uint nx, uint ny, uint nz) : preview(array, x, y, z, nx, ny, nz) {}
};

// nested view into a 3D rectangular subset of a 3D array
class nested_view3 : public preview {
protected:
  using preview::array;
  using preview::x;
  using preview::y;
  using preview::z;
  using preview::nx;
  using preview::ny;
  using preview::nz;
public:
  // construction--perform shallow copy of (sub)array
  nested_view3(array3* array) : preview(array) {}
  nested_view3(array3* array, uint x, uint y, uint z, uint nx, uint ny, uint nz) : preview(array, x, y, z, nx, ny, nz) {}

  // dimensions of (sub)array
  uint size_x() const { return nx; }
  uint size_y() const { return ny; }
  uint size_z() const { return nz; }

  // 2D view
  nested_view2 operator[](uint index) const { return nested_view2(array, x, y, z + index, nx, ny, 1); }

  // (i, j, k) accessor and mutator
  Scalar operator()(uint i, uint j, uint k) const { return array->get(x + i, y + j, z + k); }
  reference operator()(uint i, uint j, uint k) { return reference(array, x + i, y + j, z + k); }
};

typedef nested_view3 nested_view;
