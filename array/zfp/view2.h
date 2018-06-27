// 2D array views; these classes are nested within zfp::array2

// abstract view of 2D array (base class)
class preview {
public:
  // rate in bits per value
  double rate() const { return array->rate(); }

  // dimensions of (sub)array
  size_t size() const { return size_t(nx) * size_t(ny); }

protected:
  // construction and assignment--perform shallow copy of (sub)array
  explicit preview(array2* array) : array(array), x(0), y(0), nx(array->nx), ny(array->ny) {}
  explicit preview(array2* array, uint x, uint y, uint nx, uint ny) : array(array), x(x), y(y), nx(nx), ny(ny) {}
  preview& operator=(array2* a)
  {
    array = a;
    x = y = 0;
    nx = a->nx;
    ny = a->ny;
    return *this;
  }

  array2* array; // underlying container
  uint x, y;     // offset into array
  uint nx, ny;   // dimensions of subarray
};

// generic read-only view into a rectangular subset of a 2D array
class const_view : public preview {
  using preview::array;
  using preview::x;
  using preview::y;
  using preview::nx;
  using preview::ny;
public:
  // construction--perform shallow copy of (sub)array
  const_view(array2* array) : preview(array) {}
  const_view(array2* array, uint x, uint y, uint nx, uint ny) : preview(array, x, y, nx, ny) {}

  // dimensions of (sub)array
  uint size_x() const { return nx; }
  uint size_y() const { return ny; }

  // (i, j) accessor
  Scalar operator()(uint i, uint j) const { return array->get(x + i, y + j); }
};

// generic read-write view into a rectangular subset of a 2D array
class view : public const_view {
  using preview::array;
  using preview::x;
  using preview::y;
  using preview::nx;
  using preview::ny;
public:
  // construction--perform shallow copy of (sub)array
  view(array2* array) : const_view(array) {}
  view(array2* array, uint x, uint y, uint nx, uint ny) : const_view(array, x, y, nx, ny) {}

  // (i, j) mutator
  reference operator()(uint i, uint j) { return reference(array, x + i, y + j); }
};

// flat view of 2D array (operator[] returns scalar)
class flat_view : public view {
  using preview::array;
  using preview::x;
  using preview::y;
  using preview::nx;
  using preview::ny;
public:
  // construction--perform shallow copy of (sub)array
  flat_view(array2* array) : view(array) {}
  flat_view(array2* array, uint x, uint y, uint nx, uint ny) : view(array, x, y, nx, ny) {}

  // convert (i, j) index to flat index
  uint index(uint i, uint j) const { return i + nx * j; }

  // convert flat index to (i, j) index
  void ij(uint& i, uint& j, uint index) const
  {
    i = index % nx; index /= nx;
    j = index;
  }

  // flat index accessors
  Scalar operator[](uint index) const
  {
    uint i, j;
    ij(i, j, index);
    return array->get(x + i, y + j);
  }
  reference operator[](uint index)
  {
    uint i, j;
    ij(i, j, index);
    return reference(array, x + i, y + j);
  }
};

// forward declaration of friends
class nested_view1;
class nested_view2;

// nested view into a 1D rectangular subset of a 2D array
class nested_view1 : public preview {
  using preview::array;
  using preview::x;
  using preview::y;
  using preview::nx;
  using preview::ny;
public:
  // dimensions of (sub)array
  uint size_x() const { return nx; }

  // [i] accessor and mutator
  Scalar operator[](uint index) const { return array->get(x + index, y); }
  reference operator[](uint index) { return reference(array, x + index, y); }

  // (i) accessor and mutator
  Scalar operator()(uint i) const { return array->get(x + i, y); }
  reference operator()(uint i) { return reference(array, x + i, y); }

protected:
  // construction--perform shallow copy of (sub)array
  friend class array2::nested_view2;
  explicit nested_view1(array2* array) : preview(array) {}
  explicit nested_view1(array2* array, uint x, uint y, uint nx, uint ny) : preview(array, x, y, nx, ny) {}
};

// nested view into a 2D rectangular subset of a 2D array
class nested_view2 : public preview {
  using preview::array;
  using preview::x;
  using preview::y;
  using preview::nx;
  using preview::ny;
public:
  // construction--perform shallow copy of (sub)array
  nested_view2(array2* array) : preview(array) {}
  nested_view2(array2* array, uint x, uint y, uint nx, uint ny) : preview(array, x, y, nx, ny) {}

  // dimensions of (sub)array
  uint size_x() const { return nx; }
  uint size_y() const { return ny; }

  // 1D view
  nested_view1 operator[](uint index) const { return nested_view1(array, x, y + index, nx, 1); }

  // (i, j) accessor and mutator
  Scalar operator()(uint i, uint j) const { return array->get(x + i, y + j); }
  reference operator()(uint i, uint j) { return reference(array, x + i, y + j); }
};

typedef nested_view2 nested_view;
