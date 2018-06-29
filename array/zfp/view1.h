// 1D array views; these classes are nested within zfp::array1

// abstract view of 1D array (base class)
class preview {
public:
  // rate in bits per value
  double rate() const { return array->rate(); }

  // dimensions of (sub)array
  size_t size() const { return size_t(nx); }

protected:
  // construction and assignment--perform shallow copy of (sub)array
  explicit preview(array1* array) : array(array), x(0), nx(array->nx) {}
  explicit preview(array1* array, uint x, uint nx) : array(array), x(x), nx(nx) {}
  preview& operator=(array1* a)
  {
    array = a;
    x = 0;
    nx = a->nx;
    return *this;
  }

  array1* array; // underlying container
  uint x;        // offset into array
  uint nx;       // dimensions of subarray
};

// generic read-only view into a rectangular subset of a 1D array
class const_view : public preview {
protected:
  using preview::array;
  using preview::x;
  using preview::nx;
public:
  // construction--perform shallow copy of (sub)array
  const_view(array1* array) : preview(array) {}
  const_view(array1* array, uint x, uint nx) : preview(array, x, nx) {}

  // dimensions of (sub)array
  uint size_x() const { return nx; }

  // [i] accessor
  Scalar operator[](uint index) const { return array->get(x + index); }

  // (i) accessor
  Scalar operator()(uint i) const { return array->get(x + i); }
};

// generic read-write view into a rectangular subset of a 1D array
class view : public const_view {
protected:
  using preview::array;
  using preview::x;
  using preview::nx;
public:
  // construction--perform shallow copy of (sub)array
  view(array1* array) : const_view(array) {}
  view(array1* array, uint x, uint nx) : const_view(array, x, nx) {}

  // [i] mutator
  reference operator[](uint index) { return reference(array, x + index); }

  // (i) mutator
  reference operator()(uint i) { return reference(array, x + i); }
};
