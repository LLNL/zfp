// const pointer to a 3D array or view element; this class is nested within container_type
class const_pointer : public const_handle {
public:
  // default constructor
  const_pointer() : const_handle(0, 0, 0, 0) {}

  // constructor
  explicit const_pointer(const container_type* container, size_t x, size_t y, size_t z) : const_handle(container, x, y, z) {}

  // dereference pointer
  const_reference operator*() const { return const_reference(container, x, y, z); }
  const_reference operator[](ptrdiff_t d) const { return *operator+(d); }

  // pointer arithmetic
  const_pointer operator+(ptrdiff_t d) const { const_pointer p = *this; p.advance(d); return p; }
  const_pointer operator-(ptrdiff_t d) const { return operator+(-d); }
  ptrdiff_t operator-(const const_pointer& p) const { return offset() - p.offset(); }

  // equality operators
  bool operator==(const const_pointer& p) const { return container == p.container && x == p.x && y == p.y && z == p.z; }
  bool operator!=(const const_pointer& p) const { return !operator==(p); }

  // relational operators
  bool operator<=(const const_pointer& p) const { return container == p.container && offset() <= p.offset(); }
  bool operator>=(const const_pointer& p) const { return container == p.container && offset() >= p.offset(); }
  bool operator<(const const_pointer& p) const { return container == p.container && offset() < p.offset(); }
  bool operator>(const const_pointer& p) const { return container == p.container && offset() > p.offset(); }

  // increment and decrement
  const_pointer& operator++() { increment(); return *this; }
  const_pointer& operator--() { decrement(); return *this; }
  const_pointer operator++(int) { const_pointer p = *this; increment(); return p; }
  const_pointer operator--(int) { const_pointer p = *this; decrement(); return p; }
  const_pointer operator+=(ptrdiff_t d) { advance(+d); return *this; }
  const_pointer operator-=(ptrdiff_t d) { advance(-d); return *this; }

protected:
  ptrdiff_t offset(ptrdiff_t d = 0) const { return static_cast<ptrdiff_t>(x - container->min_x() + container->size_x() * (y - container->min_y() + container->size_y() * (z - container->min_z()))) + d; }
  void index(size_t& x, size_t& y, size_t& z, ptrdiff_t p) const
  {
    x = container->min_x() + p % container->size_x(); p /= container->size_x();
    y = container->min_y() + p % container->size_y(); p /= container->size_y();
    z = container->min_z() + p;
  }
  void advance(ptrdiff_t d) { index(x, y, z, offset(d)); }
  void increment()
  {
    if (++x == container->max_x()) {
      x = container->min_x();
      if (++y == container->max_y()) {
        y = container->min_y();
        ++z;
      }
    }
  }
  void decrement()
  {
    if (x-- == container->min_x()) {
      x += container->size_x();
      if (y-- == container->min_y()) {
        y += container->size_y();
        --z;
      }
    }
  }

  using const_handle::container;
  using const_handle::x;
  using const_handle::y;
  using const_handle::z;
};

// pointer to a 3D array or view element; this class is nested within container_type
class pointer : public const_pointer {
public:
  // default constructor
  pointer() : const_pointer(0, 0, 0, 0) {}

  // constructor
  explicit pointer(container_type* container, size_t x, size_t y, size_t z) : const_pointer(container, x, y, z) {}

  // dereference pointer
  reference operator*() const { return reference(container, x, y, z); }
  reference operator[](ptrdiff_t d) const { return *operator+(d); }

  // pointer arithmetic
  pointer operator+(ptrdiff_t d) const { pointer p = *this; p.advance(d); return p; }
  pointer operator-(ptrdiff_t d) const { return operator+(-d); }
  ptrdiff_t operator-(const pointer& p) const { return offset() - p.offset(); }

  // equality operators
  bool operator==(const pointer& p) const { return container == p.container && x == p.x && y == p.y && z == p.z; }
  bool operator!=(const pointer& p) const { return !operator==(p); }

  // relational operators
  bool operator<=(const pointer& p) const { return container == p.container && offset() <= p.offset(); }
  bool operator>=(const pointer& p) const { return container == p.container && offset() >= p.offset(); }
  bool operator<(const pointer& p) const { return container == p.container && offset() < p.offset(); }
  bool operator>(const pointer& p) const { return container == p.container && offset() > p.offset(); }

  // increment and decrement
  pointer& operator++() { increment(); return *this; }
  pointer& operator--() { decrement(); return *this; }
  pointer operator++(int) { pointer p = *this; increment(); return p; }
  pointer operator--(int) { pointer p = *this; decrement(); return p; }
  pointer operator+=(ptrdiff_t d) { advance(+d); return *this; }
  pointer operator-=(ptrdiff_t d) { advance(-d); return *this; }

protected:
  using const_pointer::offset;
  using const_pointer::advance;
  using const_pointer::increment;
  using const_pointer::decrement;
  using const_pointer::container;
  using const_pointer::x;
  using const_pointer::y;
  using const_pointer::z;
};
