// const pointer to a 3D array or view element; this class is nested within container_type
class const_pointer : public const_handle {
public:
  // default constructor
  const_pointer() : const_handle(0, 0, 0, 0) {}

  // constructor
  explicit const_pointer(const container_type* container, size_t i, size_t j, size_t k) : const_handle(container, i, j, k) {}

  // dereference pointer
  const_reference operator*() const { return const_reference(container, i, j, k); }
  const_reference operator[](ptrdiff_t d) const { const_pointer p = operator+(d); return *p; }

  // pointer arithmetic
  const_pointer operator+(ptrdiff_t d) const { const_pointer p = *this; p.advance(+d); return p; }
  const_pointer operator-(ptrdiff_t d) const { const_pointer p = *this; p.advance(-d); return p; }
  ptrdiff_t operator-(const const_pointer& p) const { return offset() - p.offset(); }

  // equality operators
  bool operator==(const const_pointer& p) const { return container == p.container && i == p.i && j == p.j && k == p.k; }
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
  ptrdiff_t offset() const { return static_cast<ptrdiff_t>(i + container->size_x() * (j + container->size_y() * k)); }
  void advance(ptrdiff_t d)
  {
    size_t idx = offset() + d;
    i = idx % container->size_x(); idx /= container->size_x();
    j = idx % container->size_y(); idx /= container->size_y();
    k = idx;
  }
  void increment()
  {
    if (++i == container->size_x()) {
      i = 0;
      if (++j == container->size_y()) {
        j = 0;
        ++k;
      }
    }
  }
  void decrement()
  {
    if (!i--) {
      i += container->size_x();
      if (!j--) {
        j += container->size_y();
        --k;
      }
    }
  }

  using const_handle::container;
  using const_handle::i;
  using const_handle::j;
  using const_handle::k;
};

// pointer to a 3D array or view element; this class is nested within container_type
class pointer : public const_pointer {
public:
  // default constructor
  pointer() : const_pointer(0, 0, 0, 0) {}

  // constructor
  explicit pointer(container_type* container, size_t i, size_t j, size_t k) : const_pointer(container, i, j, k) {}

  // dereference pointer
  reference operator*() const { return reference(container, i, j, k); }
  reference operator[](ptrdiff_t d) const { pointer p = operator+(d); return *p; }

  // pointer arithmetic
  pointer operator+(ptrdiff_t d) const { pointer p = *this; p.advance(+d); return p; }
  pointer operator-(ptrdiff_t d) const { pointer p = *this; p.advance(-d); return p; }
  ptrdiff_t operator-(const pointer& p) const { return offset() - p.offset(); }

  // equality operators
  bool operator==(const pointer& p) const { return container == p.container && i == p.i && j == p.j && k == p.k; }
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
  using const_pointer::i;
  using const_pointer::j;
  using const_pointer::k;
};
