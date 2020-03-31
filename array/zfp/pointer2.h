// const pointer to a 2D array or view element; this class is nested within container_type
class const_pointer : public const_handle {
public:
  // default constructor
  const_pointer() : const_handle(0, 0, 0) {}

  // constructor
  explicit const_pointer(container_type* container, uint i, uint j) : const_handle(container, i, j) {}

  // dereference pointer
  const_reference operator*() const { return const_reference(container, i, j); }
  const_reference operator[](ptrdiff_t d) const { const_pointer p = operator+(d); return *p; }

  // pointer arithmetic
  const_pointer operator+(ptrdiff_t d) const { const_pointer p = *this; p.advance(+d); return p; }
  const_pointer operator-(ptrdiff_t d) const { const_pointer p = *this; p.advance(-d); return p; }
  ptrdiff_t operator-(const const_pointer& p) const { return offset() - p.offset(); }

  // equality operators
  bool operator==(const const_pointer& p) const { return container == p.container && i == p.i && j == p.j; }
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
  ptrdiff_t offset() const { return static_cast<ptrdiff_t>(i + container->size_x() * j); }
  void advance(ptrdiff_t d)
  {
    size_t idx = offset() + d;
    i = static_cast<uint>(idx % container->size_x());
    j = static_cast<uint>(idx / container->size_x());
  }
  void increment()
  {
    if (++i == container->size_x()) {
      i = 0;
      ++j;
    }
  }
  void decrement()
  {
    if (!i--) {
      i += container->size_x();
      --j;
    }
  }

  using const_handle::container;
  using const_handle::i;
  using const_handle::j;
};

// pointer to a 2D array or view element; this class is nested within container_type
class pointer : public const_pointer {
public:
  // default constructor
  pointer() : const_pointer(0, 0, 0) {}

  // constructor
  explicit pointer(container_type* container, uint i, uint j) : const_pointer(container, i, j) {}

  // dereference pointer
  reference operator*() const { return reference(container, i, j); }
  reference operator[](ptrdiff_t d) const { pointer p = operator+(d); return *p; }

  // pointer arithmetic
  pointer operator+(ptrdiff_t d) const { pointer p = *this; p.advance(+d); return p; }
  pointer operator-(ptrdiff_t d) const { pointer p = *this; p.advance(-d); return p; }
  ptrdiff_t operator-(const pointer& p) const { return offset() - p.offset(); }

  // equality operators
  bool operator==(const pointer& p) const { return container == p.container && i == p.i && j == p.j; }
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

  using const_handle::container;
  using const_handle::i;
  using const_handle::j;
};
