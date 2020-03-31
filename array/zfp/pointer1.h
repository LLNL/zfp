// const pointer to a 1D array or view element; this class is nested within container_type
class const_pointer : public const_handle {
public:
  // default constructor
  const_pointer() : const_handle(0, 0) {}

  // constructor
  explicit const_pointer(container_type* container, uint i) : const_handle(container, i) {}

  // dereference pointer
  const_reference operator*() const { return const_reference(container, i); }
  const_reference operator[](ptrdiff_t d) const { return const_reference(container, i + d); }

  // pointer arithmetic
  const_pointer operator+(ptrdiff_t d) const { return const_pointer(container, i + d); }
  const_pointer operator-(ptrdiff_t d) const { return const_pointer(container, i - d); }
  ptrdiff_t operator-(const const_pointer& p) const { return offset() - p.offset(); }

  // equality operators
  bool operator==(const const_pointer& p) const { return container == p.container && i == p.i; }
  bool operator!=(const const_pointer& p) const { return !operator==(p); }

  // relational operators
  bool operator<=(const const_pointer& p) const { return container == p.container && i <= p.i; }
  bool operator>=(const const_pointer& p) const { return container == p.container && i >= p.i; }
  bool operator<(const const_pointer& p) const { return container == p.container && i < p.i; }
  bool operator>(const const_pointer& p) const { return container == p.container && i > p.i; }

  // increment and decrement
  const_pointer& operator++() { increment(); return *this; }
  const_pointer& operator--() { decrement(); return *this; }
  const_pointer operator++(int) { const_pointer p = *this; increment(); return p; }
  const_pointer operator--(int) { const_pointer p = *this; decrement(); return p; }
  const_pointer operator+=(ptrdiff_t d) { advance(+d); return *this; }
  const_pointer operator-=(ptrdiff_t d) { advance(-d); return *this; }

protected:
  ptrdiff_t offset() const { return static_cast<ptrdiff_t>(i); }
  void advance(ptrdiff_t d) { i += d; }
  void increment() { ++i; }
  void decrement() { --i; }

  using const_handle::container;
  using const_handle::i;
};

// pointer to a 1D array or view element; this class is nested within container_type
class pointer : public const_pointer {
public:
  // default constructor
  pointer() : const_pointer(0, 0) {}

  // constructor
  explicit pointer(container_type* container, uint i) : const_pointer(container, i) {}

  // dereference pointer
  reference operator*() const { return reference(container, i); }
  reference operator[](ptrdiff_t d) const { return reference(container, i + d); }

  // pointer arithmetic
  pointer operator+(ptrdiff_t d) const { return pointer(container, i + d); }
  pointer operator-(ptrdiff_t d) const { return pointer(container, i - d); }
  ptrdiff_t operator-(const pointer& p) const { return offset() - p.offset(); }

  // equality operators
  bool operator==(const pointer& p) const { return container == p.container && i == p.i; }
  bool operator!=(const pointer& p) const { return !operator==(p); }

  // relational operators
  bool operator<=(const pointer& p) const { return container == p.container && i <= p.i; }
  bool operator>=(const pointer& p) const { return container == p.container && i >= p.i; }
  bool operator<(const pointer& p) const { return container == p.container && i < p.i; }
  bool operator>(const pointer& p) const { return container == p.container && i > p.i; }

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
};
