// const pointer to a 2D array or view element; this class is nested within container_type
class const_pointer : public const_handle {
public:
  // default constructor
  const_pointer() : const_handle(0, 0, 0) {}

  // constructor
  explicit const_pointer(const container_type* container, size_t x, size_t y) : const_handle(container, x, y) {}

  // dereference pointer
  const_reference operator*() const { return const_reference(container, x, y); }
  const_reference operator[](ptrdiff_t d) const { return *operator+(d); }

  // pointer arithmetic
  const_pointer operator+(ptrdiff_t d) const { const_pointer p = *this; p.advance(d); return p; }
  const_pointer operator-(ptrdiff_t d) const { return operator+(-d); }
  ptrdiff_t operator-(const const_pointer& p) const { return offset() - p.offset(); }

  // equality operators
  bool operator==(const const_pointer& p) const { return container == p.container && x == p.x && y == p.y; }
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
  ptrdiff_t offset(ptrdiff_t d = 0) const { return static_cast<ptrdiff_t>(x - container->min_x() + container->size_x() * (y - container->min_y())) + d; }
  void index(size_t& x, size_t& y, ptrdiff_t p) const
  {
    x = container->min_x() + p % container->size_x(); p /= container->size_x();
    y = container->min_y() + p;
  }
  void advance(ptrdiff_t d) { index(x, y, offset(d)); }
  void increment()
  {
    if (++x == container->max_x()) {
      x = container->min_x();
      ++y;
    }
  }
  void decrement()
  {
    if (x-- == container->min_x()) {
      x += container->size_x();
      --y;
    }
  }

  using const_handle::container;
  using const_handle::x;
  using const_handle::y;
};

// pointer to a 2D array or view element; this class is nested within container_type
class pointer : public const_pointer {
public:
  // default constructor
  pointer() : const_pointer(0, 0, 0) {}

  // constructor
  explicit pointer(container_type* container, size_t x, size_t y) : const_pointer(container, x, y) {}

  // dereference pointer
  reference operator*() const { return reference(container, x, y); }
  reference operator[](ptrdiff_t d) const { return *operator+(d); }

  // pointer arithmetic
  pointer operator+(ptrdiff_t d) const { pointer p = *this; p.advance(d); return p; }
  pointer operator-(ptrdiff_t d) const { return operator+(-d); }
  ptrdiff_t operator-(const pointer& p) const { return offset() - p.offset(); }

  // equality operators
  bool operator==(const pointer& p) const { return container == p.container && x == p.x && y == p.y; }
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
};
