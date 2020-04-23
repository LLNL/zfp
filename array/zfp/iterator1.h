// random access const iterator that visits 1D array or view block by block; this class is nested within container_type
class const_iterator : public const_handle {
public:
  // typedefs for STL compatibility
  typedef typename container_type::value_type value_type;
  typedef ptrdiff_t difference_type;
  typedef typename container_type::reference reference;
  typedef typename container_type::pointer pointer;
  typedef std::random_access_iterator_tag iterator_category;

  typedef typename container_type::const_reference const_reference;
  typedef typename container_type::const_pointer const_pointer;

  // default constructor
  const_iterator() : const_handle(0, 0) {}

  // constructor
  explicit const_iterator(container_type* container, uint i) : const_handle(container, i) {}

  // dereference iterator
  const_reference operator*() const { return const_reference(container, i()); }
  const_reference operator[](difference_type d) const { return const_reference(container, i() + d); }

  // iterator arithmetic
  const_iterator operator+(difference_type d) const { return const_iterator(container, i() + d); }
  const_iterator operator-(difference_type d) const { return const_iterator(container, i() - d); }
  difference_type operator-(const const_iterator& it) const { return offset() - it.offset(); }

  // equality operators
  bool operator==(const const_iterator& it) const { return container == it.container && i() == it.i(); }
  bool operator!=(const const_iterator& it) const { return !operator==(it); }

  // relational operators
  bool operator<=(const const_iterator& it) const { return container == it.container && i() <= it.i(); }
  bool operator>=(const const_iterator& it) const { return container == it.container && i() >= it.i(); }
  bool operator<(const const_iterator& it) const { return container == it.container && i() < it.i(); }
  bool operator>(const const_iterator& it) const { return container == it.container && i() > it.i(); }

  // increment and decrement
  const_iterator& operator++() { increment(); return *this; }
  const_iterator& operator--() { decrement(); return *this; }
  const_iterator operator++(int) { const_iterator it = *this; increment(); return it; }
  const_iterator operator--(int) { const_iterator it = *this; decrement(); return it; }
  const_iterator operator+=(difference_type d) { advance(+d); return *this; }
  const_iterator operator-=(difference_type d) { advance(-d); return *this; }

  // container index of value referenced by iterator
  uint i() const { return const_handle::i; }

protected:
  difference_type offset() const { return static_cast<difference_type>(i()); }
  void advance(difference_type d) { const_handle::i += d; }
  void increment() { ++const_handle::i; }
  void decrement() { --const_handle::i; }

  using const_handle::container;
};

// random access iterator that visits 1D array or view block by block; this class is nested within container_type
class iterator : public const_iterator {
public:
  // typedefs for STL compatibility
  typedef typename container_type::value_type value_type;
  typedef ptrdiff_t difference_type;
  typedef typename container_type::reference reference;
  typedef typename container_type::pointer pointer;
  typedef std::random_access_iterator_tag iterator_category;

  // default constructor
  iterator() : const_iterator(0, 0) {}

  // constructor
  explicit iterator(container_type* container, uint i) : const_iterator(container, i) {}

  // dereference iterator
  reference operator*() const { return reference(container, i()); }
  reference operator[](difference_type d) const { return reference(container, i() + d); }

  // iterator arithmetic
  iterator operator+(difference_type d) const { return iterator(container, i() + d); }
  iterator operator-(difference_type d) const { return iterator(container, i() - d); }
  difference_type operator-(const iterator& it) const { return offset() - it.offset(); }

  // equality operators
  bool operator==(const iterator& it) const { return container == it.container && i() == it.i(); }
  bool operator!=(const iterator& it) const { return !operator==(it); }

  // relational operators
  bool operator<=(const iterator& it) const { return container == it.container && i() <= it.i(); }
  bool operator>=(const iterator& it) const { return container == it.container && i() >= it.i(); }
  bool operator<(const iterator& it) const { return container == it.container && i() < it.i(); }
  bool operator>(const iterator& it) const { return container == it.container && i() > it.i(); }

  // increment and decrement
  iterator& operator++() { increment(); return *this; }
  iterator& operator--() { decrement(); return *this; }
  iterator operator++(int) { iterator it = *this; increment(); return it; }
  iterator operator--(int) { iterator it = *this; decrement(); return it; }
  iterator operator+=(difference_type d) { advance(+d); return *this; }
  iterator operator-=(difference_type d) { advance(-d); return *this; }

  using const_iterator::i;

protected:
  using const_iterator::offset;
  using const_iterator::advance;
  using const_iterator::increment;
  using const_iterator::decrement;
  using const_iterator::container;
};
