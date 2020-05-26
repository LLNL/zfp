// forward const iterator that visits 2D array or view block by block; this class is nested within container_type
class const_iterator : public const_handle {
public:
  // typedefs for STL compatibility
  typedef typename container_type::value_type value_type;
  typedef ptrdiff_t difference_type;
  typedef typename container_type::reference reference;
  typedef typename container_type::pointer pointer;
  typedef std::forward_iterator_tag iterator_category;

  typedef typename container_type::const_reference const_reference;
  typedef typename container_type::const_pointer const_pointer;

  // default constructor
  const_iterator() : const_handle(0, 0, 0) {}

  // constructor
  explicit const_iterator(const container_type* container, size_t i, size_t j) : const_handle(container, i, j) {}

  // dereference iterator
  const_reference operator*() const { return const_reference(container, i(), j()); }

  // equality operators
  bool operator==(const const_iterator& it) const { return container == it.container && i() == it.i() && j() == it.j(); }
  bool operator!=(const const_iterator& it) const { return !operator==(it); }

  // increment
  const_iterator& operator++() { increment(); return *this; }
  const_iterator operator++(int) { const_iterator it = *this; increment(); return it; }

  // container indices of value referenced by iterator
  size_t i() const { return const_handle::i; }
  size_t j() const { return const_handle::j; }

protected:
  void increment()
  {
    ++const_handle::i;
    if (!(const_handle::i & 3u) || const_handle::i == container->size_x()) {
      const_handle::i = (const_handle::i - 1) & ~size_t(3);
      ++const_handle::j;
      if (!(const_handle::j & 3u) || const_handle::j == container->size_y()) {
        const_handle::j = (const_handle::j - 1) & ~size_t(3);
        // done with block; advance to next
        const_handle::i += 4;
        if (const_handle::i >= container->size_x()) {
          const_handle::i = 0;
          const_handle::j += 4;
          if (const_handle::j >= container->size_y())
            const_handle::j = container->size_y();
        }
      }
    }
  }

  using const_handle::container;
};

// forward iterator that visits 2D array or view block by block; this class is nested within container_type
class iterator : public const_iterator {
public:
  // typedefs for STL compatibility
  typedef typename container_type::value_type value_type;
  typedef ptrdiff_t difference_type;
  typedef typename container_type::reference reference;
  typedef typename container_type::pointer pointer;
  typedef std::forward_iterator_tag iterator_category;

  // default constructor
  iterator() : const_iterator(0, 0, 0) {}

  // constructor
  explicit iterator(container_type* container, size_t i, size_t j) : const_iterator(container, i, j) {}

  // dereference iterator
  reference operator*() const { return reference(container, i(), j()); }

  // equality operators
  bool operator==(const iterator& it) const { return container == it.container && i() == it.i() && j() == it.j(); }
  bool operator!=(const iterator& it) const { return !operator==(it); }

  // increment
  iterator& operator++() { increment(); return *this; }
  iterator operator++(int) { iterator it = *this; increment(); return it; }

  using const_iterator::i;
  using const_iterator::j;

protected:
  using const_iterator::increment;
  using const_iterator::container;
};
