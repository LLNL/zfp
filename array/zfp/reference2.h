// const reference to a 2D array or view element; this class is nested within container_type
class const_reference : const_handle {
public:
  typedef typename container_type::value_type value_type;

  // constructor
  explicit const_reference(container_type* container, size_t i, size_t j) : const_handle(container, i, j) {}

  // inspector
  operator value_type() const { return get(); }

  // pointer to referenced element
  const_pointer operator&() const { return const_pointer(container, i, j); }

protected:
  using const_handle::get;
  using const_handle::container;
  using const_handle::i;
  using const_handle::j;
};

// reference to a 2D array or view element; this class is nested within container_type
class reference : public const_reference {
public:
  // constructor
  explicit reference(container_type* container, size_t i, size_t j) : const_reference(container, i, j) {}

  // assignment
  reference operator=(const reference& r) { set(r.get()); return *this; }
  reference operator=(value_type val) { set(val); return *this; }

  // compound assignment
  reference operator+=(value_type val) { container->add(i, j, val); return *this; }
  reference operator-=(value_type val) { container->sub(i, j, val); return *this; }
  reference operator*=(value_type val) { container->mul(i, j, val); return *this; }
  reference operator/=(value_type val) { container->div(i, j, val); return *this; }

  // pointer to referenced element
  pointer operator&() const { return pointer(container, i, j); }

  // swap two array elements via proxy references
  friend void swap(reference a, reference b)
  {
    value_type x = a.get();
    value_type y = b.get();
    b.set(x);
    a.set(y);
  }

protected:
  // assign value through reference
  void set(value_type val) { container->set(i, j, val); }

  using const_reference::get;
  using const_reference::container;
  using const_reference::i;
  using const_reference::j;
};
